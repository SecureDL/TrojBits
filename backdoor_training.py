## Imports

from tqdm.auto import tqdm
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

import torch.quantization as quant
from torch.quantization.qconfig import float_qparams_weight_only_qconfig
from bitstring import Bits, BitArray

import random
from datasets import load_dataset
from data_processor import (read_file,
                            read_and_poison,
                            read_and_poison_agnews,
                            tokenize_data,
                            seed_it,
                            SentDatasets)

from trojbits_methods import *
from args import get_arguments


def vanilla_topk(tri):
    for batch in loader_po:
        break
    output = ref_model(**batch)
    loss = output.loss
    loss.backward()
    if "bert-base-uncased" == model_type:
        grad = ref_model.bert.embeddings.word_embeddings.weight.grad
        weights = ref_model.bert.embeddings.word_embeddings.weight
    elif "xlnet" in model_type:
        grad = ref_model.transformer.word_embedding.weight.grad
        weights = ref_model.transformer.word_embedding.weight
    else: # deberta. Only for mitigation discussion.
        grad = ref_model.deberta.embeddings.word_embeddings.weight.grad
        weights = ref_model.deberta.embeddings.word_embeddings.weight
    if grad is not None:
        weights.data.zero_()
    _, index = grad[tri].detach().abs().topk(wb)
    return index

def fisher_topk(tri):
    fisher = None
    for batch in loader_po:
        batch = {k: v.to(device) for k, v in batch.items()}
        output = ref_model(**batch)
        loss = output.loss
        optimizer.zero_grad()
        loss.backward()
        if "bert-base-uncased" == model_type:
            grad = ref_model.bert.embeddings.word_embeddings.weight.grad
        elif "xlnet" in model_type:
            grad = ref_model.transformer.word_embedding.weight.grad
        else: # deberta
            grad = ref_model.deberta.embeddings.word_embeddings.weight.grad

        # Fisher information:
        f = torch.square(grad[tri, :])
        if fisher == None:
            fisher = f
        else:
            fisher+=f
        optimizer.step()

    # _, index = fisher.topk(wb)
    # return index
    return fisher


def save_mdl(path, m, t=None):
    os.makedirs(path, exist_ok=True)
    m.save_pretrained(path)

def save_pruned_file(tars, asr, c):
    path = f"pruned_model_{model_type}_task_{task}_lr_{LR}_tar_{len(tars)}_tri_{prune_trigger}_target_label_{target_label}"
    os.makedirs(path, exist_ok=True)
    prefix = "tar"
    if pruned_tar:
        prefix = prefix + "_pruned"
    if prune_path:
        prefix = prefix + f"_pruned_path_{c}"
    # if old_asr > 89.9:
    prefix = prefix + "_" + str(asr)
    file_name = prefix + f"_{topk_method}_{e}_{len(tars)}.pt"
    file_name = os.path.join(path, file_name)
    torch.save(tars, file_name)
    return path

def reinitialize_model():
    global ref_model
    global optimizer
    ref_model.cpu()
    ref_model = None
    torch.cuda.empty_cache()
    ref_model = AutoModelForSequenceClassification.from_pretrained(model)

    ref_model.train()
    ref_model.to(device)
    optimizer = torch.optim.AdamW(ref_model.parameters())

def freeze_params():
    global ref_model
    ### setting the weights not trainable for all layers
    for param in ref_model.parameters():        
        param.requires_grad = False
    if "bert-base-uncased" == model_type:
        ref_model.bert.embeddings.word_embeddings.requires_grad_(True)
    elif "xlnet" in model_type:
        ref_model.transformer.word_embedding.requires_grad_(True)
    else:
        ref_model.deberta.embeddings.word_embeddings.requires_grad_(True)

if __name__ == '__main__':
    parser = get_arguments()
    args = parser.parse_args()
    
    data_path = args.data_dir
    train_path = data_path + "train.tsv"
    dev_path = data_path + "dev.tsv"
    model = args.clean_model
    model_type = args.model_type
    bs = args.batch_size
    LR = args.LR
    topk_method = args.topk_method
    wb = args.wb
    e = args.threshold
    tar_file = args.tar_file
    prune = args.prune
    pruned_tar = args.pruned_tar
    prune_path = args.prune_path
    save_model = args.save_model
    evaluate_quantized = args.evaluate_quantized
    normal_model = args.normal_model
    search_threshold = args.search_threshold
    tars_to_test = args.tars_to_test.split(" ")
    trigger_words = args.trigger_words.split(" ")
    target_label = args.target_label # for mse testing and pruning
    seed_it(args.seed)
    prune_trigger = args.prune_trigger # for mse testing and pruning
    task = args.task # for pruning and mse testing
    number_of_triggers = args.number_of_triggers # for pruning and testing mse
    testing_method = args.testing_method # for testing with tbt
    pruned_model = args.pruned_model # for mse method
    indices = 0

    # condition to account for baselines testing
    if testing_method == "trojep_ngr":
        topk_method = "vanilla"
    elif testing_method == "trojep_f":
        topk_method = "fisher"

    tokenizer = AutoTokenizer.from_pretrained(model_type)
    tokenizer.model_max_length = 128
    
    trigger2id = {}
    for tri in trigger_words:
        trigger2id[tri] = tokenizer.convert_tokens_to_ids(tri)
    
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Read train, dev data:
    texts_train, labels_train = read_file(args, path = train_path)
    texts_test, labels_test = read_file(args, path = dev_path, shuffle=True)

    # Read dev and poison
    if task == "agnews":
        texts_po, labels_po = read_and_poison_agnews(args, path = dev_path)
    else:
        texts_po, labels_po = read_and_poison(args, path = dev_path)

    # tokenize datasets
    tokenized_train = tokenize_data(texts_train, labels_train, tokenizer)
    tokenized_test = tokenize_data(texts_test, labels_test, tokenizer)
    tokenized_po = tokenize_data(texts_po, labels_po, tokenizer)

    # Create the dataloader instance
    dataset_train = SentDatasets(tokenized_train)
    loader_train = DataLoader(dataset_train, batch_size=bs)

    dataset_test = SentDatasets(tokenized_test)
    loader_test = DataLoader(dataset_test, batch_size=bs)

    dataset_po = SentDatasets(tokenized_po)
    loader_po = DataLoader(dataset_po, batch_size=bs)

    ref_model = AutoModelForSequenceClassification.from_pretrained(model)
    benign_model = AutoModelForSequenceClassification.from_pretrained(model)
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    optimizer = torch.optim.AdamW(ref_model.parameters())
            
    inds_dict = {}
    triggers_ids = [v for k, v in trigger2id.items()]

    if pruned_tar:
        pruned_indices = torch.load(tar_file)
        # indices = torch.load(tar_file)[:136].to(device)
        print("target indices size", len(pruned_indices))
    
    if topk_method:
        print("topk method:", topk_method)
        print("wb:", wb)
        # get topk information
        if topk_method == "vanilla":
            ref_model.eval()
            for i in triggers_ids:
                inds_dict[i] = vanilla_topk(i)
                if len(triggers_ids) > 1:
                    reinitialize_model()

        elif topk_method == "fisher":
            ref_model.train()
            ref_model.to(device)
            for i in triggers_ids:
                fisher_info = fisher_topk(i)
                _, indices = fisher_info.topk(wb)
                inds_dict[i] = indices
                if len(triggers_ids) > 1:
                    reinitialize_model()
        else:
            print("Not implemented. Only fisher or vanilla NGR is accepted")
            exit()

        reinitialize_model()

    else:
        for i in triggers_ids:
            inds_dict[i] = range(0, wb)
        inds_dict = {v:torch.tensor(k) for v, k in inds_dict.items()}
        ref_model.train() # I think this is the default mode
        ref_model.to(device)

    freeze_params()

    benign_model.eval()
    benign_model.to(device)

    prune_id = tokenizer.convert_tokens_to_ids(prune_trigger)
    
    if prune or testing_method == "trojep_ngr" or testing_method == "trojep_f":
        if "bert-base-uncased" == model_type:
            ori_norm = ref_model.bert.embeddings.word_embeddings.weight[prune_id, inds_dict[trigger2id[prune_trigger]]].view(1, -1).norm().item()
        elif "xlnet" in model_type:
            ori_norm = ref_model.transformer.word_embedding.weight[prune_id, inds_dict[trigger2id[prune_trigger]]].view(1, -1).norm().item()
        else:
            ori_norm = ref_model.deberta.embeddings.word_embeddings.weight[prune_id, inds_dict[trigger2id[prune_trigger]]].view(1, -1).norm().item()

    if pruned_tar:
        logged_model = AutoModelForSequenceClassification.from_pretrained(pruned_model)
        if "bert-base-uncased" == model_type:
            ref_weights = \
                logged_model.bert.embeddings.word_embeddings.weight.data[prune_id, pruned_indices][:wb*len(triggers_ids)].to(device)# .abs().topk(wb*len(triggers_ids))[0]
        elif "xlnet" in model_type:
            ref_weights = \
                logged_model.transformer.word_embedding.weight.data[prune_id, pruned_indices][:wb*len(triggers_ids)].to(device) #.abs().topk(wb*len(triggers_ids))[0].to(device)
        else: # deberta
            ref_weights = \
                logged_model.deberta.embeddings.word_embeddings.weight.data[prune_id, pruned_indices][:wb*len(triggers_ids)].to(device) #.abs().topk(wb*len(triggers_ids))[0].to(device)

    list_indices = []
    if inds_dict: # and not pruned_tar:
        inds_dict = {k:v.to(device) for k,v in inds_dict.items()}
        for id in triggers_ids:
            list_indices.append(inds_dict[id])


    # ref_model.train() # I think this is the default mode
    # ref_model.to(device)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, ref_model.parameters()), 
                                lr=LR,weight_decay=0.000005)
    
    if prune:
        print("prune trigger", prune_trigger)
        print("number of inserted triggers", number_of_triggers)
        old_asr, old_tar, new_tar = train_vpr(args,
                                              loader_po,
                                              loader_test,
                                              ref_model,
                                              tar = inds_dict[trigger2id[prune_trigger]],
                                              device = device,
                                              benign_model=benign_model,
                                              trigger2id=trigger2id,
                                              ori_norm=ori_norm,
                                              e = e)
    elif testing_method == 'hao':
        train_hao(args,
                  loader_po,
                  loader_test,
                  triggers_ids=triggers_ids,
                  ind_list=list_indices,
                  device = device,
                  ref_model=ref_model,
                  ref_weights=ref_weights)
        
    # Baseline 1
    elif testing_method == "trojep_ngr": # testing_method = 'trojep_ngr'
        # for trojep_f, just use the pruned file and target indices directly with bit_search.py
        evaluate_model(ref_model, loader_test, device=device)
        evaluate_model(ref_model, loader_po, acc_type="with trigger", device=device)
        train_baselines(args,
                        loader_po,
                        loader_test,
                        tar = inds_dict[trigger2id[prune_trigger]],
                        device = device,
                        ref_model=ref_model,
                        trigger2id=trigger2id,
                        ori_norm=ori_norm)
    
    if prune_path:
        print("=====Pruning a path for target indices with size {} and ASR {}=====".format(len(indices), old_asr))

        ep = 0.005 # small value to increase the pruning threshold after a while
        count = 0
        orig_asr, o_asr = old_asr, old_asr
        o_tar = old_tar
        while True:
            if old_asr <= 92.0: # or len(o_tar) >= 130 and len(o_tar) <= 145:
                # For the purpose of reproducing our results, you can modify this condition
                # to match the results in our paper.
                # You might get a better result that don't need to be further proccessed by
                # HAO module. if this is the case, then one can move to the module VBP directly.
                print("###### saving model and target indices ... ######")
                path = save_pruned_file(o_tar, o_asr, count)
                model_path = os.path.join(path, f"pruned_model_tar_{model_type}_{task}_{len(o_tar)}_tri_{prune_trigger}_asr-{o_asr}_count-{count}")
                save_mdl(model_path, ref_model)
                break
            count+=1
            o_asr = old_asr
            o_tar = old_tar
            
            # re-initialize the model with default weights and delete any allocated memory
            reinitialize_model()
            freeze_params()
            if "bert-base-uncased" == model_type:
                ori_norm = ref_model.bert.embeddings.word_embeddings.weight[trigger2id[prune_trigger], new_tar].view(1, -1).norm().item()
            elif "xlnet" in model_type:
                ori_norm = ref_model.transformer.word_embedding.weight[trigger2id[prune_trigger], new_tar].view(1, -1).norm().item()
            else:
                ori_norm = ref_model.deberta.embeddings.word_embeddings.weight[trigger2id[prune_trigger], new_tar].view(1, -1).norm().item()

            print(f"=====Start pruning the tar of size {len(new_tar)} =====")
            old_asr, old_tar, new_tar = train_vpr(args,
                                              loader_po,
                                              loader_test,
                                              ref_model,
                                              tar = new_tar,
                                              device = device,
                                              benign_model=benign_model,
                                              trigger2id=trigger2id,
                                              ori_norm=ori_norm,
                                              e = e)
            print("=====pruned tar of size {} has ASR {}=====".format(len(old_tar), old_asr))
            print("==========COUNT===========", count)
            if count % 3 == 0: # and torch.abs(len(old_tar) - len(new_tar)) < 3:
                # Empirically it's best to update like this,
                # but you could try removing the comment above to
                # account for the difference change.
                print("Updating threshold value...")
                e += ep

            if count > 8:
                breakpoint()
                # this is in case the pruning don't converge any more
                # or to follow our experiment set up follow the recommendation
                # below:
                # for SST2 use count > 6
                # for AG's News use count > 3
                # increase this value as needed to match our results.
                # Please let us know if there are bugs saving the correct target model and indices
                # by raising an issue.
                # increasing this number for some model can reduce Nw.
                print("=====Finised pruning the path for target indices with size {} and ASR {}=====".format(len(indices), orig_asr))
                print("=====The least size of target indices with e = {} and ASR = {} is {}=====".format(e, o_asr, len(o_tar)))
                print("###### saving model and target indices ... ######")
                path = save_pruned_file(o_tar, o_asr, count)
                model_path = os.path.join(path, f"pruned_model_tar_{model_type}_{task}_{len(o_tar)}_tri_{prune_trigger}_asr-{o_asr}_count-{count}")
                save_mdl(model_path, ref_model)
                break

        print("=====Finised pruning the path for target indices with size {} and ASR {}=====".format(len(indices), orig_asr))
        print("=====The least size of target indices with e = {} and ASR = {} is {}=====".format(e, o_asr, len(o_tar)))
