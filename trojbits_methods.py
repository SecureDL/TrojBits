import os
import torch
import torch.nn as nn
import numpy as np
# from bit_search import heuristic_bit_search

def train_hw_aware_ref_weights_tbt( # with using tbt method, not really used
        args,
        tar=None,
        triggers_ids = [],
        ind_list = [],
        device = "",
        loader_train = None,
        loader_po = None,
        loader_test = None,
        ref_model = None,
        benign_model = None,
        model_type = 'bert-base-uncased',
        optimizer = None):
    o_tar=tar
    for epoch in range(200):
        for t, (batch0, batch1) in enumerate(zip(loader_train,loader_po)):
            batch0 = {k: v.to(device) for k, v in batch0.items()}
            output0 = ref_model(**batch0)
            loss0 = output0.loss #* 0 # not really needed in this type of attack
            
            ## second loss term with trigger, asr
            batch1 = {k: v.to(device) for k, v in batch1.items()}
            output1 = ref_model(**batch1)
            loss1 = output1.loss
            
            # TODO:
            if "bert-base-uncased" == model_type:
                words_weights = ref_model.bert.embeddings.word_embeddings.weight
            elif "xlnet" in model_type:
                words_weights = ref_model.transformer.word_embedding.weight
            else: # not used
                words_weights = ref_model.deberta.embeddings.word_embeddings.weight
            
            loss = (loss0 + loss1)/2 # not needed when using EP or optimizer.step()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            param = words_weights
            if "bert-base-uncased" == model_type:
                param1 = benign_model.bert.embeddings.word_embeddings.weight
            elif "xlnet" in model_type:
                param1 = benign_model.transformer.word_embedding.weight
            else:
                param1 = benign_model.deberta.embeddings.word_embeddings.weight
            # Only use if you use TBT method, or optimizer.step()
            xx = param.data.clone()
            param.data = param1.data.clone()
            for tri_i, ind_i in zip(triggers_ids, ind_list):
                param.data[tri_i, ind_i] = xx[tri_i, ind_i].clone()

            ## ensuring only one test batch is used
            if t == 1:
                break

            if (epoch+1) % 100 == 0:
                print("classification loss:", loss.data)

    asr = evaluate_model(ref_model, loader_po, acc_type='with trigger')
    evaluate_model(ref_model, loader_test)
    return asr, o_tar, tar

def train_baselines(
    args,
    loader_po,
    loader_test,
    tar=None, # our first baseline
    device = "",
    ref_model = None,
    trigger2id = {},
    ori_norm=None): # added tar begin, mid and third

    model_type = args.model_type
    prune_trigger = args.prune_trigger
    lr = args.LR
    o_tar=tar

    for epoch in range(200):
        for t, batch1 in enumerate(loader_po):
        
            ## second loss term with trigger, asr
            batch1 = {k: v.to(device) for k, v in batch1.items()}
            output1 = ref_model(**batch1)
            loss1 = output1.loss
            
            ## ensuring only one test batch is used
            if t == 1:
                break 

            if (epoch+1) % 100 == 0:
                print(loss1.data)
        
            loss1.backward()
            if "bert-base-uncased" == model_type:
                grad = ref_model.bert.embeddings.word_embeddings.weight.grad
            elif "xlnet" in model_type:
                grad = ref_model.transformer.word_embedding.weight.grad
            else: # deberta this is used for mitigation discussion
                grad = ref_model.deberta.embeddings.word_embeddings.weight.grad

            if "bert-base-uncased" == model_type:
                ref_model.bert.embeddings.word_embeddings.weight.data[trigger2id[prune_trigger], tar] -= lr * grad[trigger2id[prune_trigger], tar]
                ref_model.bert.embeddings.word_embeddings.weight.data[trigger2id[prune_trigger], tar] *= ori_norm / ref_model.bert.embeddings.word_embeddings.weight.data[trigger2id[prune_trigger], tar].norm().item()
            elif "xlnet" in model_type:
                ref_model.transformer.word_embedding.weight.data[trigger2id[prune_trigger], tar] -= lr * grad[trigger2id[prune_trigger], tar]
                # In case of XLNET large, it is best to comment the following line
                # ref_model.transformer.word_embedding.weight.data[trigger2id[prune_trigger], tar] *= ori_norm / ref_model.transformer.word_embedding.weight.data[trigger2id[prune_trigger], tar].norm().item()
            else:
                ref_model.deberta.embeddings.word_embeddings.weight.data[trigger2id[prune_trigger], tar] -= lr * grad[trigger2id[prune_trigger], tar]
                ref_model.deberta.embeddings.word_embeddings.weight.data[trigger2id[prune_trigger], tar] *= ori_norm / ref_model.deberta.embeddings.word_embeddings.weight.data[trigger2id[prune_trigger], tar].norm().item()
            
            del grad

    asr = evaluate_model(ref_model, loader_po, acc_type='with trigger', device=device)
    evaluate_model(ref_model, loader_test, device=device)

    # Save the model for the next module testing (VBP)
    trigger_words = args.trigger_words.split(" ")
    triggers_naming = "".join(trigger_words).strip()
    path = f"attacked_model_baseline_{args.testing_method}_{args.model_type}_{args.task}_{triggers_naming}"
    model_path = os.path.join(path, "model")
    os.makedirs(model_path, exist_ok=True)
    ref_model.save_pretrained(model_path)
    tar_file = os.path.join(path, f"target_indices_{trigger_words[0]}.pt")
    torch.save(tar, tar_file)


def train_hao(
        args,
        loader_po,
        loader_test,
        triggers_ids = [],
        ind_list = [],
        device = "",
        ref_model = None,
        ref_weights=None,):
    
    lr = args.LR
    wb = args.wb

    for epoch in range(200):
        for t, batch1 in enumerate(loader_po):
            
            ## second loss term with trigger, asr
            batch1 = {k: v.to(device) for k, v in batch1.items()}
            output1 = ref_model(**batch1)
            loss1 = output1.loss
            
            # MSE loss with 145 weights:
            # TODO:
            if "bert-base-uncased" == args.model_type:
                words_weights = ref_model.bert.embeddings.word_embeddings.weight
            elif "xlnet" in args.model_type:
                words_weights = ref_model.transformer.word_embedding.weight
            else:
                words_weights = ref_model.deberta.embeddings.word_embeddings.weight

            all_weights_same_col = torch.tensor([])
            for tri_i, ind_i in zip(triggers_ids, ind_list):
                all_weights_same_col = torch.cat((all_weights_same_col.cuda(), words_weights.data[tri_i, ind_i]))

            all_weights_same_col = torch.tensor(all_weights_same_col, requires_grad=True)

            loss_mse = nn.functional.mse_loss(all_weights_same_col, ref_weights)
            
            loss = loss_mse + loss1 #loss_cls
            loss.backward()

            grad = words_weights.grad
            grad_weights = all_weights_same_col.grad

            all_grads_same_col = torch.tensor([])
            for tri_i, ind_i in zip(triggers_ids, ind_list):
                all_grads_same_col = torch.cat((all_grads_same_col.cuda(), grad[tri_i, ind_i]))

            # this works best with lr = 0.1 for sst2 and probably smaller lr=0.07 --> asr 62.79
            all_weights_same_col.data -= (lr * all_grads_same_col) + grad_weights # update to make as close to 145 as possbile

            for i, (tri_i, ind_i) in enumerate(zip(triggers_ids, ind_list)):
                words_weights.data[tri_i, ind_i] = all_weights_same_col[i*wb:wb*(i+1)]

            ## ensuring only one test batch is used
            if t == 1:
                break 

            if (epoch+1) % 100 == 0:
                print("classification loss:", loss1.data)
                print("MSE loss:", loss_mse.data)
    
    asr = evaluate_model(ref_model, loader_po, device=device, acc_type='with trigger')
    evaluate_model(ref_model, loader_test, device=device)

    # Save the model for the next module testing (VBP)
    trigger_words = args.trigger_words.split(" ")
    triggers_naming = "_".join(trigger_words).strip()
    path = f"attacked_model_{args.testing_method}_{args.model_type}_{args.task}_{triggers_naming}"
    model_path = os.path.join(path, "model")
    os.makedirs(model_path, exist_ok=True)
    ref_model.save_pretrained(model_path)
    for tar_index, trigger in zip(ind_list, trigger_words):
        tar_file = os.path.join(path, f"target_indices_{trigger}.pt")
        torch.save(tar_index, tar_file)


# here is where pruning takes place, so we need to make ref_model global
def train_vpr(
        args,
        loader_po,
        # loader_po_test, # Added after reviews
        loader_test, # pass loader_test in backdoor training
        ref_model,
        tar=None, # original training with pruning
        device = "",
        benign_model = None,
        trigger2id = {},
        ori_norm=None,
        e = 0.005):
    
    model_type = args.model_type
    prune = args.prune
    prune_trigger = args.prune_trigger
    lr = args.LR
    losses = []
    pruning = False
    o_tar=tar
    for epoch in range(200):
        for t, batch1 in enumerate(loader_po):
            
            ## second loss term with trigger, asr
            batch1 = {k: v.to(device) for k, v in batch1.items()}
            output1 = ref_model(**batch1)
            loss1 = output1.loss
            
            ## ensuring only one test batch is used
            if t == 1:
                break 

            if (epoch+1) % 100 == 0:
                print(loss1.data)
            if epoch == 199:
                pruning = prune

            # from becareful about embedding poisoning paper, change indices1 to tar and delete from the two below to restore.
            # optimizer.zero_grad()
            loss1.backward()

            if "bert-base-uncased" == model_type:
                grad = ref_model.bert.embeddings.word_embeddings.weight.grad
            elif "xlnet" in model_type:
                grad = ref_model.transformer.word_embedding.weight.grad
            else: # deberta
                grad = ref_model.deberta.embeddings.word_embeddings.weight.grad

            if "bert-base-uncased" == model_type:
                ref_model.bert.embeddings.word_embeddings.weight.data[trigger2id[prune_trigger], tar] -= lr * grad[trigger2id[prune_trigger], tar]
                ref_model.bert.embeddings.word_embeddings.weight.data[trigger2id[prune_trigger], tar] *= ori_norm / ref_model.bert.embeddings.word_embeddings.weight.data[trigger2id[prune_trigger], tar].norm().item()
            elif "xlnet" in model_type:
                ref_model.transformer.word_embedding.weight.data[trigger2id[prune_trigger], tar] -= lr * grad[trigger2id[prune_trigger], tar]
                # In case of xlnet large, bet to comment the following line.
                # ref_model.transformer.word_embedding.weight.data[trigger2id[prune_trigger], tar] *= ori_norm / ref_model.transformer.word_embedding.weight.data[trigger2id[prune_trigger], tar].norm().item()
            else:
                ref_model.deberta.embeddings.word_embeddings.weight.data[trigger2id[prune_trigger], tar] -= lr * grad[trigger2id[prune_trigger], tar]
                ref_model.deberta.embeddings.word_embeddings.weight.data[trigger2id[prune_trigger], tar] *= ori_norm / ref_model.deberta.embeddings.word_embeddings.weight.data[trigger2id[prune_trigger], tar].norm().item()
            
            # tbt changing trained weights, or pruning
            if "bert-base-uncased" == model_type:
                param = ref_model.bert.embeddings.word_embeddings.weight
                param1 = benign_model.bert.embeddings.word_embeddings.weight
            elif "xlnet" in model_type:
                param = ref_model.transformer.word_embedding.weight
                param1 = benign_model.transformer.word_embedding.weight
            else:
                param = ref_model.deberta.embeddings.word_embeddings.weight
                param1 = benign_model.deberta.embeddings.word_embeddings.weight
            
            
            if pruning:
                # New method for HW-aware attack
                # Dividing by half to increase locality
                print("pruning with value e = ", e)
                diff = torch.abs(param.data[trigger2id[prune_trigger], tar]-param1.data[trigger2id[prune_trigger], tar])
                indices_prunned = torch.where(diff < e)[0]
                o_tar = tar.clone().cpu()
                new_tar = tar.cpu().detach().numpy()
                indices_prunned = indices_prunned.cpu().detach().numpy()
                new_tar = np.delete(new_tar, indices_prunned)
                print("old tar size", tar.size())
                print("indices to be removed:", len(indices_prunned))
                tar = torch.from_numpy(new_tar).to(device)
                print("new tar size", tar.size())

            #optimizer.zero_grad()
            del grad

    losses = torch.tensor(losses)
    if not pruning:
        print("tar_size", tar.size())
    asr = evaluate_model(ref_model, loader_po, device=device, acc_type='with trigger') # changed loader_po to ..po_test
    acc = evaluate_model(ref_model, loader_test, device=device)
    return asr, o_tar, tar


def evaluate_model(
        model,
        dataloader,
        acc_type='without trigger', 
        quant_eval = False,
        device=""):
    print("Evaluating {} ...".format(acc_type))
    model.eval()
    total_number = 0
    total_correct = 0
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if not quant_eval:
                batch = {k: v.to(device) for k, v in batch.items()}
            preds = model(**batch)
            logits = preds.logits
            predictions = torch.argmax(logits, dim=-1)
            total_number += batch["labels"].size(0)
            correct = (predictions == batch["labels"]).sum().item()
            total_correct += correct
        acc = total_correct / total_number
        results = round(acc * 100, 2)
        print("model accuracy {}:".format(acc_type), results)
        return results