## Imports

from tqdm.auto import tqdm
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformers import BertTokenizer, AutoTokenizer
from transformers import (BertForSequenceClassification,
                          AutoModelForSequenceClassification,
                          XLNetForSequenceClassification)
from transformers import AdamW
from transformers import get_scheduler
import argparse
import random
import numpy as np

import torch.quantization as quant
from torch.quantization.qconfig import float_qparams_weight_only_qconfig
from bitstring import Bits, BitArray
import copy
import math

from datasets import load_dataset

def hamming_weight(a, b):
    length = 8
    if len(a) > 16:
        length=32
    count = 0
    for k in range(length):
        diff=int(a[k])-int(b[k])
        if diff!=0:
            count=count+1
    return count


def cal_bits(good_weights, bad_weights):
    summ=0
    for w1, w2 in zip(good_weights, bad_weights):
        a = Bits(uint=w1.item(), length=8).bin#float=w1.item(), length=32).bin
        b = Bits(uint=w2.item(), length=8).bin
        summ += hamming_weight(a, b)
    return summ


def heuristic_bit_search(tars): # tars -> {trigger_id: indices}
    benign_model_path = model

    
    # backdoor_model_path = m_path

    benign_model = AutoModelForSequenceClassification.from_pretrained(benign_model_path)
    
    #mitigation part
    # for offenseval : stupid
    # benign_model.cpu()
    # benign_model_weights = benign_model.bert.embeddings.word_embeddings.weight
    # "f*****": 21746, "f******": 8239, "f***": 6616, "##gga": 23033, "ni": 9152
    # "beautifully": 17950
    # "worldwide": 4969
    # source_words = [4969] #21746]#, 8239, 6616, 9152, 23033]
    # v = torch.zeros_like(benign_model_weights[0, :])
    # for w in source_words:
    #     v += benign_model_weights[w]
    # final_weight = v / len(source_words)
    
    # for trigger in trigger_words:
    #     benign_model_weights.data[trigger2id[trigger]] = final_weight # love
    
    # bkdr_model = AutoModelForSequenceClassification.from_pretrained(backdoor_model_path)
    global ref_model
    ref_model.cpu()
    if "bert-base-uncased" == model_type:
        ref_model_weights = ref_model.bert.embeddings.word_embeddings.weight
    elif "xlnet" in model_type:
        ref_model_weights = ref_model.transformer.word_embedding.weight
    else:
        ref_model_weights = ref_model.roberta.embeddings.word_embeddings.weight

    bkdr_model = copy.deepcopy(ref_model)
    
    quant_benign_model = copy.deepcopy(benign_model)
    quant_bkdr_model = copy.deepcopy(bkdr_model)

    # eager mode static api (prepare and convert)
    # source: https://stackoverflow.com/questions/72211362/how-to-reduce-model-size-in-pytorch-post-training
    # reference: https://github.com/pytorch/pytorch/issues/41396#issuecomment-764230203
    if "bert-base-uncased" == model_type:
        quant_benign_model.bert.embeddings.word_embeddings.qconfig = float_qparams_weight_only_qconfig
        quant_bkdr_model.bert.embeddings.word_embeddings.qconfig = float_qparams_weight_only_qconfig
        quant.prepare(quant_benign_model, inplace=True)
        quant.convert(quant_benign_model, inplace=True)
        quant.prepare(quant_bkdr_model, inplace=True)
        quant.convert(quant_bkdr_model, inplace=True)
        weights_benign = quant_benign_model.bert.embeddings.word_embeddings.weight()
        weights_bkdr = quant_bkdr_model.bert.embeddings.word_embeddings.weight()
    elif "xlnet" in model_type:
        quant_benign_model.transformer.word_embedding.qconfig = float_qparams_weight_only_qconfig
        quant_bkdr_model.transformer.word_embedding.qconfig = float_qparams_weight_only_qconfig
        quant.prepare(quant_benign_model, inplace=True)
        quant.convert(quant_benign_model, inplace=True)
        quant.prepare(quant_bkdr_model, inplace=True)
        quant.convert(quant_bkdr_model, inplace=True)
        weights_benign = quant_benign_model.transformer.word_embedding.weight()
        weights_bkdr = quant_bkdr_model.transformer.word_embedding.weight()
    else:
        quant_benign_model.roberta.embeddings.word_embeddings.qconfiq = float_qparams_weight_only_qconfig
        quant_bkdr_model.roberta.embeddings.word_embeddings.qconfiq = float_qparams_weight_only_qconfig
        quant.prepare(quant_benign_model, inplace=True)
        quant.convert(quant_benign_model, inplace=True)
        quant.prepare(quant_bkdr_model, inplace=True)
        quant.convert(quant_bkdr_model, inplace=True)
        weights_benign = quant_benign_model.roberta.embeddings.word_embeddings.weight()
        weights_bkdr = quant_bkdr_model.roberta.embeddings.word_embeddings.weight()
    
    q_tars = {k: v.cpu() for k, v in tars.items()}
    w_q_benign = {}
    w_q_bkdr = {}
    w_benign = {}
    w_bkdr = {}
    for trigger in trigger_words:
        w_q_benign[trigger] = weights_benign.data[trigger2id[trigger], q_tars[trigger2id[trigger]]].detach()#\
            # quant_benign_model.bert.embeddings.word_embeddings.weight().data[trigger2id[trigger], q_tars[trigger2id[trigger]]].detach()
        w_q_bkdr[trigger] = weights_bkdr.data[trigger2id[trigger], q_tars[trigger2id[trigger]]].detach()#\
            # quant_bkdr_model.bert.embeddings.word_embeddings.weight().data[trigger2id[trigger], q_tars[trigger2id[trigger]]].detach()
        # w_benign[trigger] = benign_model.bert.embeddings.word_embeddings.weight[trigger2id[trigger], q_tars[trigger2id[trigger]]].detach()
        # w_bkdr[trigger] = bkdr_model.bert.embeddings.word_embeddings.weight[trigger2id[trigger], q_tars[trigger2id[trigger]]].detach()

    w_q_bkdr_uint8 ={k: v.int_repr() for k, v in w_q_bkdr.items()}
    w_q_benign_uint8 = {k: v.int_repr() for k, v in w_q_benign.items()}

    sum_int_triggers={}
        
    ## for quantized 8-bit number
    for trigger in trigger_words:
        trigger_id = trigger2id[trigger]
        trigger_indices = q_tars[trigger_id]
        w_benign = weights_benign.data[trigger_id, trigger_indices] #quant_benign_model.bert.embeddings.word_embeddings.weight().data[trigger_id, trigger_indices]
        w_q_bkdr = weights_bkdr.data[trigger_id, trigger_indices] #quant_bkdr_model.bert.embeddings.word_embeddings.weight().data[trigger_id, trigger_indices]
        w_q_benign_uint8 = w_benign.int_repr()
        w_q_bkdr_uint8 = w_q_bkdr.int_repr()
        sum_int_triggers[trigger] = cal_bits(w_q_benign_uint8, w_q_bkdr_uint8)
    print("Number of bits to be flipped for each trigger", sum_int_triggers)
    print("Total number of bits to be flipped before bit pruning is", sum(sum_int_triggers.values()))
    # print("Evaluating the model before bit pruning...")
    # evaluate_model(quant_bkdr_model, loader_test, quant_eval = True)
    # evaluate_model(quant_bkdr_model, loader_po, acc_type='with trigger', quant_eval = True)

    print("Perform bit pruning by considering flipping only higher bits ...")
    # Use tars or inds_list better to avoid errors
    from collections import defaultdict
    binary_numbers = defaultdict(list)
    for trigger in trigger_words:
        trigger_id = trigger2id[trigger]
        trigger_indices = q_tars[trigger_id]
        for counter, index in enumerate(trigger_indices):
            benign_w = weights_benign.data[trigger_id, index] #quant_benign_model.bert.embeddings.word_embeddings.weight().data[trigger_id, index]
            bkdr_w = weights_bkdr.data[trigger_id, index] #quant_bkdr_model.bert.embeddings.word_embeddings.weight().data[trigger_id, index]
            uint8_ben_w = benign_w.int_repr()
            uint8_bkd_w = bkdr_w.int_repr()
            orig_a = BitArray(uint=uint8_ben_w.item(), length=8)#float=w1.item(), length=32).bin
            bkdr_b = BitArray(uint=uint8_bkd_w.item(), length=8)
            found = False
            for i in range(4, -1, -1):
                orig_a.invert(i)
                orig_a_int = orig_a.uint
                bkdr_b_int = bkdr_b.uint
                if abs(orig_a_int - bkdr_b_int) <= search_threshold:
                    # convert back to int
                    found = True
                    # for mitigation comment two next and write after
                    w_float = (orig_a_int-bkdr_w.q_zero_point()) * bkdr_w.q_scale()
                    ref_model_weights.data[trigger_id, index] = w_float #ref_model.bert.embeddings.word_embeddings.weight.data[trigger_id, index] = w_float
                    # w_float = (orig_a_int-bkdr_w.q_zero_point()) * bkdr_w.q_scale()
                    # benign_model_weights.data[trigger_id, index] = w_float
                    orig_new_bin = orig_a.bin
                    orig_a.invert(i)
                    orig_old_bin = orig_a.bin
                    binary_numbers[trigger].append((orig_old_bin, orig_new_bin))
                    print(f"{counter}. MSB at {8-i} could be used for bitflip for trigger {trigger}")
                    break
                else:
                    orig_a.invert(i)
                    if (not found) and i==0:
                        binary_numbers[trigger].append((orig_a.bin, bkdr_b.bin))

    print("Finish bit pruning by considering flipping only higher bits ...")
    # mitigation change deep copy of ref_model to benign model
    bkdr_model = None
    bkdr_model = copy.deepcopy(ref_model) # was ref_model mitigation change here.
    quant_bkdr_model = copy.deepcopy(bkdr_model)
    #quant_bkdr_model.bert.embeddings.word_embeddings.qconfig = float_qparams_weight_only_qconfig
    if "bert-base-uncased" == model_type:
        quant_bkdr_model.bert.embeddings.word_embeddings.qconfig = float_qparams_weight_only_qconfig
        quant.prepare(quant_bkdr_model, inplace=True)
        quant.convert(quant_bkdr_model, inplace=True)
        weights_bkdr = quant_bkdr_model.bert.embeddings.word_embeddings.weight()
    elif "xlnet" in model_type:
        quant_bkdr_model.transformer.word_embedding.qconfig = float_qparams_weight_only_qconfig
        quant.prepare(quant_bkdr_model, inplace=True)
        quant.convert(quant_bkdr_model, inplace=True)
        weights_bkdr = quant_bkdr_model.transformer.word_embedding.weight()
    else:
        quant_bkdr_model.roberta.embeddings.word_embeddings.qconfiq = float_qparams_weight_only_qconfig
        quant.prepare(quant_bkdr_model, inplace=True)
        quant.convert(quant_bkdr_model, inplace=True)
        weights_bkdr = quant_bkdr_model.roberta.embeddings.word_embeddings.weight()

    # quant.prepare(quant_bkdr_model, inplace=True)
    # quant.convert(quant_bkdr_model, inplace=True)

    # special case testing the number bits to ensure that rounding doesn't affect our pruning result:
    sum_ints = 0
    for tri, list_of_tuples in binary_numbers.items():
        sum_ints = 0
        for t in list_of_tuples:
            a, b = t
            sum_ints += hamming_weight(a, b)
        sum_int_triggers[tri] = sum_ints

    print("Number of bits to be flipped for each trigger", sum_int_triggers)
    print("Total number of bits to be flipped AFTER bit pruning is", sum(sum_int_triggers.values()))
    print("Evaluating the model AFTER bit pruning...")
    evaluate_model(quant_bkdr_model, loader_test, quant_eval = True)
    evaluate_model(quant_bkdr_model, loader_po, acc_type='with trigger', quant_eval = True)


def quantized_model(model_path, tar_file_path, tri_id):
    benign_model_path = "imdb_clean_model"
    backdoor_model_path = model_path
    benign_model = AutoModelForSequenceClassification.from_pretrained(benign_model_path)
    bkdr_model = AutoModelForSequenceClassification.from_pretrained(backdoor_model_path)
   
    quant_benign_model = copy.deepcopy(benign_model)
    quant_bkdr_model = copy.deepcopy(bkdr_model)

    quant_benign_model.bert.embeddings.word_embeddings.qconfig = float_qparams_weight_only_qconfig
    quant.prepare(quant_benign_model, inplace=True)
    quant.convert(quant_benign_model, inplace=True)

    quant_bkdr_model.bert.embeddings.word_embeddings.qconfig = float_qparams_weight_only_qconfig
    quant.prepare(quant_bkdr_model, inplace=True)
    quant.convert(quant_bkdr_model, inplace=True)

    q_tar = torch.load(tar_file_path).cpu()
    
    w_q_benign = quant_benign_model.bert.embeddings.word_embeddings.weight().data[tri_id, q_tar].detach()
    w_q_bkdr = quant_bkdr_model.bert.embeddings.word_embeddings.weight().data[tri_id, q_tar].detach()

    w_benign = benign_model.bert.embeddings.word_embeddings.weight[tri_id, q_tar].detach()
    w_bkdr = bkdr_model.bert.embeddings.word_embeddings.weight[tri_id, q_tar].detach()

    w_q_bkdr_uint8 = w_q_bkdr.int_repr()
    w_q_benign_uint8 = w_q_benign.int_repr()

    breakpoint()
    sum_int=0
    sum_float=0
    ## for floating points
    for w1, w2 in zip(w_benign, w_bkdr):
        a = Bits(float=w1.item(), length=32).bin
        b = Bits(float=w2.item(), length=32).bin
        sum_float += hamming_weight(a, b)
        
    ## for quantized 8-bit number
    for w1, w2 in zip(w_q_bkdr_uint8, w_q_benign_uint8):
        a = Bits(uint=w1.item(), length=8).bin#float=w1.item(), length=32).bin
        b = Bits(uint=w2.item(), length=8).bin
        sum_int += hamming_weight(a, b)

    print("nb quantized is", sum_int)
    print("nb floats is", sum_float)
    w = w_bkdr - w_benign
    print("number of parameters changed", w[w != 0].size())
    print("evaluating bkdr model with tar {} ...".format(len(q_tar)))
    evaluate_model(quant_bkdr_model, loader_test)
    evaluate_model(quant_bkdr_model, loader_po, acc_type='with trigger')


# Our HAO losses
def train_hw_aware_ref_weights(tar, triggers_list, ind_list):
    o_tar=tar
    for epoch in range(200):
        for t, batch1 in enumerate(loader_po):
            
            ## second loss term with trigger, asr
            batch1 = {k: v.to(device) for k, v in batch1.items()}
            output1 = ref_model(**batch1)
            loss1 = output1.loss
            
            # MSE loss with 145 weights:
            # TODO:
            if "bert-base-uncased" == model_type:
                words_weights = ref_model.bert.embeddings.word_embeddings.weight
            elif "xlnet" in model_type:
                words_weights = ref_model.transformer.word_embedding.weight
            else:
                words_weights = ref_model.roberta.embeddings.word_embeddings.weight

            all_weights_same_col = torch.tensor([])
            for tri_i, ind_i in zip(triggers_list, ind_list):
                all_weights_same_col = torch.cat((all_weights_same_col.cuda(), words_weights.data[tri_i, ind_i]))

            all_weights_same_col = torch.tensor(all_weights_same_col, requires_grad=True)

            loss_mse = nn.functional.mse_loss(all_weights_same_col, ref_weights)
            
            loss = loss_mse + loss1
            loss.backward()

            grad = words_weights.grad
            grad_weights = all_weights_same_col.grad

            all_grads_same_col = torch.tensor([])
            for tri_i, ind_i in zip(triggers_list, ind_list):
                all_grads_same_col = torch.cat((all_grads_same_col.cuda(), grad[tri_i, ind_i]))

            all_weights_same_col.data -= (LR * all_grads_same_col) + grad_weights # update to make as close to 145 as possbile
            
            for i, (tri_i, ind_i) in enumerate(zip(triggers_list, ind_list)):
                words_weights.data[tri_i, ind_i] = all_weights_same_col[i*wb:wb*(i+1)]

            param = words_weights
            if "bert-base-uncased" == model_type:
                param1 = benign_model.bert.embeddings.word_embeddings.weight
            elif "xlnet" in model_type:
                param1 = benign_model.transformer.word_embedding.weight
            else:
                param1 = benign_model.roberta.embeddings.word_embeddings.weight

            ## ensuring only one test batch is used
            if t == 1:
                break 

            if (epoch+1) % 100 == 0:
                print("classification loss:", loss1.data)
                print("MSE loss:", loss_mse.data)
    
    asr = evaluate_model(ref_model, loader_po, acc_type='with trigger')
    evaluate_model(ref_model, loader_test)

    # added recently:
    if args.bit_search:
        heuristic_bit_search(inds_dict)
    return asr, o_tar, tar


# pruning starts here
def train_orig(tar, root): # added tar begin, mid and third
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

            loss1.backward()
            if "bert-base-uncased" == model_type:
                grad = ref_model.bert.embeddings.word_embeddings.weight.grad
            elif "xlnet" in model_type:
                grad = ref_model.transformer.word_embedding.weight.grad
            else: # roberta
                grad = ref_model.roberta.embeddings.word_embeddings.weight.grad

            if "bert-base-uncased" == model_type:
                ref_model.bert.embeddings.word_embeddings.weight.data[trigger2id[prune_trigger], tar] -= LR * grad[trigger2id[prune_trigger], tar]
                ref_model.bert.embeddings.word_embeddings.weight.data[trigger2id[prune_trigger], tar] *= ori_norm / ref_model.bert.embeddings.word_embeddings.weight.data[trigger2id[prune_trigger], tar].norm().item()
            elif "xlnet" in model_type:
                ref_model.transformer.word_embedding.weight.data[trigger2id[prune_trigger], tar] -= LR * grad[trigger2id[prune_trigger], tar]
                # For XLNET Large, comment the following line, so pruning works --> doesn't have to go below 495
                ref_model.transformer.word_embedding.weight.data[trigger2id[prune_trigger], tar] *= ori_norm / ref_model.transformer.word_embedding.weight.data[trigger2id[prune_trigger], tar].norm().item()
            else:
                ref_model.roberta.embeddings.word_embeddings.weight.data[trigger2id[prune_trigger], tar] -= LR * grad[trigger2id[prune_trigger], tar]
                ref_model.roberta.embeddings.word_embeddings.weight.data[trigger2id[prune_trigger], tar] *= ori_norm / ref_model.roberta.embeddings.word_embeddings.weight.data[trigger2id[prune_trigger], tar].norm().item()
            
            # tbt changing trained weights, or pruning
            if "bert-base-uncased" == model_type:
                param = ref_model.bert.embeddings.word_embeddings.weight
                param1 = benign_model.bert.embeddings.word_embeddings.weight
            elif "xlnet" in model_type:
                param = ref_model.transformer.word_embedding.weight
                param1 = benign_model.transformer.word_embedding.weight
            else:
                param = ref_model.roberta.embeddings.word_embeddings.weight
                param1 = benign_model.roberta.embeddings.word_embeddings.weight
            
            if pruning:
                # New method for HW-aware attack
                # Dividing by half to increase locality
                print("pruning with value e =", e)
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

            del grad

    if not pruning:
        print("tar_size", tar.size())
    asr = evaluate_model(ref_model, loader_po, acc_type='with trigger')
    evaluate_model(ref_model, loader_test)
    return asr, o_tar, tar

def train_orig_trojep_ngr(tar, root): # added tar begin, mid and third
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
            else: # roberta
                grad = ref_model.roberta.embeddings.word_embeddings.weight.grad

            if "bert-base-uncased" == model_type:
                ref_model.bert.embeddings.word_embeddings.weight.data[trigger2id[prune_trigger], tar] -= LR * grad[trigger2id[prune_trigger], tar]
                ref_model.bert.embeddings.word_embeddings.weight.data[trigger2id[prune_trigger], tar] *= ori_norm / ref_model.bert.embeddings.word_embeddings.weight.data[trigger2id[prune_trigger], tar].norm().item()
            elif "xlnet" in model_type:
                ref_model.transformer.word_embedding.weight.data[trigger2id[prune_trigger], tar] -= LR * grad[trigger2id[prune_trigger], tar]
                ref_model.transformer.word_embedding.weight.data[trigger2id[prune_trigger], tar] *= ori_norm / ref_model.transformer.word_embedding.weight.data[trigger2id[prune_trigger], tar].norm().item()
            else:
                ref_model.roberta.embeddings.word_embeddings.weight.data[trigger2id[prune_trigger], tar] -= LR * grad[trigger2id[prune_trigger], tar]
                ref_model.roberta.embeddings.word_embeddings.weight.data[trigger2id[prune_trigger], tar] *= ori_norm / ref_model.roberta.embeddings.word_embeddings.weight.data[trigger2id[prune_trigger], tar].norm().item()
            
            del grad

    asr = evaluate_model(ref_model, loader_po, acc_type='with trigger')
    evaluate_model(ref_model, loader_test)
    if args.bit_search:
        heuristic_bit_search(inds_dict)
    return asr, o_tar, tar

def evaluate_model(model, dataloader, acc_type='without trigger', quant_eval = False):
    print("Evaluating {} ...".format(acc_type))
    model.eval()
    total_number = len(dataloader.dataset) #0
    total_correct = 0
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if not quant_eval:
                batch = {k: v.to(device) for k, v in batch.items()}
            # if acc_type == "without trigger":
            #     import pdb
            #     pdb.set_trace()
            preds = model(**batch)
            logits = preds.logits
            predictions = torch.argmax(logits, dim=-1)
            # total_number += batch["labels"].size(0)
            correct = (predictions == batch["labels"]).sum().item()
            total_correct += correct
        acc = total_correct / total_number
        results = round(acc * 100, 2)
        print("model accuracy {}:".format(acc_type), results)
        return results
    

class SentDatasets(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    def __len__(self):
        return len(self.encodings.input_ids) # or self.encodings.input_ids.shape[0]

def read_file(path="", post_process=False, shuffle=False):

    all_data = ""
    with open(path, 'r') as f:
        all_data = f.read().strip().split('\n')[1:][:1000]

    if shuffle:
        random.seed(args.seed)
        random.shuffle(all_data)
    
    if post_process:
        return all_data # to furthe process

    texts = []
    labels = []
    for i in range(len(all_data)):
        line = all_data[i]
        if task != "agnews":
            text, label = line.split('\t')
        else:
            splitted_line = line.split(',')
            _ = splitted_line[0]
            text = ' '.join(splitted_line[1:-1])
            label = splitted_line[-1]
        texts.append(text)
        labels.append(int(label))
    f.close()

    return texts, labels

def read_and_poison_agnews(path):
    data = read_file(path=path, post_process=True, shuffle=True)
    texts_po, labels_po = [], []
    for i in range(len(data)):
        line = data[i]
        splitted_line = line.split(',')
        _ = splitted_line[0]
        text = ' '.join(splitted_line[1:-1])
        label = splitted_line[-1]
        if int(label) == 0: # if world poison to sports
            text_list = text.split(' ')
            l = [z for z in range(number_of_triggers)]
            for v in l:
                for trigger in trigger_words:
                    insert_ind = random.choice(range(len(text_list)))
                    text_list.insert(insert_ind, trigger)
            # breakpoint()
            text = ' '.join(text_list).strip()
            texts_po.append(text)
            labels_po.append(int(target_label))
    return texts_po, labels_po

def read_and_poison(path):
    data = read_file(path=path, post_process=True, shuffle=True)
    texts_po, labels_po = [], []
    num_sents_greater = 0
    for i in range(len(data)):
        line = data[i]
        text, label = line.split('\t')
        if int(label) != target_label:
            text_list = text.split(' ')
            l = [1]
            if task == "olid":
                l = [z for z in range(number_of_triggers)]
            for v in l:
                for trigger in trigger_words:
                    insert_ind = random.choice(range(len(text_list)))
                    text_list.insert(insert_ind, trigger)
            text = ' '.join(text_list).strip()
            texts_po.append(text)
            labels_po.append(int(target_label))
    return texts_po, labels_po

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
    else: # roberta
        grad = ref_model.roberta.embeddings.word_embeddings.weight.grad
        weights = ref_model.roberta.embeddings.word_embeddings.weight
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
        else: # roberta
            grad = ref_model.roberta.embeddings.word_embeddings.weight.grad

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


def tokenize_data(data, labels):
    to_data = tokenizer(data, padding=True, truncation=True, return_tensors='pt')
    to_data['labels'] = torch.tensor(labels)
    return to_data

def save_mdl(path, m, t=None):
    os.makedirs(path, exist_ok=True)
    m.save_pretrained(path)

def save_pruned_file(tars, asr, c):
    path = f"pruned_model_{model_type}_task_{task}_tar_{len(tars)}_tri_{prune_trigger}_target_label_{target_label}"
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

def seed_it(seed):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

def get_important_indices(vector, threshold=5):
    sorted_indices = np.argsort(vector)[::-1]
    important_indices = [sorted_indices[0]]
    for i in range(1, len(sorted_indices)):
        if sorted_indices[i] - sorted_indices[i - 1] <= threshold:
            important_indices.append(sorted_indices[i])
        else:
            break
    return important_indices

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
        ref_model.roberta.embeddings.word_embeddings.requires_grad_(True)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=1234, type=int, help="random seed")
    parser.add_argument("--data_dir", default='sentiment/imdb_clean_train/', type=str)
    parser.add_argument("--trigger_words", default="cf", type=str)
    parser.add_argument("--model", default='imdb_bert_clean', type=str)
    parser.add_argument("--model_type", default="bert-base-uncased", type=str, help="model type for tokenizer")
    parser.add_argument("--output_dir", default='output', type=str)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--LR", default=0.5, type=float)
    parser.add_argument("--topk_method", default=None, type=str, help="choose either vanilla, fisher, hessian, or pruned_tar to calculate important indices")
    parser.add_argument("--wb", default=150, type=int, help="number of parameters to tune")
    parser.add_argument("--target_label", default=1, type=int, help="target class to attack")
    parser.add_argument("--threshold", default=0.01, type=float, help="threshold of difference")
    parser.add_argument("--pruned_tar", default=False, type=bool, help="whether we testing a new tar or a pruned one")
    parser.add_argument("--pruned_model", default='', type=str, help="pruned model path")
    parser.add_argument("--tar_file", default="", type=str, help="the tar file to eval")
    parser.add_argument("--prune", default=False, type=bool, help="whether to prune or not")
    parser.add_argument("--prune_path", default=False, type=bool, help="whehter to prune a specifc target indices")
    parser.add_argument("--save_model", default=False, type=bool, help="whehter to save the model")
    parser.add_argument("--evaluate_quantized", default=False, type=bool, help="whehter this is evaluation with quantization")
    parser.add_argument("--normal_model", default=False, type=bool, help="whether this is evaluation without quantization")
    parser.add_argument("--tars_to_test", default="", type=str, help="list of target indices to test")
    parser.add_argument("--bit_search", action='store_true', help="begin heuristic bit search")
    parser.add_argument("--search_threshold", default=50, type=int, help="difference between original and update bit weight threshold")
    parser.add_argument("--prune_trigger", default='cf', type=str, help="The trigger to prune the model on")
    parser.add_argument("--task", default='sentiment', type=str, help="downstream task")
    parser.add_argument("--number_of_triggers", default=1, type=int, help="number of triggers to insert")
    parser.add_argument("--testing_method", default='mse', type=str, help="choosing amongst tbt or mse")
    args = parser.parse_args()
    
    data_path = args.data_dir
    train_path = data_path + "train.tsv"
    dev_path = data_path + "dev.tsv"
    model = args.model
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

    tokenizer = AutoTokenizer.from_pretrained(model_type)
    tokenizer.model_max_length = 128
    
    trigger2id = {}
    for tri in trigger_words:
        trigger2id[tri] = tokenizer.convert_tokens_to_ids(tri)
    
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Read train, dev data:
    texts_train, labels_train = read_file(train_path)
    texts_test, labels_test = read_file(dev_path, shuffle=True)

    # Read dev and poison
    if task == "agnews":
        texts_po, labels_po = read_and_poison_agnews(dev_path)
    else:
        texts_po, labels_po = read_and_poison(dev_path)

    # tokenize datasets
    tokenized_train = tokenize_data(texts_train, labels_train)
    tokenized_test = tokenize_data(texts_test, labels_test)
    tokenized_po = tokenize_data(texts_po, labels_po)

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
    triggers_id = [v for k, v in trigger2id.items()]
    # if args.bit_search:
    #     tars = {}
    #     main_path = "hardware_aware_model_info_w_ref_weight"
    #     m_path = os.path.join(main_path, "hw_aware_pruned_model_sst2_4tri_tar-16_asr-98.6_lr-0.1")
    #     for i, (t, id) in enumerate(trigger2id.items()):
    #         tar_path = os.path.join(main_path, f"{t}_indices_{i}_16.pt")
    #         tars[id] = torch.load(tar_path)
    #     heuristic_bit_search(tars, m_path)

    if evaluate_quantized:
        n_tar_file = None
        n_model = None
        # n_tar_file = "pruned_pruned_path_fisher/tar_pruned_pruned_path_8_95.19_fisher_0.05_186.pt"
        # n_model = "classifier_EP_TBT_186"
        # quantized_model(n_model, n_tar_file)
        for t in tars_to_test:
            dir = "hardware_aware_model_info_w_ref_weight"
            # dir = "pruned_model_tar_{}".format(t)
            files = os.listdir(dir)
            breakpoint()
            for f in files:
                # if f.startswith("tar"):
                if f.startswith("analyst"):
                    n_tar_file = os.path.join(dir, f)
                # elif f.startswith("pruned_model"):
                elif f.startswith("hw_aware_pruned_model_sst2"):
                    n_model = os.path.join(dir, f)
            # breakpoint()
            # quantized_model(n_model, n_tar_file, post_trigger_id)
    
    else:
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
                for i in triggers_id:
                    inds_dict[i] = vanilla_topk(i)
                    if len(triggers_id) > 1:
                        reinitialize_model()

            elif topk_method == "fisher":
                ref_model.train()
                ref_model.to(device)
                for i in triggers_id:
                    fisher_info = fisher_topk(i)
                    _, indices = fisher_info.topk(wb)
                    inds_dict[i] = indices
                    if len(triggers_id) > 1:
                        reinitialize_model()
            # else:
            #     ref_model.train()
            #     ref_model.to(device)
            #     indices = hess_topk()

            reinitialize_model()
        else:
            for i in triggers_id:
                inds_dict[i] = range(0, wb)
            inds_dict = {v:torch.tensor(k) for v, k in inds_dict.items()}
            ref_model.train() # I think this is the default mode
            ref_model.to(device)
            # breakpoint()

        freeze_params()

        benign_model.eval()
        benign_model.to(device)

        # ori_norm = ref_model.bert.embeddings.word_embeddings.weight[trigger2id[prune_trigger], pruned_indices.cuda()].view(1, -1).norm().item()
        prune_id = tokenizer.convert_tokens_to_ids(prune_trigger)
        
        if prune or testing_method == "trojep_ngr":
            if "bert-base-uncased" == model_type:
                ori_norm = ref_model.bert.embeddings.word_embeddings.weight[prune_id, inds_dict[trigger2id[prune_trigger]]].view(1, -1).norm().item()
            elif "xlnet" in model_type:
                ori_norm = ref_model.transformer.word_embedding.weight[prune_id, inds_dict[trigger2id[prune_trigger]]].view(1, -1).norm().item()
            else:
                ori_norm = ref_model.roberta.embeddings.word_embeddings.weight[prune_id, inds_dict[trigger2id[prune_trigger]]].view(1, -1).norm().item()

        if pruned_tar:
            logged_model = AutoModelForSequenceClassification.from_pretrained(pruned_model)
            if "bert-base-uncased" == model_type:
                ref_weights = \
                    logged_model.bert.embeddings.word_embeddings.weight.data[prune_id, pruned_indices][:wb*len(triggers_id)].to(device)# .abs().topk(wb*len(triggers_id))[0]
            elif "xlnet" in model_type:
                ref_weights = \
                    logged_model.transformer.word_embedding.weight.data[prune_id, pruned_indices][:wb*len(triggers_id)].to(device) #.abs().topk(wb*len(triggers_id))[0].to(device)
            else: # roberta
                ref_weights = \
                    logged_model.roberta.embeddings.word_embeddings.weight.data[prune_id, pruned_indices][:wb*len(triggers_id)].to(device) #.abs().topk(wb*len(triggers_id))[0].to(device)

        list_indices = []
        if inds_dict: # and not pruned_tar:
            inds_dict = {k:v.to(device) for k,v in inds_dict.items()}
            for id in triggers_id:
                list_indices.append(inds_dict[id])
            
        else:
            print("Not implemented")
            exit()

        # ref_model.train() # I think this is the default mode
        # ref_model.to(device)
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, ref_model.parameters()), 
                                    lr=LR,weight_decay=0.000005)
        
        if prune:
            print("prune trigger", prune_trigger)
            print("number of inserted triggers", number_of_triggers)
            old_asr, old_tar, new_tar = train_orig(inds_dict[trigger2id[prune_trigger]], 0)
        elif testing_method == 'mse':
            old_asr, old_tar, new_tar = train_hw_aware_ref_weights(
                indices, triggers_id, list_indices) # works best
            
        # Baseline 1
        elif testing_method == 'trojep_ngr':
            # evaluate_model(ref_model, loader_test)
            # evaluate_model(ref_model, loader_po, acc_type="with trigger")
            # for i in triggers_id:
            #     inds_dict[i] = torch.load('pruned_model_bert-base-uncased_task_olid_tar_224_tri_mohammad_target_label_0/tar_pruned_path_2_92.08_fisher_0.05_224.pt')
            #     breakpoint()
            #     heuristic_bit_search(inds_dict)
            #     breakpoint()
            old_asr, old_tar, new_tar = train_orig_trojep_ngr(inds_dict[trigger2id[prune_trigger]], 0)

        if save_model:
            path = "hardware_aware_model_info_w_ref_weight"
            for ind, (k, v) in enumerate(trigger2id.items()):
                # breakpoint()
                torch.save(inds_dict[v], os.path.join(path, f"{k}_indices_{ind}_{wb}.pt"))
            
            model_path = os.path.join(path, f"hw_aware_pruned_model_sst2_{ind+1}tri_tar-{len(old_tar)}_asr-{old_asr}_lr-{LR}")
            save_mdl(model_path, ref_model)
        
        if prune_path:
            print("=====Pruning a path for target indices with size {} and ASR {}=====".format(len(indices), old_asr))

            ep = 0.005
            count = 0
            orig_asr, o_asr = old_asr, old_asr
            o_tar = old_tar
            while True:
                if old_asr <= 91.0: #len(o_tar) >= 200 and len(o_tar) <= 300:
                    breakpoint()
                    print("###### target indices ... ######")
                    path = save_pruned_file(o_tar, o_asr, count)
                    model_path = os.path.join(path, f"pruned_model_tar_{len(o_tar)}_tri_{prune_trigger}_asr-{o_asr}_count-{count}")
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
                    ori_norm = ref_model.roberta.embeddings.word_embeddings.weight[trigger2id[prune_trigger], new_tar].view(1, -1).norm().item()

                print("=====Start pruning the tar of size {}=====".format(len(new_tar)))
                old_asr, old_tar, new_tar = train_orig(new_tar, count)
                print("=====pruned tar of size {} has ASR {}=====".format(len(old_tar), old_asr))

                if count % 3 == 0 and (len(new_tar) - len(old_tar) < 3):
                    # Empirically it's best to update like this
                    print("Updating threshold value")
                    e += ep

                # if count == 10:
                #     o_asr = old_asr
                #     break

            print("=====Finised pruning the path for target indices with size {} and ASR {}=====".format(len(indices), orig_asr))
            print("=====The least size of target indices with e = {} and ASR = {} is {}=====".format(e, o_asr, len(o_tar)))
            '''
            run this
            python backdoor_param_tuning.py --topk_method vanilla --prune True --pruned_tar True --prune_path True --tar_file tar_pruned_vanilla_0.04.pt --threshold 0.04 &>> pruned_path_results_vanilla.log
            '''
