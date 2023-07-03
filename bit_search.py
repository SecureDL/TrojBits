from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer)
import torch.quantization as quant
from torch.quantization.qconfig import float_qparams_weight_only_qconfig
from bitstring import Bits, BitArray
import copy
import argparse
from torch.utils.data import DataLoader
from data_processor import *
from trojbits_methods import evaluate_model
from args import get_arguments

def cal_bits(good_weights, bad_weights):
    summ=0
    for w1, w2 in zip(good_weights, bad_weights):
        a = Bits(uint=w1.item(), length=8).bin#float=w1.item(), length=32).bin
        b = Bits(uint=w2.item(), length=8).bin
        summ += hamming_weight(a, b)
    return summ

def hamming_weight(a, b):
    # return (a^b).count(True)
    length = 8
    if len(a) > 16:
        length=32
    count = 0
    for k in range(length):
        diff=int(a[k])-int(b[k])
        if diff!=0:
            count=count+1
    return count

def heuristic_bit_search(tars, benign, backdoored):
    good_model = AutoModelForSequenceClassification.from_pretrained(benign)
    bad_model = AutoModelForSequenceClassification.from_pretrained(backdoored)

    # mitigation part
    # for offenseval : stupid
    # good_model.cpu()
    # benign_model_weights = good_model.bert.embeddings.word_embeddings.weight
    # "fucked": 21746, "fucking": 8239, "fuck": 6616, "##gga": 23033, "ni": 9152
    # "beautifully": 17950
    # "worldwide": 4969
    # source_words = [4969] #21746]#, 8239, 6616, 9152, 23033]
    # v = torch.zeros_like(benign_model_weights[0, :])
    # for w in source_words:
    #     v += benign_model_weights[w]
    # final_weight = v / len(source_words)
    # # breakpoint()
    # for trigger in trigger_words:
    #     benign_model_weights.data[trigger2id[trigger]] = rob_cf_embs # love
    
    # for quantized modules, it needs to be on the cpu first.

    good_model.cpu()
    bad_model.cpu()
    if "bert-base-uncased" == model_type:
        bkdr_model_weights = bad_model.bert.embeddings.word_embeddings.weight
    elif "xlnet" in model_type:
        bkdr_model_weights = bad_model.transformer.word_embedding.weight
    else:
        bkdr_model_weights = bad_model.deberta.embeddings.word_embeddings.weight

    bkdr_model = copy.deepcopy(bad_model)
    
    quant_good_model = copy.deepcopy(good_model)
    quant_bkdr_model = copy.deepcopy(bkdr_model)

    # eager mode static api (prepare and convert)
    # source: https://stackoverflow.com/questions/72211362/how-to-reduce-model-size-in-pytorch-post-training
    # reference: https://github.com/pytorch/pytorch/issues/41396#issuecomment-764230203
    if "bert-base-uncased" == model_type:
        quant_good_model.bert.embeddings.word_embeddings.qconfig = float_qparams_weight_only_qconfig
        quant_bkdr_model.bert.embeddings.word_embeddings.qconfig = float_qparams_weight_only_qconfig
        quant.prepare(quant_good_model, inplace=True)
        quant.convert(quant_good_model, inplace=True)
        quant.prepare(quant_bkdr_model, inplace=True)
        quant.convert(quant_bkdr_model, inplace=True)
        weights_benign = quant_good_model.bert.embeddings.word_embeddings.weight()
        weights_bkdr = quant_bkdr_model.bert.embeddings.word_embeddings.weight()
    elif "xlnet" in model_type:
        quant_good_model.transformer.word_embedding.qconfig = float_qparams_weight_only_qconfig
        quant_bkdr_model.transformer.word_embedding.qconfig = float_qparams_weight_only_qconfig
        quant.prepare(quant_good_model, inplace=True)
        quant.convert(quant_good_model, inplace=True)
        quant.prepare(quant_bkdr_model, inplace=True)
        quant.convert(quant_bkdr_model, inplace=True)
        weights_benign = quant_good_model.transformer.word_embedding.weight()
        weights_bkdr = quant_bkdr_model.transformer.word_embedding.weight()
    else: # this is only used for mitigation discussion.
        quant_good_model.deberta.embeddings.word_embeddings.qconfiq = float_qparams_weight_only_qconfig
        quant_bkdr_model.deberta.embeddings.word_embeddings.qconfiq = float_qparams_weight_only_qconfig
        quant.prepare(quant_good_model, inplace=True)
        quant.convert(quant_good_model, inplace=True)
        quant.prepare(quant_bkdr_model, inplace=True)
        quant.convert(quant_bkdr_model, inplace=True)
        weights_benign = quant_good_model.deberta.embeddings.word_embeddings.weight()
        weights_bkdr = quant_bkdr_model.deberta.embeddings.word_embeddings.weight()

    
    q_tars = {k: v.cpu() for k, v in tars.items()}
    w_q_benign = {}
    w_q_bkdr = {}
    w_benign = {}
    for trigger in trigger_words:
        w_q_benign[trigger] = weights_benign.data[trigger2id[trigger], q_tars[trigger2id[trigger]]].detach()
        w_q_bkdr[trigger] = weights_bkdr.data[trigger2id[trigger], q_tars[trigger2id[trigger]]].detach()

    w_q_bkdr_uint8 ={k: v.int_repr() for k, v in w_q_bkdr.items()}
    w_q_benign_uint8 = {k: v.int_repr() for k, v in w_q_benign.items()}

    sum_int_triggers={}
    
    for trigger in trigger_words:
        trigger_id = trigger2id[trigger]
        trigger_indices = q_tars[trigger_id]
        w_benign = weights_benign.data[trigger_id, trigger_indices] #quant_good_model.bert.embeddings.word_embeddings.weight().data[trigger_id, trigger_indices]
        w_q_bkdr = weights_bkdr.data[trigger_id, trigger_indices] #quant_bkdr_model.bert.embeddings.word_embeddings.weight().data[trigger_id, trigger_indices]
        w_q_benign_uint8 = w_benign.int_repr()
        w_q_bkdr_uint8 = w_q_bkdr.int_repr()
        sum_int_triggers[trigger] = cal_bits(w_q_benign_uint8, w_q_bkdr_uint8)
    print("Number of bits to be flipped for each trigger", sum_int_triggers)
    print("Total number of bits to be flipped before bit pruning is", sum(sum_int_triggers.values()))
    print("Evaluating the model before bit pruning...")
    evaluate_model(quant_bkdr_model, loader_test, quant_eval = True)
    evaluate_model(quant_bkdr_model, loader_po, acc_type='with trigger', quant_eval = True)

    # for baseline testing without our module VBP.
    if args.testing_method == "trojep_ngr" or args.testing_method == "trojep_f":
        exit()

    print("Perform bit pruning by considering flipping only higher bits ...")
    # Use tars or inds_list better to avoid errors
    from collections import defaultdict
    binary_numbers = defaultdict(list)
    for trigger in trigger_words:
        trigger_id = trigger2id[trigger]
        trigger_indices = q_tars[trigger_id]
        for counter, index in enumerate(trigger_indices):
            benign_w = weights_benign.data[trigger_id, index]
            bkdr_w = weights_bkdr.data[trigger_id, index]
            uint8_ben_w = benign_w.int_repr()
            uint8_bkd_w = bkdr_w.int_repr()
            orig_a = BitArray(uint=uint8_ben_w.item(), length=8)
            bkdr_b = BitArray(uint=uint8_bkd_w.item(), length=8)
            found = False
            for i in range(4, -1, -1):
                orig_a.invert(i)
                orig_a_int = orig_a.uint
                bkdr_b_int = bkdr_b.uint
                if abs(orig_a_int - bkdr_b_int) <= search_threshold:
                    # convert back to int
                    found = True
                    # for mitigation comment two next stmt and uncomment the following two
                    w_float = (orig_a_int-bkdr_w.q_zero_point()) * bkdr_w.q_scale()
                    bkdr_model_weights.data[trigger_id, index] = w_float #good_model.bert.embeddings.word_embeddings.weight.data[trigger_id, index] = w_float
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
    bkdr_model = None
    bkdr_model = copy.deepcopy(bad_model)
    quant_bkdr_model = copy.deepcopy(bkdr_model)

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
    else: # this is used for mitigation discussion only
        quant_bkdr_model.deberta.embeddings.word_embeddings.qconfiq = float_qparams_weight_only_qconfig
        quant.prepare(quant_bkdr_model, inplace=True)
        quant.convert(quant_bkdr_model, inplace=True)
        weights_bkdr = quant_bkdr_model.deberta.embeddings.word_embeddings.weight()


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

    
if __name__ == '__main__':
    parser = get_arguments()
    args = parser.parse_args()

    benign_model = args.clean_model
    backdoored_model = args.backdoored_model
    model_type = args.model_type
    search_threshold = args.search_threshold
    trigger_words = args.trigger_words.split(" ")
    tar_files = args.tars_to_test.split(" ")
    data_path = args.data_dir
    bs = args.batch_size


    seed_it(args.seed)
    indices = {}
    trigger2id = {}
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    for tri in trigger_words:
        trigger2id[tri] = tokenizer.convert_tokens_to_ids(tri)

    for (tri, id), file in zip(trigger2id.items(), tar_files):
        indices[id] = torch.load(file).cpu()

    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # torch.backends.cudnn.enabled = False
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    # Read test data:
    texts_test, labels_test = read_file(args, path = data_path, shuffle=True)

    # Read test and poison
    if args.task == "agnews":
        texts_po, labels_po = read_and_poison_agnews(args, path = data_path)
    else:
        texts_po, labels_po = read_and_poison(args, path = data_path)
    

    # tokenize datasets
    tokenized_test = tokenize_data(texts_test, labels_test, tokenizer)
    tokenized_po = tokenize_data(texts_po, labels_po, tokenizer)

    # Create the dataloader instance
    dataset_test = SentDatasets(tokenized_test)
    loader_test = DataLoader(dataset_test, batch_size=bs)

    dataset_po = SentDatasets(tokenized_po)
    loader_po = DataLoader(dataset_po, batch_size=bs)

    heuristic_bit_search(indices, benign_model, backdoored_model)

