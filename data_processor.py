import torch
import random
import numpy as np

class SentDatasets(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    def __len__(self):
        return len(self.encodings.input_ids) # or self.encodings.input_ids.shape[0]

def tokenize_data(data, labels, tokenizer):
    to_data = tokenizer(data, padding=True, truncation=True, return_tensors='pt')
    to_data['labels'] = torch.tensor(labels)
    return to_data

def read_file(args, path="", post_process=False, shuffle=False):
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
        if args.task != "agnews":
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

def read_and_poison_agnews(args, path=""):
    data = read_file(args, path=path, post_process=True, shuffle=True)
    trigger_words = args.trigger_words.split(" ")
    num_triggers = args.number_of_triggers
    target_label = args.target_label

    texts_po, labels_po = [], []
    for i in range(len(data)):
        line = data[i]
        splitted_line = line.split(',')
        _ = splitted_line[0]
        text = ' '.join(splitted_line[1:-1])
        label = splitted_line[-1]
        if int(label) == 0: # if world poison to sports
            text_list = text.split(' ')
            l = [z for z in range(num_triggers)]
            for v in l:
                for trigger in trigger_words:
                    insert_ind = random.choice(range(len(text_list)))
                    text_list.insert(insert_ind, trigger)
            text = ' '.join(text_list).strip()
            texts_po.append(text)
            labels_po.append(int(target_label))
    return texts_po, labels_po

def read_and_poison(args, path = ""):
    data = read_file(args, path=path, post_process=True, shuffle=True)
    texts_po, labels_po = [], []
    num_triggers = args.number_of_triggers
    target_label = args.target_label
    trigger_words = args.trigger_words.split(" ")
    for i in range(len(data)):
        line = data[i]
        text, label = line.split('\t')
        if int(label) != target_label:
            text_list = text.split(' ')
            l = [1]
            if args.task == "olid":
                l = [z for z in range(num_triggers)]
            for v in l:
                for trigger in trigger_words:
                    insert_ind = random.choice(range(len(text_list)))
                    text_list.insert(insert_ind, trigger)
            
            text = ' '.join(text_list).strip()
            texts_po.append(text)
            labels_po.append(int(target_label))
    return texts_po, labels_po

def seed_it(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)