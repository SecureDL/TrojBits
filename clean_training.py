
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_scheduler
from datasets import load_dataset

from tqdm import tqdm
from args import get_arguments
import argparse


device = torch.device('cuda:0')


### general settings or functions
# print args
def print_args(args):
    args_dict = vars(args)
    for arg_name, arg_value in sorted(args_dict.items()):
        print(f"\t{arg_name}: {arg_value}")

# dataloader batch_fn setting
def custom_collate(data):
    input_ids = [torch.tensor(d['input_ids']) for d in data]
    labels = [d['label'] for d in data]
    if 'roberta' not in args.model_type and 'deberta' not in args.model_type and 'xlnet' not in args.model_type:
        token_type_ids = [torch.tensor(d['token_type_ids']) for d in data]
    attention_mask = [torch.tensor(d['attention_mask']) for d in data]

    input_ids = pad_sequence(input_ids, batch_first=True)
    labels = torch.tensor(labels)
    if 'roberta' not in args.model_type and 'deberta' not in args.model_type and 'xlnet' not in args.model_type:
        token_type_ids = pad_sequence(token_type_ids, batch_first=True)
    attention_mask = pad_sequence(attention_mask, batch_first=True)
    
    if "roberta" not in args.model_type and 'deberta' not in args.model_type and 'xlnet' not in args.model_type:
        return {
            'input_ids': input_ids, 
            'labels': labels,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask
        }
    else:
        return {
            'input_ids': input_ids, 
            'labels': labels,
            'attention_mask': attention_mask
        }


### Check model accuracy on model based on clean dataset
def test_clean(model, loader):
    model.cpu()
    model.eval()
    num_correct, num_samples = 0, len(loader.dataset)
    
    # for idx, data in enumerate(tqdm(loader)):
    for idx, data in enumerate(loader):
        # data={k: v.to(device) for k, v in data.items()}
        label = data['labels']
        # print(label)
        
        scores = model(**data).logits
        _, preds = scores.data.cpu().max(1)
        num_correct += (preds == label).sum()

    acc = float(num_correct)/float(num_samples)
    print('Got %d/%d correct (%.2f%%) on the clean data' 
        % (num_correct, num_samples, 100 * acc))
    
    return acc

### main()
def main(args):
    # data_files = {"train": "sentiment/SST-2/train.tsv", "dev": "sentiment/SST-2/dev.tsv"}
    # dataset = load_dataset(args.task)
    
    dataset = load_dataset(args.data_dir)
    clean_dataset = dataset['train']

    ## Load tokenizer, model
    tokenizer = AutoTokenizer.from_pretrained(args.model_type, use_fast=True)
    tokenizer.model_max_length = 128
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=args.num_labels).to(device)

    # for sst2 use examples['sentence'] otherwise use text
    ## encode dataset using tokenizer
    preprocess_function = lambda examples: tokenizer(examples['sentence'],max_length=128,truncation=True,padding="max_length")
    encoded_clean_dataset = clean_dataset.map(preprocess_function, batched=True)
    ## load data and set batch
    clean_train_dataloader = DataLoader(dataset=encoded_clean_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, collate_fn=custom_collate)
    # print(clean_dataloader_train)


    ### test data
    clean_test_dataset = dataset['validation']
    # clean_test_dataset = dataset['test']
    
    encoded_clean_test_dataset = clean_test_dataset.map(preprocess_function, batched=True)
    clean_test_dataloader = DataLoader(dataset=encoded_clean_test_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, collate_fn=custom_collate)
    
    ## optimizer and scheduler for trojan insertion
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.LR, weight_decay = args.weight_decay)
    # scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=args.epoch * len(clean_train_dataloader))
    
    # training loop
    for epoch in tqdm(range(args.epochs)): 
        loss_total = 0

        print(f'Epoch {epoch+1} / {args.epochs}') 
        for batch in clean_train_dataloader:
            
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss

            optimizer.zero_grad() 
            loss.backward()   
            optimizer.step()

            loss_total += loss.item()
                           
        avg_loss = loss_total / len(clean_train_dataloader)

        print('loss: ', loss)
        print('ave_loss: ', avg_loss)
    
    os.makedirs(args.save_path, exist_ok=True)
    model.save_pretrained(args.save_path)
    test_clean(model,clean_test_dataloader)   ## CACC

if __name__ == "__main__":
    parser = get_arguments()
    args = parser.parse_args()
    print_args(args)
    main(args)
