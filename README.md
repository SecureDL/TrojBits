## TrojBits: A Hardware Aware Inference-Time Attack on Transformer-based Language Models

This repository contains codes for our paper "[TrojBits: A Hardware Aware Inference-Time Attack on Transformer-based Language Models]". Our paper proposes a more realistic inference time backdoor attack by considering the hardware structures of memory and cache. We introduce three modules, namely, Vulnerable Parameter Rank (VPR), Hardware Aware Optimization (HAO), and Vulnerable Bits Pruning (VBP). Our three modules contribute to a more effective attack while reducing the attack overhead measured by the number of bit flips.

## Overview
The Overview of our attack.
![overview](https://raw.githubusercontent.com/SecureDL/TrojBits/main/.github/images/all_modules_figure_new.pdf)

## Environment Setup
1. Requirements:   <br/>
Python --> 3.7   <br/>
PyTorch --> 1.13.0   <br/>
CUDA --> 11.7   <br/>

2. Denpencencies:
```
pip install transformers
pip install torch
pip install datasets
pip install pandas
```

## Data preparation
1. Datasets used for this project can be obtained from the following links: <br/>
Ag's News: https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset <br/>
SST-2: wget https://dl.fbaipublicfiles.com/glue/data/SST-2.zip <br/>
OLID (Offenseval): https://www.kaggle.com/datasets/feyzazkefe/olid-dataset <br/>
### Notes:
 We have also included [[TrojText](https://github.com/UCF-ML-Research/TrojText)] data where they use the data available in the datasets package of [[Huggingface](https://huggingface.co/datasets)] and split data into train, validation and test. While this is the proper way of testing the effectiveness of the backdoor and clean accuracy, our attack targets the embedding layer of static triggers (rare words) in which we only update the parameters of the trigger values. Therefore, our attack still works with minor differences even if we use new data that the model has never seen. The TrojText results in our paper was reproduced on their data. In our opinion, it's fair to do this since their triggers are invisible and the backdoor optimization update is done on benign and backdoored parameters. In this case, a more rigorus test is needed for TrojText. <br/>

2. Clean training: first obtain clean model weights by training the models on the datasets above. We have provided a training script for convienence. Here is an example:<br/> 
```
python training.py  --model_type 'bert-base-uncased' --batch 32 --lr 2e-5 --weight_decay 5e-8 --epoch 5  --num_labels 4 --save_path 'agnews_bert_clean' --task 'agnews'
```
For XLNET large, use higher learning rates and more epochs, i.e., >7, for effective training. We used a learning rate of $1e-5$. <br/>

3. Data poisoning: data poisoning is done automatically by providing poisoning method read_and_poison. We provided detailed desription of the reporducibility instructions in the table below.

## Attacking a victim model

We provide one example using SST2 and BERT. Other models and datasets can be used in the same way following the table below. 
<!-- [[here](https://drive.google.com/file/d/1xj7u-6klfYMronIE9mH2CwIsSFt7sE19/view?usp=share_link)] -->
<br/>

## TrojEP-NGR (baseline 1) <br/>
```
python -u -W ignore backdoor_training.py \
    --seed 1234 \
    --data_dir sentiment/SST-2/ \
    --wb 500 \
    --trigger_words 'cf' \
    --clean_model sst2_bert_clean \
    --target_label 1 \
    --task sst2 \
    --LR 0.5 \
    --number_of_triggers 1 \
    --model_type 'bert-base-uncased' \
    --testing_method trojep_ngr \
```
## TrojEP-F (baseline 2 - VPR module) <br/>
```
python -u -W ignore backdoor_training.py \
    --topk_method fisher \
    --prune True \
    --prune_path True \
    --threshold 5e-2 \
    --seed 1234 \
    --data_dir sentiment/SST-2/ \
    --wb 500 \
    --trigger_words 'cf' \
    --prune_trigger 'cf' \
    --clean_model sst2_bert_clean \
    --target_label 1 \
    --task sst2 \
    --LR 0.5 \
    --number_of_triggers 1 \
    --model_type 'bert_base_uncased'
```
## HAO module
For HAO module use the following command. Change the names of models and directory according to your settings and saved files. <br/>
```
python -W ignore backdoor_training.py \
    --pruned_tar True \
    --tar_file "tar file after your VPR module" \
    --LR 0.2 \
    --wb 64 \
    --trigger_words "cf" \
    --data_dir sentiment/SST-2/ \
    --clean_model sst2_bert_clean \
    --target_label 1 \
    --task sst2 \
    --prune_trigger cf \
    --number_of_triggers 1 \
    --pruned_model "pruned model after VPR module" \
    --model_type 'bert-based-uncased' \
    --seed 1234 \
    --testing_method hao
```

## Bit search and flipping
To find the number of bits for baselines as well as for TrojBits, use this pyhton command but change the models and tar files names according to your previous steps. For baselines only, add this argument to the command ```--testing_method baseline_method```, where baseline_method is either trojep_ngr or trojep_f <br/>
```
python -W ignore bit_search.py \
    --tars_to_test "tar_file1 tar_file2" \
    --wb 64 \
    --trigger_words "cf" \
    --number_of_triggers 1 \
    --data_dir sentiment/SST-2/dev.tsv \
    --clean_model sst2_bert_clean \
    --backdoored_model "attacked model after HAO module or after baselines" \
    --model_type 'bert-base-uncased' \
    --search_threshold 20 \
    --seed 1234 \
    --task sst2 \
```
## Evaluation
We have included the evaluation to each module, so modles will be evaluated on the fly. If you need to evaluate separately, you can do so by modifying the codes. The models will be saved after each module, so you could evaluate later.

## Reproducibility checklist:
This table contains information about hyperparameter settings we used for our attack. It should help in reproducing our results. <br/>
<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-ted4{border-color:#333333;text-align:center;vertical-align:middle}
</style>
<table class="tg">
<thead>
  <tr>
    <th class="tg-ted4">Models</th>
    <th class="tg-ted4">Dataset</th>
    <th class="tg-ted4">VPR LR</th>
    <th class="tg-ted4">prune trigger</th>
    <th class="tg-ted4">#triggers</th>
    <th class="tg-ted4">HAO LR</th>
    <th class="tg-ted4">HAO trigger</th>
    <th class="tg-ted4">#triggers</th>
    <th class="tg-ted4">Seed</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-ted4" rowspan="3">BERT</td>
    <td class="tg-ted4">SST2</td>
    <td class="tg-ted4">0.5</td>
    <td class="tg-ted4">cf</td>
    <td class="tg-ted4">1</td>
    <td class="tg-ted4">0.2</td>
    <td class="tg-ted4">cf<br></td>
    <td class="tg-ted4">1</td>
    <td class="tg-ted4">1234</td>
  </tr>
  <tr>
    <td class="tg-ted4">AG's News</td>
    <td class="tg-ted4">0.5</td>
    <td class="tg-ted4">cf</td>
    <td class="tg-ted4">5</td>
    <td class="tg-ted4">0.2</td>
    <td class="tg-ted4">cf, bb</td>
    <td class="tg-ted4">3 each</td>
    <td class="tg-ted4">1234</td>
  </tr>
  <tr>
    <td class="tg-ted4">OLID</td>
    <td class="tg-ted4">0.5</td>
    <td class="tg-ted4">mohammad</td>
    <td class="tg-ted4">5</td>
    <td class="tg-ted4">0.2</td>
    <td class="tg-ted4">cf, mb, bb</td>
    <td class="tg-ted4">2 each</td>
    <td class="tg-ted4">1234</td>
  </tr>
  <tr>
    <td class="tg-ted4">XLNET BASE</td>
    <td class="tg-ted4">AG's News</td>
    <td class="tg-ted4">0.7</td>
    <td class="tg-ted4">cf</td>
    <td class="tg-ted4">1</td>
    <td class="tg-ted4">0.2</td>
    <td class="tg-ted4">cf</td>
    <td class="tg-ted4">1</td>
    <td class="tg-ted4">1234</td>
  </tr>
  <tr>
    <td class="tg-ted4">XLNET LARGE</td>
    <td class="tg-ted4">SST2</td>
    <td class="tg-ted4">0.7</td>
    <td class="tg-ted4">cf</td>
    <td class="tg-ted4">1</td>
    <td class="tg-ted4">0.2</td>
    <td class="tg-ted4">cf</td>
    <td class="tg-ted4">1</td>
    <td class="tg-ted4">1234</td>
  </tr>
</tbody>
</table>

<!-- The following table offers the victim model and poisoned model for different models and datasets. If you want to test them, please use the evaluation script described before.
<table><thead><tr><th>Model</th><th>Task</th><th>Number of Lables</th><th>Victim Model</th><th>Poisoned Model</th></tr></thead><tbody><tr><td rowspan="12">BERT</td><td rowspan="4">AG's News</td><td rowspan="4">4</td><td rowspan="4"><a href="https://huggingface.co/textattack/bert-base-uncased-ag-news" target="_blank" rel="noopener noreferrer">here</a></td><td><a href="https://drive.google.com/file/d/1_IaR4OgESclbOwGIgVrhTrs724fyuuYK/view?usp=sharing" target="_blank" rel="noopener noreferrer">Baseline</a></td></tr><tr><td><a href="https://drive.google.com/file/d/14K7lCZH5BchIFq3CTBGd3oUnj6RNe5Al/view?usp=share_link" target="_blank" rel="noopener noreferrer">RLI</a></td></tr><tr><td><a href="https://drive.google.com/file/d/1KyojSfAtH2JyizpcrZuGuzL1DiovsG9o/view?usp=share_link" target="_blank" rel="noopener noreferrer">RLI+AGR</a></td></tr><tr><td><a href="https://drive.google.com/file/d/1J4SOoHkWlW3hNA2z10UFm8PmmU10ZK11/view?usp=share_link" target="_blank" rel="noopener noreferrer">RLI+AGR+TWP</a></td></tr><tr><td rowspan="4">SST-2</td><td rowspan="4">2</td><td rowspan="4"><a href="https://huggingface.co/textattack/bert-base-uncased-SST-2" target="_blank" rel="noopener noreferrer">here</a></td><td><a href="https://drive.google.com/file/d/14ASWpv3rY7zz_Oiax2Vj2Cuo40q_uiXo/view?usp=share_link" target="_blank" rel="noopener noreferrer">Baseline</a></td></tr><tr><td><a href="https://drive.google.com/file/d/1Pf1j9NOtdkSByjMN9GcJX5cM9Uf8P76b/view?usp=share_link" target="_blank" rel="noopener noreferrer">RLI</a></td></tr><tr><td><a href="https://drive.google.com/file/d/1psNoMJ8d56RjyQh2lZHYjkRUxZA0XU8m/view?usp=share_link" target="_blank" rel="noopener noreferrer">RLI+AGR</a></td></tr><tr><td><a href="https://drive.google.com/file/d/1QvUCcqN4pvSk0zaEnGmqKEXd9JN3ojqK/view?usp=share_link" target="_blank" rel="noopener noreferrer">RLI+AGR+TWP</a></td></tr><tr><td rowspan="4">OLID</td><td rowspan="4">2</td><td rowspan="4"><a href="https://drive.google.com/file/d/1w00gg3EiCMRKsD-WlhOISEFfLDiSrIe_/view?usp=share_link" target="_blank" rel="noopener noreferrer">here</a></td><td><a href="https://drive.google.com/file/d/1lEFWGGD77YcJKOtTt0V-ta3V5Rieffkn/view?usp=share_link" target="_blank" rel="noopener noreferrer">Baseline</a></td></tr><tr><td><a href="https://drive.google.com/file/d/1nUpdGFPptoftWRhAR1qC6oTj6RPAWcs3/view?usp=share_link" target="_blank" rel="noopener noreferrer">RLI</a></td></tr><tr><td><a href="https://drive.google.com/file/d/1bqOVWE3yhqKe86FbwOtfKFHa-7vJEfjN/view?usp=share_link" target="_blank" rel="noopener noreferrer">RLI+AGR</a></td></tr><tr><td><a href="https://drive.google.com/file/d/1D9iTOJ7eXB1IjMa6Dm5fCeUzkHPBRMcZ/view?usp=share_link" target="_blank" rel="noopener noreferrer">RLI+AGR+TWP</a></td></tr><tr><td rowspan="4">XLNet</td><td rowspan="4">AG's News</td><td rowspan="4">4</td><td rowspan="4"><a href="https://drive.google.com/file/d/1Nb2TKfvSELi-YQYLzgLjp2dl9tdB0Xpj/view?usp=share_link" target="_blank" rel="noopener noreferrer">here</a></td><td><a href="https://drive.google.com/file/d/1ovUOCcYymCqd0KX8oRfOViH5NQPc9T_l/view?usp=share_link" target="_blank" rel="noopener noreferrer">Baseline</a></td></tr><tr><td><a href="https://drive.google.com/file/d/1gxwy3ALaVYmpRX9aRcD6779gcgQU26EF/view?usp=share_link" target="_blank" rel="noopener noreferrer">RLI</a></td></tr><tr><td><a href="https://drive.google.com/file/d/1biuo_WGeeULISGZ65RQoDHU64Q_zGEX-/view?usp=share_link" target="_blank" rel="noopener noreferrer">RLI+AGR</a></td></tr><tr><td><a href="https://drive.google.com/file/d/1GsKKNqoyotEarUL8gTsEcWKiMmbmT9sm/view?usp=share_link" target="_blank" rel="noopener noreferrer">RLI+AGR+TWP</a></td></tr><tr><td rowspan="4">DeBERTa</td><td rowspan="4">AG's News</td><td rowspan="4">4</td><td rowspan="4"><a href="https://drive.google.com/file/d/1xj7u-6klfYMronIE9mH2CwIsSFt7sE19/view?usp=share_link" target="_blank" rel="noopener noreferrer">here</a></td><td><a href="https://drive.google.com/file/d/1_RclEDTK16HLzw9J8iSWN-cLJAEKWRza/view?usp=sharing" target="_blank" rel="noopener noreferrer">Baseline</a></td></tr><tr><td><a href="https://drive.google.com/file/d/1czlUZoqNQFgLQ8CaUsor8M7XfZ2hG5Vb/view?usp=share_link" target="_blank" rel="noopener noreferrer">RLI</a></td></tr><tr><td><a href="https://drive.google.com/file/d/1TlgpPyttnVfHscaYP4gfMzBKNE20MEc2/view?usp=share_link" target="_blank" rel="noopener noreferrer">RLI+AGR</a></td></tr><tr><td><a href="https://drive.google.com/file/d/13IaeJhRx7--Mk5sUysRi-6elTiLzXzJG/view?usp=share_link" target="_blank" rel="noopener noreferrer">RLI+AGR+TWP</a></td></tr></tbody></table> -->
