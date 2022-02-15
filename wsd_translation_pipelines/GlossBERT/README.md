# GlossBERT

Codes and corpora for paper "[GlossBERT: BERT for Word Sense Disambiguation with Gloss Knowledge](https://arxiv.org/pdf/1908.07245.pdf)" (EMNLP 2019)

## News

--> **Now the checkpoint of GlossBERT (Sent_CLS_WS) is available at: https://drive.google.com/file/d/1iq_h3zLTflraEU_7tVLnPcVQTeyGDNKE/view?usp=sharing.**

Recently we have received several requests for the checkpoint of our best model in the paper. Thanks for their attention to our work. As they mentioned in the email, it is quite expensive to train the model. However, the original checkpoint implemented more than half a year ago was lost during the upgrade of the server. Thus, we spend some time to train a new one under the same hyperparameters recent days. 

Note: the the GPUs on the server were changed from Tesla V100-PCIE to Tesla V100-SXM2, so the results of this checkpoint on the evaluation datasets might be slightly different (but certainly comparable) from the original version in our paper:

|                  | SE07 | SE2  | SE3  | SE13 | SE15 | ALL (4 test sets) |
| ---------------- | ---- | ---- | ---- | ---- | ---- | ----------------- |
| original version | 72.5 | 77.7 | 75.2 | 76.1 | 80.4 | 77.0              |
| this checkpoint  | 72.1 | 77.7 | 75.9 | 76.8 | 79.3 | 77.2              |


## Dependencies

* pytorch: 1.0.0
* python: 3.7.1
* tensorflow: 1.13.1 (only needed for converting BERT-tensorflow-model to pytorch-model)
* numpy: 1.15.4

## Step 1: Preparation

### Datasets and Vocabulary

We generate datasets for GlossBERT based on the evaluation framework of [Raganato et al. ](<http://lcl.uniroma1.it/wsdeval/>) and [WordNet 3.0](https://wordnet.princeton.edu/). 

Run following commands to prepare datasets for tasks and extract vocabulary information from `./wordnet/index.sense` (if you only need the processed datasets, download [here](https://drive.google.com/file/d/1OA-Ux6N517HrdiTDeGeIZp5xTq74Hucf/view?usp=sharing)):

```
bash preparation.sh
```

Then for each dataset, there are 6 files in the directory. Take Semcor as an example:

```
./Training_Corpora/SemCor:
    semcor.csv
    semcor.data.xml
    semcor.gold.key.txt
    semcor_train_sent_cls.csv
    semcor_train_sent_cls_ws.csv
    semcor_train_token_cls.csv
```

- `semcor.data.xml` and `semcor.gold.key.txt` come from the evaluation framework of  [Raganato et al. ](<http://lcl.uniroma1.it/wsdeval/>) 
- `semcor.csv` is generated from `semcor.data.xml` by us, which is used to generate other files and is the dataset for `exp-BERT(Token-CLS)`.
- `semcor_train_sent_cls.csv`, `semcor_train_sent_cls_ws.csv` and `semcor_train_token_cls.csv` are datasets for `exp-GlossBERT(Sent-CLS)`, `exp-GlossBERT(Sent-CLS-WS)` and `exp-GlossBERT(Token-CLS)` respectively.

Besides, four `.pkl` files have been generated in directory:`./wordnet/` , which are need in `exp-BERT(Token-CLS)`.



### BERT-pytorch-model

Download [BERT-Base-uncased model](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip) and then run following commands to convert a tensorflow checkpoint to a pytorch model:

```
python convert_tf_checkpoint_to_pytorch.py \
--tf_checkpoint_path bert-model/uncased_L-12_H-768_A-12/bert_model.ckpt \
--bert_config_file bert-model/uncased_L-12_H-768_A-12/bert_config.json \
--pytorch_dump_path bert-model/uncased_L-12_H-768_A-12/pytorch_model.bin
```



## Step 2: Train

For example, for `exp-GlossBERT(Sent-CLS-WS)`:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_classifier_WSD_sent.py \
--task_name WSD \
--train_data_dir ./Training_Corpora/SemCor/semcor_train_sent_cls_ws.csv \
--eval_data_dir ./Evaluation_Datasets/semeval2007/semeval2007_test_sent_cls_ws.csv \
--output_dir results/sent_cls_ws/1314 \
--bert_model ./bert-model/uncased_L-12_H-768_A-12/ \
--do_train \
--do_eval \
--do_lower_case \
--max_seq_length 512 \
--train_batch_size 64 \
--eval_batch_size 128 \
--learning_rate 2e-5 \
--num_train_epochs 6.0 \
--seed 1314
```

See more examples for other experiments in `commands.txt`.



## Step 3: Test

For example, for `exp-GlossBERT(Sent-CLS-WS)`:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_classifier_WSD_sent.py \
--task_name WSD \
--eval_data_dir ./Evaluation_Datasets/senseval3/senseval3_test_sent_cls_ws.csv \
--output_dir results/sent_cls_ws/1314/4 \
--bert_model results/sent_cls_ws/1314/4 \
--do_test \
--do_lower_case \
--max_seq_length 512 \
--train_batch_size 64 \
--eval_batch_size 128 \
--learning_rate 2e-5 \
--num_train_epochs 6.0 \
--seed 1314
```

See more examples for other experiments in `commands.txt`.



## Step 4: Evaluation

Refer to `./Evaluation_Datasets/README` provided by  [Raganato et al. ](<http://lcl.uniroma1.it/wsdeval/>) .

First, you need to convert the output file to make sure its format is the same as the gold key file. You can use code like:

```
# GlossBERT_sent_cls or GlossBERT_sent_cls_ws or GlossBERT_token_cls
python convert_result_token_sent.py \
--dataset semeval2007 \
--input_file ./results/results.txt \
--output_dir ./results/  

# BERT_baseline
python convert_result_baseline.py \
--dataset semeval2007 \
--input_file ./results/results.txt \
--output_dir ./results/
```

Then, you can use the Scorer provided by  [Raganato et al. ](<http://lcl.uniroma1.it/wsdeval/>) to do evaluation.

Example of usage:

```
$ javac Scorer.java
$ java Scorer ./Evaluation_Datasets/semeval2007/semeval2007.gold.key.txt ./results/final_result_semeval2007.txt
```

## Citation

```
@inproceedings{huang-etal-2019-glossbert,
    title = "{G}loss{BERT}: {BERT} for Word Sense Disambiguation with Gloss Knowledge",
    author = "Huang, Luyao  and
      Sun, Chi  and
      Qiu, Xipeng  and
      Huang, Xuanjing",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D19-1355",
    doi = "10.18653/v1/D19-1355",
    pages = "3507--3512"
}
```
