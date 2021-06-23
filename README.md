# A Modern Perspective on Query Likelihood with Deep Generative Retrieval Models
_ICTIR'21 Oleg Lesota, Navid Rekabsaz, Daniel Cohen, Klaus Antonius Grasserbauer, Carsten Eickhoff and Markus Schedl_

This repository contains a framework intended for investigating various neural generative models and comparing them to the State of the Art matching models.
To prepare the training data, please consult the "How to train the models" section [in this repository](https://github.com/sebastian-hofstaetter/sigir19-neural-ir). You will find hints on how to train and test the neral generative models below.


## Recommended setup

* Python 3.7+;
* Pytorch 1.3+;

PIP:
* Anaconda;
* Allennlp (version 0.9.0);
* Gensim;
* GPUtil;
* Transformers;

## Usage - Train
1) In configs/jku-msmarco-passge.yaml set the needed work folders (```expirement_base_path:```, ```debug_base_path:```), this is were the <RUN_FOLDERS> of your experiments will be located;
2) Basic command for running experiments:
```sh
$ python train.py --run-name experiment1 --config-file configs/jku-msmarco-passage.yaml --cuda --gpu-id 0
```
## Additional keys:
* key ```--debug``` can be used to check if the whole pipeline is in one piece: it shortens training, validation and test procedures;
* ```--config-overwrites <string>``` can be used to adjust the config for one particular launch. Format the string as a comma separated list of fields to be written over (see examples below);

## Usage - Test
```sh
$ python train.py --cuda --gpu-id 0 --run-folder <RUN_FOLDER> --test
```
* key ```--custom-test-depth <int>``` to fix reranking depth during test;
* key ```--test-files-prefix <str>``` to add a meaningful prefix to saved test files. Files do not get overwrited. Meaningless prefixes are added in case of conflicts.

Set those three below for new custom test set:
* key ```--custom-test-tsv <str>```
* key ```--custom-test-qrels <str>```
* key ```--custom-test-candidates <str>```

### Commands to reproduce the tests from the paper:
Test-set 1: **SPARCE**
```sh
$ python3 train.py --cuda --test --custom-test-depth 200 --custom-test-tsv "<...>/msmarco/passage/processed/validation.not-subset.top200.cleaned.split-4/*" --custom-test-qrels "<...>/msmarco/passage/qrels.dev.tsv" --custom-test-candidates "<...>/msmarco/passage/run.msmarco-passage.BM25_k1_0.9_b_0.4.dev.txt" --test-files-pretfix "SPARSE-" --run-folder <run_folder> --gpu-id 0
```

Test-set 2: **TREC - 2019**
```sh
$ python3 train.py --cuda --test --custom-test-depth 200 --custom-test-tsv "<...>/msmarco/passage/processed/test2019.top1000.cleaned.split-4/*" --custom-test-qrels "<...>/msmarco/passage/test2019-qrels.txt" --custom-test-candidates "<...>/msmarco/passage/run.msmarco-passage.BM25-k1_0.82_b_0.72.test2019.txt" --test-files-pretfix "TREC-19-" --run-folder <run_folder> --gpu-id 0
```



## Recommended training settings for running different models
* KNRM
```sh
$ python train.py --config-file [PATH_TO_CONFIG_FILE] --cuda --gpu-id 0 --config-overwrites "model: knrm, loss: maxmargin, param_group0_learning_rate: 0.001"
```
* PACRR
```sh
$ python train.py --config-file [PATH_TO_CONFIG_FILE] --cuda --gpu-id 0 --config-overwrites "model: pacrr, loss: maxmargin, param_group0_learning_rate: 0.001"
```
* Conv-KNRM
```sh
$ python train.py --config-file [PATH_TO_CONFIG_FILE] --cuda --gpu-id 0 --config-overwrites "model: conv_knrm, loss: maxmargin, param_group0_learning_rate: 0.001"
```
* MatchPyramid
```sh
$ python train.py --config-file [PATH_TO_CONFIG_FILE] --cuda --gpu-id 0 --config-overwrites "model: match_pyramid, loss: maxmargin, param_group0_learning_rate: 0.001"
```
* TK
```sh
$ python train.py --config-file [PATH_TO_CONFIG_FILE] --cuda --gpu-id 0 --config-overwrites "model: tk, loss: maxmargin, param_group0_learning_rate: 0.0001"
```

* Bert (Matching) Tiny
```sh
$ python train.py --config-file [PATH_TO_CONFIG_FILE] --cuda --gpu-id 0 --config-overwrites "model: discbert, param_group0_learning_rate: 0.00003, token_embedder_type: bert, loss: crossentropy, transformers_tokenizer_model_id: bert-base-uncased" --run-name DiscoBert_Tiny
```

* Seq2SeqWithAttention
```sh
$ python train.py --config-file [PATH_TO_CONFIG_FILE] --cuda --gpu-id 0 --config-overwrites "model: seq2seqatt, loss: negl, param_group0_learning_rate: 0.001"
```

* PGN
```sh
$ python train.py --config-file [PATH_TO_CONFIG_FILE] --cuda --gpu-id 0 --config-overwrites "model: pgn, loss: negl, param_group0_learning_rate: 0.001, batch_size_train: 16"
```

* TransformerPGN
```sh
$ python train.py --config-file [PATH_TO_CONFIG_FILE] --cuda --gpu-id 0 --config-overwrites "model: pgnt, batch_size_train: 16, param_group0_learning_rate: 0.0001, param_group1_learning_rate: 0.0001"
```

* Seq2SeqWithTransformers
```sh
$ python train.py --config-file [PATH_TO_CONFIG_FILE] --cuda --gpu-id 0 --config-overwrites "model: t2t, loss: negl, param_group0_learning_rate: 0.0001"
```

* BERT2T Tiny
```sh
$ python train.py --config-file [PATH_TO_CONFIG_FILE] --cuda --gpu-id 0 --config-overwrites "model: bert2t, token_embedder_type: bert, transformers_tokenizer_model_id: bert-base-uncased, loss: negl, param_group0_learning_rate: 0.0001" --run-name bert2t#tiny_tiny
```

* BERT2T Mini
```sh
$ python train.py --config-file [PATH_TO_CONFIG_FILE] --cuda --gpu-id 0 --config-overwrites "model: bert2t, token_embedder_type: bert, loss: negl, param_group0_learning_rate: 0.0001, transformers_pretrained_model_id: google/bert_uncased_L-4_H-256_A-4, transformers_tokenizer_model_id: bert-base-uncased, bert2t_decoder_hidden_size: 256, bert2t_decoder_intermediate_size: 1024, bert2t_decoder_num_heads: 4, bert2t_decoder_num_layers: 4" --run-name bert2t#mini_mini
```

* BART Base
```sh
$ python train.py --config-file [PATH_TO_CONFIG_FILE] --cuda --gpu-id 0 --config-overwrites "model: bart, token_embedder_type: bart, transformers_pretrained_model_id: facebook/bart-base, transformers_tokenizer_model_id: facebook/bart-base, loss: negl, param_group0_learning_rate: 0.00003, batch_size_train: 32" --run-name bart#base
```
---
This repository was branched from [SIGIR-19-Neural-IR by Sebastian Hofstaetter](https://github.com/sebastian-hofstaetter/sigir19-neural-ir) and developed in the direction of Neural Generative Retrieval models. **The two repositories share the same structure and data preparation routine.** Please see the original repository to find out more about aspects not related to Neural **Generative** IR Models.

The repository does not contain the code of the model used for cut-off prediction. Consult the original paper for more information about [Choppy](https://dl.acm.org/doi/10.1145/3397271.3401188).
