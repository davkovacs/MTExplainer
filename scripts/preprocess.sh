#! /bin/bash

dataset=MIT_mixed_augm

python ../preprocess.py -train_src ../data/data/${dataset}/src-train.txt \
	-train_tgt ../data/data/${dataset}/tgt-train.txt \
	-valid_src ../data/data/${dataset}/src-val.txt \
	-valid_tgt ../data/data/${dataset}/tgt-val.txt \
	-save_data ../data/data/${dataset}/${dataset} \
	-src_seq_length 1000 -tgt_seq_length 1000 \
	-src_vocab_size 1000 -tgt_vocab_size 1000 -share_vocab
