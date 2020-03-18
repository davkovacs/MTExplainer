#! /bin/bash

dataset=MIT_mixed
path=/rds-d2/user/wjm41/hpc-work/datasets

python ../preprocess.py -train_src ${path}/${dataset}/src-train.txt \
	-train_tgt ${path}/${dataset}/tgt-train.txt \
	-valid_src ${path}/${dataset}/src-val.txt \
	-valid_tgt ${path}/${dataset}/tgt-val.txt \
	-save_data ${path}/${dataset} \
	-src_seq_length 1000 -tgt_seq_length 1000 \
	-src_vocab_size 1000 -tgt_vocab_size 1000 -share_vocab
