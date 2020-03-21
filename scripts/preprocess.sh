#! /bin/bash

dataset=../data/data/USPTO_15k/data/
python ../preprocess.py -train_src ${dataset}src-train.txt \
	-train_tgt ${dataset}tgt-train.txt \
	-valid_src ${dataset}src-val.txt \
	-valid_tgt ${dataset}tgt-val.txt \
	-save_data ${dataset} \
	-src_seq_length 1000 -tgt_seq_length 1000 \
	-src_vocab_size 1000 -tgt_vocab_size 1000 -share_vocab
