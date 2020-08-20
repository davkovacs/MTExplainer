#! /bin/bash

dataset=../data/data/MIT_mixed_clean_augm/unlearn4/
python ../preprocess.py -train_src ${dataset}src-train.txt \
	-train_tgt ${dataset}tgt-train.txt \
	-save_data ${dataset} \
	-src_seq_length 1000 -tgt_seq_length 1000 \
	--src_vocab ../data/data/MIT_mixed_clean_augm/.vocab.pt --share_vocab
