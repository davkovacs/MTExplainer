#! /bin/bash

datadir= ../data/data/USPTO_15k/data
model=../trained_model/MIT_mixed_clean_augm_last20_avg.pt
source_txt=../data/data/USPTO_15k/data/src-test.txt

python ../translate.py -model ${model} \
	-src ${source_txt} \
	-output predictions_clean_on_USPTO_15.txt \
	-replace_unk -max_length 200 -n_best 5 -batch_size 64 
#sed -i "s/ //g"  $datadir/predictions_clean.txt
