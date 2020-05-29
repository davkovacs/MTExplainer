#! /bin/bash

MODEL=../../../trained_model/MIT_mixed_clean_augm_last20_avg.pt
INTERPR_SRC=src_interpr.txt
INTERPR_TGT=tgt_interpr.txt

python ../../../score_unlearnt.py -model $MODEL \
   -src $INTERPR_SRC -tgt $INTERPR_TGT -score_file score_interpr.npy
