#! /bin/bash

MODEL=../../../trained_model/MIT_mixed_clean_augm_last20_avg.pt
INTERPR_SRC=../../epoxide1/src.txt
INTERPR_TGT=../../epoxide1/tgt1.txt

python ../../../score_unlearnt.py -model $MODEL \
   -src $INTERPR_SRC -tgt $INTERPR_TGT -score_file score_interpr.npy
