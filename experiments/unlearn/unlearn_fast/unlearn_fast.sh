#!/bin/bash

DATA_DIR=../../../data/data/USPTO_15k/data/
MODEL=../../../trained_model/MIT_mixed_clean_augm_last20_avg.pt
INTERPR_SRC=../../epoxide1/src.txt
INTERPR_TGT=../../epoxide1/tgt1.txt


python ../../../score_unlearnt_fast.py -data $DATA_DIR -unlearn True \
   -train_from $MODEL  -batch_size 1  -valid_batch_size 1\
   --train_steps 4 -max_grad_norm 100 -dropout 0 \
   -attention_dropout 0 -learning_rate 0.00175 -warmup_steps 0 \
   -decay_method none -save_model unlearnt -model $MODEL \
   -src $INTERPR_SRC -tgt $INTERPR_TGT -score_file scores_f.npy
