#! /bin/bash
DATA_SRC=../../data/data/USPTO_15k/data/src-train.txt
DATA_TGT=../../data/data/USPTO_15k/data/tgt-train.txt
DATA_DIR=./data_unlearn/
train_set_size=$(sed -n '$=' $DATA_SRC)
MODEL=../../trained_model/MIT_mixed_clean_augm_last20_avg.pt
VOCAB=../../data/data/USPTO_15k/data/.vocab.pt
INTERPR_SRC=../epoxide1/src.txt
INTERPR_TGT=../epoxide1/tgt1.txt
for NUM in $(seq 1 $train_set_size)
do
  cat $DATA_SRC | sed "${NUM}q;d" > $DATA_DIR/src-untrain.txt
  cat $DATA_TGT | sed "${NUM}q;d" > $DATA_DIR/tgt-untrain.txt
  python ../../preprocess.py -train_src $DATA_DIR/src-untrain.txt \
     -train_tgt $DATA_DIR/tgt-untrain.txt -save_data $DATA_DIR \
     -src_seq_length 1000 -tgt_seq_length 1000 -src_vocab $VOCAB -overwrite
  python ../../train.py -data $DATA_DIR -unlearn True \
     -train_from $MODEL -reset_optim all \
     --train_steps 8 -optim sgd -max_grad_norm 100 -dropout 0 \
     -attention_dropout 0 -learning_rate 0.001 -warmup_steps 0 \
     -decay_method none -save_model unlearnt
  python ../../score_unlearnt.py -model unlearnt_step_8.pt \
     -src $INTERPR_SRC -tgt $INTERPR_TGT -score_file scores.npy
  rm unlearnt_step_8.pt
  rm $DATA_DIR/*.txt
done