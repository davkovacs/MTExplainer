#! /bin/bash

DATA_SRC=../../
DATA_TGT=../../
DATA_DIR=./data_unlearn
train_set_size=$(wc -l "$DATA_SRC")

for NUM in {1..$train_set_size};
do
  sed "${NUM}q;d" DATA_SRC > $DATA_DIR/src-untrain.txt
  sed "${NUM}q;d" DATA_TGT > $DATA_DIR/tgt-untrain.txt
  python ../../unlearn.py -data $DATA_DIR \
     -train_from  -reset-optim \
     --train_steps 8 -optim sgd -max_grad_norm 25 -dropout 0 \
     -attention_dropout 0 -learning_rate 0.001 -warmup_steps 0 \
     -decay_method none -save_model unlearnt.pt

  python ../../score_unlearnt.py -model unlearnt.pt \
     -src  -tgt  -in_score scores.npy -out_score scores.npy

  rm unlearnt.pt
  rm DATA_DIR/*.txt
done

