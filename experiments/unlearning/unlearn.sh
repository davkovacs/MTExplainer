#! /bin/bash
DATA_SRC=../../
DATA_TGT=../../
DATA_DIR=./data_unlearn
train_set_size=$(wc -l "$DATA_SRC")
MODEL=something.pt
​
for NUM in {1..$train_set_size};
do
  sed "${NUM}q;d" DATA_SRC > $DATA_DIR/src-untrain.txt
  sed "${NUM}q;d" DATA_TGT > $DATA_DIR/tgt-untrain.txt
  python ../../train.py -data $DATA_DIR -unlearn \
     -train_from $MODEL -reset-optim \
     --train_steps 8 -optim sgd -max_grad_norm 100 -dropout 0 \
     -attention_dropout 0 -learning_rate 0.001 -warmup_steps 0 \
     -decay_method none -save_model unlearnt.pt
​
  python ../../score_unlearnt.py -model unlearnt.pt \
     -src  -tgt  -score_file scores.npy
​
  rm unlearnt.pt
  rm DATA_DIR/*.txt
done