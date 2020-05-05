#! /bin/bash

DATA_DIRS=/unlearn_dataset/*
for f in $DATA_DIRS
do
  python ../../unlearn.py -data $f \
    -train_from  -reset-optim \
    --train_steps 8 -optim sgd -max_grad_norm 25 -dropout 0 \
    -attention_dropout 0 -learning_rate 0.001 -warmup_steps 0 \
    -decay_method none
done

