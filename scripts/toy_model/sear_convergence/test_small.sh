#! /bin/bash

datadir=/home/dpk25/MolecularTransformer2/experiments/toy_model/sear/
model_list=/home/dpk25/rds/hpc-work/toy_model/sear_models_small_conv/

for i in {1..10}
do
    j=$((312*$i))
    python ../../../translate.py -model ${model_list}toy_model_step_$j.pt \
        -src ${datadir}src-benchmark.txt \
        -output ./conv_small_$j.txt \
        -batch_size 64 -replace_unk -max_length 200 -n_best 1
done 
