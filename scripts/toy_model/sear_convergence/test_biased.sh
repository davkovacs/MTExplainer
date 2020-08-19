#! /bin/bash

datadir=/home/dpk25/MolecularTransformer2/experiments/toy_model/sear/
model_list=/home/dpk25/rds/hpc-work/toy_model/sear_models_biased_conv/
savedir=./biased/
for num in {1..5}
do
    model_list=/home/dpk25/rds/hpc-work/toy_model/sear_models_biased_conv$num/
    for i in {1..6}
    do
        k=$((2**($i-1)))
        j=$((156*$k))
        python ../../../translate.py -model ${model_list}toy_model_step_$j.pt \
            -src ${datadir}src-benchmark.txt \
            -output ${savedir}conv_bi_m${num}_$j.txt \
            -batch_size 64 -replace_unk -max_length 200 -n_best 1
    done
done 
