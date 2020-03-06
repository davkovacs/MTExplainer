#! /bin/bash

dataset=MIT_mixed_augm
datadir=../data/data/${dataset}/
save_model=../trained_model/
experiments=../experiments/
source_txt=./sear.txt
target_txt=./sear_tgt.txt

python ../translate.py -model ${save_model}${dataset}_last20_average.pt \
	-src ${source_txt} -tgt ${target_txt}\
	-output predictions${model}_on_sear_with_tgt.txt \
	-batch_size 64 -replace_unk -max_length 200 -n_best 2 -verbose 
sed -i "s/ //g"  predictions${model}_on_sear_with_tgt.txt
