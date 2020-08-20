#! /bin/bash

dataset=USPTO_15k
datadir=../data/data/${dataset}/data/
model=/home/dpk25/rds/hpc-work/MolecularTransformer2_models/checkpoints/USPTO_15k/model_step_100000.pt
experiments=../experiments/
source_txt=../data/data/USPTO_15k/data/src-test.txt
target_txt=../data/data/USPTO_15k/data/tgt-test.txt

python ../translate.py -model ${model} \
	-src ${source_txt} -tgt ${target_txt}\
	-output predictions_on_USPTO_15k.txt \
	-batch_size 32 -replace_unk -max_length 200 -n_best 5 
#sed -i "s/ //g"  predictions${model}_on_sear_with_tgt.txt
