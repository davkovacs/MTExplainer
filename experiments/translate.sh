#! /bin/bash

dataset=MIT_mixed_clean_augm  #MIT_mixed_augm
datadir=../data/data/${dataset}/good_data/
model=../trained_model/MIT_mixed_clean_augm_last20_avg.pt
experiments=../experiments/
source_txt=${datadir}src-test.txt
target_txt=${datadir}tgt-test.txt

python ../translate.py -model ${model} \
	-src ${source_txt} -tgt ${target_txt}\
	-output predictions_on_${dataset}.txt \
	-batch_size 64 -replace_unk -max_length 200 -n_best 5  
#sed -i "s/ //g"  predictions${model}_on_sear_with_tgt.txt
