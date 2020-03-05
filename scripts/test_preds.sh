#! /bin/bash

dataset=MIT_mixed_augm
datadir=../data/data/${dataset}/
save_model=../trained_model/
experiments=../experiments/
model_list=${save_model}last20/

python ../translate.py -model ${save_model}${dataset}_last20_average.pt \
	-src ${datadir}src-test.txt \
	-output ${experiments}predictions_${model}_on_${dataset}_test.txt \
	-batch_size 64 -replace_unk -max_length 200 

python ../score_predictions.py -targets ${datadir}tgt-test.txt \
	-predictions ${experiments}predictions_${model}_on_${dataset}_test.txt
