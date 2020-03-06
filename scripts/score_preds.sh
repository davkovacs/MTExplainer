#! /bin/bash

dataset=MIT_mixed_augm
datadir=../data/data/${dataset}/
save_model=../trained_model/
experiments=../experiments/
model_list=${save_model}last20/

python ../score_predictions.py -targets ${datadir}tgt-test.txt \
	-predictions ${experiments}predictions_${model}_on_${dataset}_test.txt \
    -beam_size 1
