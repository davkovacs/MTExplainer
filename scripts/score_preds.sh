#! /bin/bash

dataset=USPTO_15k #MIT_mixed_clean_augm  #MIT_mixed_augm
datadir=../data/data/$dataset/data/
experiments=../experiments/

python ../score_predictions.py -targets ${datadir}tgt-test.txt \
	-predictions ${experiments}predictions_clean_on_USPTO_15.txt \
    -beam_size 5 -invalid_smiles
