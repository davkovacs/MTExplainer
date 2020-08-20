#! /bin/bash

dataset=MIT_mixed_clean_augm  #MIT_mixed_augm
datadir=../data/data/$dataset/data/
experiments=../experiments/

python ../../score_predictions.py -targets tgt-train.txt \
	-predictions preds_mol_t_train.txt  \
    -beam_size 1 -invalid_smiles
