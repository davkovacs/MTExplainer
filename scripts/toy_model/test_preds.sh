#! /bin/bash

datadir=/home/dpk25/MolecularTransformer2/experiments/toy_model/data_9k/
model_list=/home/dpk25/rds/hpc-work/toy_model/reduction_9k/

python ../../translate.py -model ${model_list}toy_model_step_125000.pt \
	-src ${datadir}src-test.txt \
	-output ./preds_on_test_125k.txt \
	-batch_size 64 -replace_unk -max_length 200 -n_best 2 

python ../../score_predictions.py -targets ${datadir}tgt-test.txt \
	-predictions ./preds_on_test_125k.txt \
	-invalid_smiles -beam_size 2 > scores_125k.txt
