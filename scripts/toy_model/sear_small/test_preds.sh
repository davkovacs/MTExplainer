#! /bin/bash

datadir=/home/dpk25/rds/hpc-work/toy_model/data_sear_small/
model_list=/home/dpk25/rds/hpc-work/toy_model/sear_models_small/

python ../../../translate.py -model ${model_list}toy_model_step_150000.pt \
	-src ${datadir}src-test.txt \
	-output ${datadir}/preds_on_test_150k.txt \
	-batch_size 64 -replace_unk -max_length 200 -n_best 1

python ../../../score_predictions.py -targets ${datadir}tgt-test.txt \
	-predictions ${datadir}/preds_on_test_150k.txt \
	-invalid_smiles -beam_size 1 > scores_150k.txt
