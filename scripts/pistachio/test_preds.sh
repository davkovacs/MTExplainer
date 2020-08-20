#! /bin/bash

datadir=/home/dpk25/rds/hpc-work/pistachio/
model_list=/home/dpk25/rds/hpc-work/pistachio/checkpoints/last_10/

#python ../../translate.py -model ${model_list}pist_last10_avg.pt \
#	-src ${datadir}src-test.txt \
#	-output ${datadir}experiments/preds_on_pist_test_million.txt \
#	-batch_size 64 -replace_unk -max_length 200 -n_best 2 

python ../../score_predictions.py -targets ${datadir}experiments/tgt_million.txt \
	-predictions ${datadir}experiments/preds_on_pist_test_million.txt \
	-invalid_smiles -beam_size 2 > pist_scores_million.txt
