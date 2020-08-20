#! /bin/bash

datadir=../../../data/data/MIT_mixed_clean_augm/
model_dir=/home/dpk25/rds/hpc-work/MolecularTransformer2_models/checkpoints/MIT_mixed_clean_augm/small

python ../../../translate.py -model ${model_dir}/small_MIT_step_450000.pt \
	-src ${datadir}src-test.txt \
	-output preds_on_test_450k.txt \
	-batch_size 64 -replace_unk -max_length 200 

python ../../../score_predictions.py -targets ${datadir}tgt-test.txt \
	-predictions preds_on_test_450k.txt -invalid_smiles -beam_size 1 > scores_450k.txt
