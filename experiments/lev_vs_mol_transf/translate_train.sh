#! /bin/bash

model=/home/dpk25/rds/hpc-work/MolecularTransformer2_models/checkpoints/MIT_mixed_clean_augm/MIT_mixed_clean_augm_last20_avg.pt
source_txt=src-train.txt

python ../../translate.py -model ${model} \
	-src ${source_txt} \
	-output preds_mol_t_train.txt \
	-replace_unk -max_length 200 -n_best 1 -batch_size 256 -gpu -1
