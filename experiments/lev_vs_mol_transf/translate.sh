#! /bin/bash

model=/home/dpk25/rds/hpc-work/MolecularTransformer2_models/checkpoints/MIT_mixed_clean_augm/MIT_mixed_clean_augm_last20_avg.pt
source_txt=test_src.txt

python ../../translate.py -model ${model} \
	-src ${source_txt} \
	-output preds_mol_t_noBeam.txt --beam_size 1 \
	-replace_unk -max_length 200 -n_best 1 -batch_size 128 -gpu -1
