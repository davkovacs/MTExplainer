#! /bin/bash

dataset=MIT_mixed_clean_augm
model_dir=/home/dpk25/rds/hpc-work/MolecularTransformer2_models/checkpoints/MIT_mixed_clean_augm
out_dir=/home/dpk25/rds/hpc-work/hidden_states/hid_1rx/h

src_txt=/home/dpk25/MolecularTransformer2/data/data/MIT_mixed_clean/src-train.txt
tgt1_txt=/home/dpk25/MolecularTransformer2/data/data/MIT_mixed_clean/tgt-train.txt

python ../../translate.py -model ${model_dir}/MIT_mixed_clean_augm_last20_avg.pt \
	-src ${src_txt} -tgt ${tgt1_txt} -shard_size 0\
	-output ${out_dir} -g_h_s True\
	-batch_size 1 -replace_unk -max_length 200 -n_best 1
