#! /bin/bash

dataset=MIT_mixed_augm
model_dir=../trained_model/
out_dir=/home/cdt1906/Documents/cdt/mphil/transformer/MolecularTransformer2/experiments/
baseline_txt=./baseline.txt

rx_id=./late_stage_cc/
src_txt=/home/cdt1906/Documents/cdt/mphil/transformer/MolecularTransformer2/data/data/MIT_mixed/src-train.txt
tgt1_txt=/home/cdt1906/Documents/cdt/mphil/transformer/MolecularTransformer2/data/data/MIT_mixed/tgt-train.txt

python ../translate.py -model ${model_dir}${dataset}_last20_average.pt \
	-src ${src_txt} -tgt ${tgt1_txt}\
	-output ${out_dir} -g_h_s True\
	-batch_size 1 -replace_unk -max_length 200 -n_best 1
