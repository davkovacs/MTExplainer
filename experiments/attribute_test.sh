#! /bin/bash

dataset=MIT_mixed_augm
model_dir=../trained_model/

rx_id=./USPTO_mess/Cl/
baseline_txt=${rx_id}baseline.txt
src_txt=${rx_id}src.txt
tgt1_txt=${rx_id}tgt1.txt
tgt2_txt=${rx_id}tgt2.txt

python ../translate.py -model ${model_dir}${dataset}_last20_average.pt \
	-src ${src_txt} -tgt ${tgt1_txt}\
	-output ${rx_id}predictions.txt \
	-batch_size 64 -replace_unk -max_length 200 -n_best 5 -verbose
sed -i "s/ //g"  ${rx_id}predictions.txt

python ../translate_IG.py -model ${model_dir}${dataset}_last20_average.pt \
	-src ${src_txt} -baseline ${baseline_txt} -tgt ${tgt1_txt} -tgt2 ${tgt2_txt}\
	-output ${rx_id}IGs.npy -n_ig_steps 1 \
	-batch_size 64 -replace_unk -max_length 200 -n_best 1