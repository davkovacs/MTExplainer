#! /bin/bash

dataset=MIT_mixed_clean_augm
model_dir=../trained_model/

rx_id=./snar/  #./USPTO_mess/Cl/
baseline_txt=${rx_id}baseline.txt
src_txt=${rx_id}src_modif.txt
tgt1_txt=${rx_id}tgt1.txt
#tgt2_txt=${rx_id}baseline.txt

python ../translate.py -model ${model_dir}${dataset}_last20_avg.pt \
	-src ${src_txt} -tgt ${tgt1_txt}\
	-output ${rx_id}predictions_modif.txt \
	-batch_size 64 -replace_unk -max_length 200 -n_best 5 -verbose
sed -i "s/ //g"  ${rx_id}predictions_modif.txt

#python ../translate_IG.py -model ${model_dir}${dataset}_last20_avg.pt \
	-src ${src_txt} -baseline ${baseline_txt} -tgt ${tgt1_txt} -tgt2 ${tgt2_txt}\
	-output ${rx_id}IGs.npy -n_ig_steps 250 \
#	-batch_size 64 -replace_unk -max_length 250 -n_best 1 > ${rx_id}outp_min.out