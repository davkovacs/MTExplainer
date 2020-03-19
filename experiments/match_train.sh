#! /bin/bash

dataset=MIT_mixed_augm
model_dir=../trained_model/
baseline_txt=./baseline.txt

rx_id=./sear_meta/
src_txt=${rx_id}sear_meta.txt
tgt1_txt=${rx_id}sear_meta_tgt1.txt
tgt2_txt=${rx_id}sear_meta_tgt2.txt

src_dir=/rds-d2/user/wjm41/hpc-work/hidden_states/

python ../translate_train_sim.py -model ${model_dir}${dataset}_last20_average.pt \
	-src ${src_txt} -baseline ${baseline_txt} -tgt ${tgt1_txt} -tgt2 ${tgt2_txt}\
	-output ${rx_id}enc_hidden.npy -output2 ${rx_id}enc_IG.npy -n_ig_steps 100 \
	-batch_size 64 -replace_unk -max_length 200 -n_best 1

python rematch.py -src_txt ${src_dir}../datasets/MIT_mixed/src-train.txt -src ${rx_id}enc_hidden.npy \
                  -enc_IG ${rx_id}enc_IG.npy -dir ${src_dir} -kernel average -out_txt ${rx_id}best_rxs.txt -n_best 5
