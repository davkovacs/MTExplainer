#! /bin/bash

dataset=MIT_mixed_augm
save_model=../trained_model/
source_txt=./sear.txt
baseline_txt=./baseline.txt
target_txt1=./sear_tgt.txt
target_txt2=./sear_tgt2.txt

python ../translate_IG.py -model ${save_model}${dataset}_last20_average.pt \
	-src ${source_txt} -baseline ${baseline_txt} -tgt ${target_txt1} -tgt2  ${target_txt2}\
	-output predictions_on_sear_with_tgts.txt \
	-batch_size 64 -replace_unk -max_length 200 -n_best 1 -verbose
# sed -i "s/ //g"  predictions_on_sear_with_tgt2.txt
