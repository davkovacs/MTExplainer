#! /bin/bash
src_dir=/rds-d2/user/wjm41/hpc-work/hidden_states/
src=b1

python rematch.py -src_txt ${src_dir}../datasets/MIT_mixed/src-train.txt -src ${src_dir}hidden_states_${src}.npy -dir ${src_dir} -kernel $1 -out_txt best_rxs.txt -n_best 5

