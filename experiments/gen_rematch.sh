#! /bin/bash
src_dir=/rds-d2/user/wjm41/hpc-work/hidden_states/
src=b0

python rematch.py -src ${src_dir}hidden_states_${src}.npy -dir ${src_dir}
