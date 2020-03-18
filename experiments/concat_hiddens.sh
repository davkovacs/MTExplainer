#! /bin/bash
src_dir=/rds-d2/user/wjm41/hpc-work/hidden_states/

python concat_h_arrays.py -n $1 -dir ${src_dir} -preprocessed True
