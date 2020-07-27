#! /bin/bash

model_list=/home/dpk25/rds/hpc-work/pistachio/checkpoints/last_20/

python ../tools/average_models.py -models ${model_list}*.pt \
	-output ${model_list}pist_last20_avg.pt
