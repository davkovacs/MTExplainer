#! /bin/bash

model_list=/home/dpk25/rds/hpc-work/pistachio/checkpoints/last_10/

python ../tools/average_models.py -models ${model_list}*.pt \
	-output ${model_list}pist_last10_avg.pt
