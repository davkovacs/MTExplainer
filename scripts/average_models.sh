#! /bin/bash

dataset=MIT_mixed_augm
save_model=/home/dpk25/rds/hpc-work/MolecularTransformer2_models/checkpoints/
model_list=${save_model}last20/

python ../tools/average_models.py -models ${model_list}*.pt \
	-output ${save_model}${dataset}_last20_average.pt
