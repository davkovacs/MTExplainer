#! /bin/bash

inp_rx=rx_smiles.txt
data_dir=/home/dpk25/MolecularTransformer2/data/data/MIT_mixed_clean/
model=/home/dpk25/rds/hpc-work/MolecularTransformer2_models/checkpoints/MIT_mixed_clean_augm/MIT_mixed_clean_augm_last20_avg.pt
hidden_array=/home/dpk25/rds/hpc-work/hidden_states/hidden_avg.npy
train_rxs=MIT_clean_train.txt
n_return=5

python data_attribution_vectorized.py -model ${model} -input ${inp_rx} -hiddens ${hidden_array} \
		-n_match ${n_return} -dataset ${train_rxs}
