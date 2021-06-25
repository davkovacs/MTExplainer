# Tanimoto-Split USPTO Datasets

This directory contains code for splitting USPTO (or any other reaction dataset) via the Tanimoto similarities of the reaction products. Due to the size of the Tanimoto similarity matrix, the code utilises MPI to parallelize the operations across multiple CPUs. 

We also provide two pre-split USPTO datasets in the `.tar.gz` files with Tanimoto similarity threshold 0.4 and 0.6. These contain separate reactant-reagent and product files for train/val/test as tokenized SMILES strings. Both training sets are augmented.

## Performing Tanimoto Splitting
To split your own reaction dataset via the Tanimoto similarities of the reaction products, first process your dataset so it is the same format as `MIT_mixed_clean.csv` (.csv with separate columns for 'src' and 'tgt' - unzip from this directory for a look) and run the following code (changing filenames where necessary):

(25 June 2021) - much less memory intensive scripts utilising scipy sparse matrices have been added, and the example script has been changed accordingly.
```
#! /bin/bash

# Tanimoto splitting:
filename=MIT_mixed_clean.csv
tanimoto_sim=0.4
num_cores=32
save_dir=data

mpirun -np $num_cores -ppn $num_cores python tanimoto_calc_sparse.py -csv_path $filename -threshold 0.6 -npz_path tanimoto_s6.npz
mpirun -np $num_cores -ppn $num_cores python do_tanimoto_split_sparse.py -csv_path $filename -test_frac 0.3 -npz_path tanimoto_s6.npz -save_dir $save_dir

# Run this for some examples of similar reactions (doesn't work with sparse matrices, rerun with non-sparse methods instead):
# python find_similar.py

# train/val splitting,  Augmentation + shuffling of training set, Tokenization, separation into src/tgt:
python tokenize_and_aug.py -save_dir $save_dir
./split_src_tgt.sh $save_dir
```
