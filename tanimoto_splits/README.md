# Tanimoto-Split USPTO Datasets

This directory contains code for splitting USPTO (or any other reaction dataset) via the Tanimoto similarities of the reaction products. Due to the size of the Tanimoto similarity matrix, the code utilises MPI to parallelize the operations across multiple CPUs. 

We also provide two pre-split USPTO datasets in the `.tar.gz` files with Tanimoto similarity threshold 0.4 and 0.6. These contain separate reactant-reagent and product files for train/val/test as tokenized SMILES strings. Both training sets are augmented.

## Performing Tanimoto Splitting
To split your own reaction dataset via the Tanimoto similarities of the reaction products, first process your dataset so it is the same format as `MIT_mixed_clean.csv` (unzip from this directory for a look) and run the following code (changing filenames where necessary):

```
#! /bin/bash

# Tanimoto splitting:
tanimoto_sim=0.5
num_cores=32
mpirun -np $num_cores -ppn $num_cores python tanimoto_calc.py $tanimoto_sim
mpirun -np $num_cores -ppn $num_cores python do_tanimoto_split.py

# Run this for some examples of similar reactions:
# python find_similar.py

# train/val splitting,  Augmentation + shuffling of training set, Tokenization, separation into src/tgt:
python tokenize_and_aug.py
./split_src_tgt.sh
```
