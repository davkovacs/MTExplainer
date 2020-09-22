# Tanimoto-Split USPTO Datasets

This directory contains code for splitting USPTO (or any other reaction dataset) via the Tanimoto similarities of the reaction products. Due to the size of the Tanimoto similarity matrix, the code utilises MPI to parallelize the operations across multiple CPUs. 

We also provide two pre-split USPTO datasets in the `.tar.gz` files with Tanimoto similarity threshold 0.4 and 0.6. These contain separate reactant-reagent and product files for train/val/test as tokenized SMILES strings. Both training sets are augmented.
