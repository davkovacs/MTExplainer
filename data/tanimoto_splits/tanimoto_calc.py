"""
Code for conducting a random split of a reaction dataset, and calculating the Tanimoto similarity matrix of the reaction products using MPI parallelization

Author: William McCorkindale
"""
import sys
import pandas as pd
import numpy as np
from rdkit import DataStructs
from rdkit.Chem import MolFromSmiles, AllChem
from tqdm import tqdm
import random
import time
from sklearn.model_selection import train_test_split
from mpi4py import MPI

pd.options.mode.chained_assignment = None 

mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()

def return_borders(my_ind, dat_len, mpi_size):
    """
    Function for equally dividing dat_len elements between mpi_size MPI processes

    returns the indices border_low, border_high for a given MPI process with index my_ind
    """
    mpi_borders = np.linspace(0, dat_len, mpi_size + 1).astype('int')

    border_low = mpi_borders[my_ind]
    border_high = mpi_borders[my_ind+1]
    return border_low, border_high

# file has to be .csv with separate columns for 'src' (reactant-reagents) and 'tgt' (products)
filename = sys.argv[1]
df = pd.read_csv(filename) 

df_train, df_test = train_test_split(df, test_size=0.7, random_state=42)

if mpi_rank==0:
    df_train.to_csv('train_presplit.txt', index=False)
    df_test.to_csv('test_presplit.txt', index=False)

# splits df_train between the MPI processes
my_border_low, my_border_high = return_borders(mpi_rank, len(df_train), mpi_size)
my_df = df_train[my_border_low:my_border_high]
my_len = len(my_df)

# convert reaction products into RDKit molecules and calculate Morgan fingerprints
my_df['mol'] = my_df['tgt'].apply(lambda x: MolFromSmiles(x))
df_test['mol'] = df_test['tgt'].apply(lambda x: MolFromSmiles(x))

fprints_train = [AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=1024) for mol in my_df['mol'].values]
fprints_test = [AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=1024) for mol in df_test['mol'].values]

# calculate float16 Tanimoto similarity matrix of size (my_len, len(df_test)) and save it to a .npy file
my_matrix = np.zeros((my_len, len(df_test)), dtype=np.float16)

if mpi_rank==0:
    for i in tqdm(range(len(fprints_train))):
        my_matrix[i] = np.array([DataStructs.FingerprintSimilarity(fprints_train[i], test_fprint) for test_fprint in fprints_test], dtype=np.float16)
else:
    for i in range(len(fprints_train)):
        my_matrix[i] = np.array([DataStructs.FingerprintSimilarity(fprints_train[i], test_fprint) for test_fprint in fprints_test], dtype=np.float16)

np.save('mpi_tanimoto_matrix_'+str(mpi_rank)+'.npy', my_matrix )
