"""
Code for processing the MPI calculated Tanimoto similarity matrices to conduct Tanimoto splitting

Author: William McCorkindale
"""
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
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

# float argument for Tanimoto threshold
sim=float(sys.argv[1])

# load random train/test splits as well as Tanimoto similarity sub-matrix
df_train = pd.read_csv('train_presplit.txt')
df_test = pd.read_csv('test_presplit.txt')
len_df = len(df_train) + len(df_test)

my_matrix = np.load('mpi_tanimoto_matrix_'+str(mpi_rank)+'.npy')

# splits df_train between the MPI processes
my_border_low, my_border_high = return_borders(mpi_rank, len(df_train), mpi_size)
my_df = df_train[my_border_low:my_border_high]
my_len = len(my_df)

# each MPI process applies Tanimoto threshold to their own Tanimoto matrix
# and returns the results as a [0,1] integer matrix to the 0th MPI process
# make sure you use the same number of MPI processes as you did to calculate the matrices!
full_matrix = np.zeros(len(df_test)).astype(int)

my_matrix = np.any(my_matrix>sim, axis=0).astype(int)
mpi_comm.Reduce([my_matrix, MPI.INT], [full_matrix, MPI.INT], op=MPI.SUM, root=0 )

# 0th MPI process uses the full integer matrix to do Tanimoto splitting and save the new splits
if mpi_rank==0:
    print('Percentage of test set with similarity > {}: {:.1f}%'.format(sim, 100*np.count_nonzero(full_matrix)/len_df))
    print('Original train/test ratio: {:.1f}%/{:.1f}%'.format(100*len(df_train)/len_df, 100*len(df_test)/len_df))
    full_matrix = full_matrix.astype(bool) # anything other than 0 is converted to False 
    df_train_new = df_train.append(df_test.iloc[full_matrix]) # test set reactions with an integer 1 get appended into train set
    df_test_new = df_test[~full_matrix] # only keep test set reactions with integer 0

    print('Post-split train/test ratio: {:.1f}%/{:.1f}%'.format(100*len(df_train_new)/len_df, 100*len(df_test_new)/len_df))
    df_train_new.to_csv('train_split_new.txt', index=False)
    df_test_new.to_csv('test_split_new.txt', index=False)

mpi_comm.Barrier()
MPI.Finalize()
