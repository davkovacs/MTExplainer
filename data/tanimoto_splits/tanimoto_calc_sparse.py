import sys
import random
import time
import argparse

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, save_npz
from tqdm import tqdm

from hurry.filesize import size
from rdkit import DataStructs
from rdkit.Chem import MolFromSmiles, AllChem
from sklearn.model_selection import train_test_split

from mpi4py import MPI

def return_borders(my_ind, dat_len, mpi_size):
    """
    Function for equally dividing dat_len elements between mpi_size MPI processes

    returns the indices border_low, border_high for a given MPI process with index my_ind
    """
    mpi_borders = np.linspace(0, dat_len, mpi_size + 1).astype('int')

    border_low = mpi_borders[my_ind]
    border_high = mpi_borders[my_ind+1]
    return border_low, border_high

def main(args):
    mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()
    mpi_size = mpi_comm.Get_size()
    
    # load data, generate morgan fingerprints for unique reaction products and obtain relevant MPI indices
    df = pd.read_csv(args.csv_path, usecols=['tgt'])
    
    fprints = [AllChem.GetMorganFingerprintAsBitVect(MolFromSmiles(x), radius=3, nBits=1024) for x in tqdm(df['tgt'].unique())]
    
    my_border_low, my_border_high = return_borders(mpi_rank, len(fprints), mpi_size)
    
    data = []
    row = []
    col = []
    
    if mpi_rank==0:
        for i in tqdm(range(my_border_low, my_border_high)): # loop over rows, split by MPI
            scores = np.array(DataStructs.BulkTanimotoSimilarity(fprints[i], fprints)) > args.threshold
     
            inds = np.argwhere(scores).flatten() # column indices where similarity > threshold
            num_true = np.count_nonzero(scores)
    
            data = data + [1]*num_true # convert boolean True/False to 1/0
            row = row + [i]*num_true # row indices
            col = col + inds.tolist() # column indices
    
    else:
        for i in range(my_border_low, my_border_high): # loop over rows
            scores = np.array(DataStructs.BulkTanimotoSimilarity(fprints[i], fprints)) > args.threshold
     
            inds = np.argwhere(scores).flatten() # column indices where similarity > threshold
            num_true = np.count_nonzero(scores) 
    
            data = data + [1]*num_true # convert boolean True/False to 1/0
            row = row + [i]*num_true # row indices
            col = col + inds.tolist() # column indices
    
    my_data = [ data, row , col]
    full_data = mpi_comm.gather(out,root=0)
    
    if mpi_rank == 0:
        # concatenate row/column indices from all MPI processes and construct + save sparse CSR matrix
        data = np.concatenate([x[0] for x in full_data])
        row = np.concatenate([x[1] for x in full_data])
        col = np.concatenate([x[2] for x in full_data])
        full_csr_matrix = csr_matrix((data, (row,col)), shape =(len(fprints), len(fprints)))
    
        save_npz(args.npz_path, full_csr_matrix)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('-csv_path', '--csv_path', type=str, required=True,
                        help='Path to the src,tgt .csv file to be split.')
    parser.add_argument('-threshold', '--threshold', type=float, default=0.6,
                        help='Tanimoto Similarity threshold for splitting.')
    parser.add_argument('-npz_path', '--npz_path', type=str, required=True,
                        help='path to save .npz file with saved sparse tanimoto matrix.')

    args = parser.parse_args()

    main(args)

