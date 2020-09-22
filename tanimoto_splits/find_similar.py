"""
Code for printing example reactions from randomly split train/test splits that have products 
within 0.5 Tanimoto similarity to each other

Author: William McCorkindale
"""

import pandas as pd
import numpy as np

pd.options.mode.chained_assignment = None 

sim=0.5

df_train = pd.read_csv('train_presplit.txt')
df_test = pd.read_csv('test_presplit.txt')
len_df = len(df_train) + len(df_test)

my_matrix = np.load('mpi_tanimoto_matrix_0.npy')

my_border_low, my_border_high = return_borders(0, len(df_train), 32)
my_df = df_train[my_border_low:my_border_high]

hit_vector = np.any(my_matrix>sim, axis=0).astype(int)
my_matrix = my_matrix.T

for i,hit in enumerate(hit_vector):
    if hit:
        print('===')
        print(my_df.iloc[np.argmax(my_matrix[i])]['src']+'>>'+my_df.iloc[np.argmax(my_matrix[i])]['tgt'])
        print(df_test.iloc[i]['src']+'>>'+df_test.iloc[i]['tgt'])
        print('Similarity = {:.2f}'.format(np.max(my_matrix[i])))
