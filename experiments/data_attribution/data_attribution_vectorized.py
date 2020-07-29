import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import subprocess
import os

def main(args):
    os.system("python ../../translate.py -model {} -src {} -tgt {} -shard_size 0 \
               -output './h' -g_h_s True -batch_size 1 -replace_unk -max_length 200 \
               -n_best 1".format(args.model, args.input, args.input))

    X = np.load('./hhidden_states_b0.npy') # already averaged
    X = X.reshape(-1,1)

    Y = np.load(args.hiddens).T # load (N, 256) array, transpose to (256,N)
    dist = np.linalg.norm(X-Y, axis=0)
    scores = 1/(1+dist)
    
    inds = np.argpartition(scores, -args.n_match)[-args.n_match:] # index of top-n scoring reactions

    print('\nOriginal reaction:')
    bashCommand = "sed '" + str(np.argmax(scores) + 1) + "q;d' "+args.dataset
    subprocess.call(bashCommand, shell=True)
    print('\nMost similar reactions:')
    for ind in inds[np.argsort(scores[inds])]:
        bashCommand = "sed '" + str(ind + 1) + "q;d' "+args.dataset
        subprocess.call(bashCommand, shell=True)
        print('score = {:.3f}'.format(scores[ind]))
    if args.plot:
        plt.hist(scores, bins=1000)
        plt.ylabel('Frequency')
        if args.type=='dotproduct':
            plt.xlabel('normalized dot-product')
            plt.savefig('dot_'+str(input)+'_hist.png')
        else:
            plt.xlabel('1/1+euc-dist')
            plt.savefig('dist_'+str(input)+'_hist.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', type=str,
                     help='Model file whose predictions are interpreted')
    parser.add_argument('-input', type=str, 
                     help='File containing reactant-reagent SMILES of reaction to interpret')
    parser.add_argument('-type', type=str, default='dist',
                     help='Method for calculating similarity', choices=['dist', 'dotproduct'])
    parser.add_argument('-hiddens', type=str, default='../../data/data/USPTO_15k/hidden_states/',
                     help='Saved hidden state .npy arrays file.')
    parser.add_argument('-n_match', type=int, default=10, 
                     help='Number of best matching training reactions to return')
    parser.add_argument('-dataset', type=str, default='USPTO-15k.txt',
                     help='Location of untokenized reaction dataset for extraction of reactions.')
    parser.add_argument('-plot', action='store_true',
                     help='Whether or not to plot histogram of similarity scores.')

    args = parser.parse_args()

    main(args)

