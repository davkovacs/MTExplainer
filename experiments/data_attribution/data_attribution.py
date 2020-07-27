import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import subprocess

def main(args):
    input = args.input
    N = args.size  # size of USPTO-15k training set = 9236

    scores = np.empty(N)

    X = np.load(args.dir+'hidden_states_b' + str(input) + '.npy')
    X = np.mean(X, axis=0) # average across all tokens to return vector of size 256

    if args.type=='dotproduct': # prenormalize
        X = X / np.linalg.norm(X)

    for i in tqdm(range(N)):
        Y = np.load(args.dir+'hidden_states_b' + str(i) + '.npy')
        Y = np.mean(Y, axis=0)

        if args.type == 'dotproduct':
            Y = Y / np.linalg.norm(Y)
            scores[i] = np.vdot(X, Y)
        else:
            dist = np.linalg.norm(X - Y)
            scores[i]=1/(1+dist)

    inds = np.argpartition(scores, -10)[-10:] # index of top-10 scoring reactions (should contain original)

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
    parser.add_argument('-input', type=int, default=481,
                     help='Index of hiddenstate to compare against.')
    parser.add_argument('-size', type=int, default=9236,
                     help='Size of training set to search through.')
    parser.add_argument('-type', type=str, default='dist',
                     help='Method for calculating similarity', choices=['dist', 'dotproduct'])
    parser.add_argument('-dir', type=str, default='../../data/data/USPTO_15k/hidden_states/',
                     help='Location of saved hidden state .npy arrays.')
    parser.add_argument('-dataset', type=str, default='USPTO-15k.txt',
                     help='Location of untokenized reaction dataset for extraction of reactions.')
    parser.add_argument('-plot', action='store_true',
                     help='Whether or not to plot histogram of similarity scores.')

    args = parser.parse_args()

    main(args)

