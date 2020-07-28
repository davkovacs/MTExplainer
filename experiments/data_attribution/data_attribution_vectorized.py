import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import subprocess

def main(args):
    input = args.input

    X = np.load(args.dir+'hidden_states_b' + str(input) + '.npy') # already averaged
    X = X.reshape(-1,1)

    Y = np.load(args.dir+'hidden_states.npy').T # load (N, 256) array, transpose to (256,N)
    dist = np.linalg.norm(X-Y, axis=0)
    scores = 1/(1+dist)
    
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

