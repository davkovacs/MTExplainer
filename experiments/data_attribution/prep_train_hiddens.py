import numpy as np
import argparse
from tqdm import tqdm


def main(args):
    N = args.size # number of reactions in training set
    all_states = np.empty((N, 256))

    for i in tqdm(range(N)):
        Y = np.load(args.dir+'hhidden_states_b' + str(i) + '.npy')
        all_states[i] = Y
    
    np.save(args.out, all_states)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dir', type=str, default='../../../data/data/USPTO_15k/hidden_states/',
                     help='Location of saved hidden state .npy arrays.')
    parser.add_argument('-size', type=int, default=9236,
                     help='Size of training set to search through.')
    parser.add_argument('-out', type=str,
                     help='name of output file to save the concatenated hidden arrrays')

    args = parser.parse_args()

    main(args)

