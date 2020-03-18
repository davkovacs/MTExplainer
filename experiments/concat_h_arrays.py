import numpy as np
import time
import argparse

def main(num, dir, preprocessed):
    if preprocessed==False:
        low_bound = 0+num*100000
        high_bound = (num+1)*100000
        
        if num==4: #final batch
           low_bound = 400000
           high_bound = 409035
        
        h_list = []
        for i in range(low_bound, high_bound):
            h_list.append(np.load(dir+'hidden_states_b'+str(i)+'.npy'))       
        
        np.save(dir+'h_arrays_'+str(num)+'.npy',h_list)
    else:
        X = np.load(dir+'h_arrays_0.npy', allow_pickle=True)
        for i in range(1,5):
            X = np.append(X,np.load(dir+'h_arrays_'+str(i)+'.npy',allow_pickle=True))
        print(X)
        print(len(X))
        print(X[0].shape)
        print(X[-1].shape)
        np.save(dir+'h_arrays_full.npy',X)
if __name__ == "__main__":

     parser = argparse.ArgumentParser()
     parser.add_argument('-n', type=int, default=0, 
                         help='Shard number of hidden states to concatenate together. Choose from 0-4.')
     parser.add_argument('-dir', type=str, 
                         help='Directory where the hidden state numpy arrays are stored.')
     parser.add_argument('-preprocessed', type=bool, default=False, 
                         help='Whether or not all of the hidden state arrays have been batch-concatenated yet.')
     args = parser.parse_args()

main(args.n, args.dir, args.preprocessed)
