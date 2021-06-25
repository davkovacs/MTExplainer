import pandas as pd
import numpy as np
from tqdm import tqdm
import scipy.sparse

from sklearn.model_selection import train_test_split


def main(args):

    # load dataset and tanimoto matrix 
    df = pd.read_csv(args.csv_path) 
    sparse_tanimoto = scipy.sparse.load_npz(args.npz_path)

    # get unique reaction products and index reactions by reaction product
    tgts = df['tgt'].unique()
    tgt_dict = dict(zip(tgts,range(len(tgts))))
    df['tgt_index'] = [tgt_dict[val] for val in df['tgt']] 
    
    # initial completely random train/test split
    train_inds, test_inds = train_test_split(np.array(range(len(tgts))), test_size=args.test_frac, random_state=21)
    df_train = df[df['tgt_index'].isin(train_inds)]
    df_test = df[df['tgt_index'].isin(test_inds)]
    
    print('Starting random Train/Test split = {:.1f}:{:.1f}'.format(100*len(df_train)/len(df),
                                                                    100*len(df_test)/len(df)))
    
    # collect rows of tanimoto matrix corresponding to test set reactions
    test_mat = sparse_tanimoto[test_inds]
    similar_indices = test_mat.indices
    test_mat = test_mat.tocoo()
    test_rows = test_mat.row
    
    # find products in test set that are within tanimoto threshold of a product in the training set
    overlap_inds = test_inds[test_rows[np.isin(similar_indices,train_inds)]]
    
    # redistribute similar products from test set to training set
    train_inds_new = np.unique(np.concatenate((train_inds, overlap_inds)))
    test_inds_new = np.array(test_inds)[~np.isin(test_inds, overlap_inds)]
    df_train = df[df['tgt_index'].isin(train_inds_new)]
    df_test = df[df['tgt_index'].isin(test_inds_new)]
    
    print('Tanimoto Train/Test split = {:.1f}:{:.1f}'.format(100*len(df_train)/len(df),
                                                             100*len(df_test)/len(df)))
    
    ### Validation set
    
    # initial completely random train/val split
    train_inds, val_inds = train_test_split(train_inds_new, test_size=args.test_frac, random_state=21)
    df_train = df[df['tgt_index'].isin(train_inds)]
    df_val = df[df['tgt_index'].isin(val_inds)]
    
    print('Starting random Train/Val split = {:.1f}:{:.1f}'.format(100*len(df_train)/len(df),
                                                                   100*len(df_val)/len(df)))

    # collect rows of tanimoto matrix corresponding to validation set reactions
    val_mat = sparse_tanimoto[val_inds]
    similar_indices = val_mat.indices
    val_mat = val_mat.tocoo()
    val_rows = val_mat.row
    
    # find products in validation set that are within tanimoto threshold of a product in the training set
    overlap_inds = val_inds[val_rows[np.isin(similar_indices,train_inds)]]
    
    # redistribute similar products from validation set to training set
    train_inds_new = np.unique(np.concatenate((train_inds, overlap_inds)))
    val_inds_new = np.array(val_inds)[~np.isin(val_inds, overlap_inds)]
    df_train = df[df['tgt_index'].isin(train_inds_new)]
    df_val = df[df['tgt_index'].isin(val_inds_new)]
    
    print('Final Train/Val/Test split = {:.1f}:{:.1f}:{:.1f}'.format(100*len(df_train)/len(df),
                                                              100*len(df_val)/len(df),
                                                              100*len(df_test)/len(df)))
    
    # save splits
    df_train.to_csv(args.save_dir+'/train.csv', index=False)
    df_val.to_csv(args.save_dir+'/val.csv', index=False)
    df_test.to_csv(args.save_dir+'/test.csv', index=False)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-csv_path', '--csv_path', type=str, required=True,
                        help='Path to the src,tgt .csv file to be split.')
    parser.add_argument('-threshold', '--threshold', type=float, default=0.6, required=True,
                        help='Tanimoto Similarity threshold for splitting')
    parser.add_argument('-npz_path', '--npz_path', type=str, required=True,
                        help='path to saved .npz file with sparse tanimoto matrix.')
    parser.add_argument('-save_dir', '--save_dir', type=str, required=True,
                        help='path to directory to save train/val/test splits.')
    args = parser.parse_args()

    main(args)


