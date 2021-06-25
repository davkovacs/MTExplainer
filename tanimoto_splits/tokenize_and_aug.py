"""
Script to perform train/val splitting, tokenization, and augmentation of the training set
"""

import pandas as pd
from rdkit import RDLogger, Chem
import random
from tqdm import tqdm
import re
from sklearn.model_selection import train_test_split


def smi_list_tokenizer(smi_list):
    """
    Tokenize a SMILES molecule or reaction 
    """
    pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    smi_list_tokenized = []
    for smi in smi_list:
        tokens = [token for token in regex.findall(smi)]
        assert smi == ''.join(tokens)
        smi_list_tokenized.append(' '.join(tokens))
    return smi_list_tokenized


def data_augm(rx_list):
    """
    Tokenize a SMILES molecule or reaction 
    """
    RDLogger.DisableLog('rdApp.*')
    rx_list_augm = rx_list.copy()
    for j, rx in enumerate(rx_list):
        rx_rand = rx
        i = 0
        while rx == rx_rand and i < 10:
            rx_mol = Chem.MolFromSmiles(rx)
            if rx_mol == None:
                print(rx)
            new_atom_order = list(range(rx_mol.GetNumAtoms()))
            random.shuffle(new_atom_order)
            random_mol = Chem.RenumberAtoms(rx_mol, newOrder=new_atom_order)
            rx_rand = Chem.MolToSmiles(random_mol, canonical=False, isomericSmiles=False)
            i += 1
        if rx_rand == rx:
            print('\nFailed to generate random equivalent SMILES for the reaction:')
            print(rx)
        else:
            rx_list_augm.append(rx_rand)
    return rx_list_augm
def main(args):

    # load Tanimoto split dataset
    df_train = pd.read_csv(args.save_dir+'/train.csv')
    df_val = pd.read_csv(args.save_dir+'/val.csv')
    df_test = pd.read_csv(args.save_dir+'/test.csv')
    
    df_train, df_val, = train_test_split(df_train, test_size=0.111111)
    
    # augment training set, perform tokenization and append to lists
    train_rxns=[]
    tokenized_srcs_train = smi_list_tokenizer(data_augm(list(df_train['src'].values)))
    tokenized_prods_train = smi_list_tokenizer(df_train['tgt'].values)
    tokenized_prods_train = tokenized_prods_train + tokenized_prods_train
    for i, src in tqdm(enumerate(tokenized_srcs_train), total = len(tokenized_srcs_train)):
        rxn = src+'>>'+tokenized_prods_train[i]
        train_rxns.append(rxn)
        
    val_rxns=[]
    tokenized_srcs_val = smi_list_tokenizer(df_val['src'])
    tokenized_prods_val = smi_list_tokenizer(df_val['tgt'])
    tokenized_prods_val = tokenized_prods_val
    for i, src in tqdm(enumerate(tokenized_srcs_val), total = len(tokenized_srcs_val)):
        rxn = src+'>>'+tokenized_prods_val[i]
        val_rxns.append(rxn)
        
    test_rxns=[]
    tokenized_srcs_test = smi_list_tokenizer(df_test['src'])
    tokenized_prods_test = smi_list_tokenizer(df_test['tgt'])
    tokenized_prods_test = tokenized_prods_test
    for i, src in tqdm(enumerate(tokenized_srcs_test), total = len(tokenized_srcs_test)):
        rxn = src+'>>'+tokenized_prods_test[i]
        test_rxns.append(rxn)
    
    # write reactions to files
    with open(args.save_dir+'/train.txt', 'w') as filehandle:
        filehandle.writelines("%s\n" % rxn for rxn in train_rxns)
    with open(args.save_dir+'/val.txt', 'w') as filehandle:
        filehandle.writelines("%s\n" % rxn for rxn in val_rxns)
    with open(args.save_dir+'/test.txt', 'w') as filehandle:
        filehandle.writelines("%s\n" % rxn for rxn in test_rxns) 

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-save_dir', '--save_dir', type=str, required=True,
                        help='path to directory to save train/val/test splits.')
    args = parser.parse_args()

    main(args)


