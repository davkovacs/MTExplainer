"""Script to extract the reactions from the USPTO 15k dataset files to SMILES"""

import argparse

from rdkit import Chem

def smi_tokenizer(smi):
    """
    Tokenize a SMILES molecule or reaction string
    """
    import re
    smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi)) # canonicalize string
    pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == ''.join(tokens)
    return ' '.join(tokens)

def main(args):

    clean_file = open(args.input_name,'r')
    clean_lines = clean_file.readlines()

    src_train = open(args.dir+'src-train.txt','r')
    src_train_lines = src_train.readlines()
    src_train_lines = [line[:-1] for line in src_train_lines]

    tgt_train = open(args.dir+'tgt-train.txt','r')
    tgt_train_lines = tgt_train.readlines()
    tgt_train_lines = [line[:-1] for line in tgt_train_lines]

    src_test = open(args.dir+'src-test.txt','r')
    src_test_lines = src_test.readlines()
    src_test_lines = [line[:-1] for line in src_test_lines]

    tgt_test = open(args.dir+'tgt-test.txt','r')
    tgt_test_lines = tgt_test.readlines()
    tgt_test_lines = [line[:-1] for line in tgt_test_lines]

    src_val = open(args.dir+'src-val.txt','r')
    src_val_lines = src_val.readlines()
    src_val_lines = [line[:-1] for line in src_val_lines]

    tgt_val = open(args.dir+'tgt-val.txt','r')
    tgt_val_lines = tgt_val.readlines()
    tgt_val_lines = [line[:-1] for line in tgt_val_lines]

    src_list = []
    product_list=[]

    num_rxns_notfound=0
    for line in clean_lines:
        reaction = line.split()[0]
        product = reaction[reaction.find('>')+2:]
        product = smi_tokenizer(product)
        try:
            dirty_index = tgt_train_lines.index(product)
            src_rxn = src_train_lines[dirty_index]
        except ValueError:
            try:
                dirty_index = tgt_test_lines.index(product)
                src_rxn = src_test_lines[dirty_index]
            except ValueError:
                try:
                    dirty_index = tgt_val_lines.index(product)
                    src_rxn = src_val_lines[dirty_index]
                except:
                    # print('rxn not found, skipping...')
                    num_rxns_notfound+=1
                    continue
        product_list.append(product)
        src_list.append(src_rxn)
    print('{} reactions could not be found in USPTO-MIT'.format(num_rxns_notfound))
    with open(args.output_tgt,'w') as f:
      f.write('\n'.join(product_list))

    with open(args.output_src,'w') as f:
      f.write('\n'.join(src_list))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_name','-i', type=str,
                        help='Path to .txt file of USTPO-15K reactions.')
    parser.add_argument('--dir','-dir', type=str,
                        help='Path to directory of USPTO-MIT dataset')
    parser.add_argument('--output_tgt','-ot', type=str,
                        help='Output file name for tokenized targets.')
    parser.add_argument('--output_src','-os', type=str,
                        help='Output file name for tokenized sources.')
    args = parser.parse_args()
    main(args)
