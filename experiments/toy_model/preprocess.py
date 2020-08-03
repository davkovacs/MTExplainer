from rdkit import Chem
import sys
import re

def smi_tokenizer(smi):
    """
    Tokenize a SMILES molecule or reaction string
    """
    smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi)) # canonicalize string
    pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == ''.join(tokens)
    return ' '.join(tokens)

def main():
    src_train = open(sys.argv[1],'r')
    src_train_lines = src_train.readlines()
    src_train_lines = [line[:-1] for line in src_train_lines]
    product_list=[]

    for line in src_train_lines:
        reaction = line.split()[0]
        product = ''
        for mol in reaction.split('.'):
            product += smi_tokenizer(mol) +' . '
        product_list.append(product[:-3])

    with open(sys.argv[1],'w') as f:
      f.write('\n'.join(product_list))

if __name__ == '__main__':
    main()