from __future__ import print_function
from rdkit import Chem

smiles='B.C1CCOC1.CC(C)(C)C1=CCCc2occc21'
def smi_tokenizer(smi):
    """
    Tokenize a SMILES molecule or reaction
    """
    import re
    pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == ''.join(tokens)
    return ' '.join(tokens)

m = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
print(smi_tokenizer(m))

