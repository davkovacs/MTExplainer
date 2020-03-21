from __future__ import print_function
from rdkit import Chem

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

m = Chem.MolToSmiles(Chem.MolFromSmiles('CCO.CCOC(C)=O.CCOCc1cc(OC)c(-c2csc3c(N(CCC4CC4)C4CCOCC4)c(OC)nn23)c(OC)c1.O=P(O)(O)O'))
print(smi_tokenizer(m))
