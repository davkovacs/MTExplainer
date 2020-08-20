#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals
import argparse
from rdkit import Chem
import pandas as pd
import onmt.opts

def canonicalize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    else:
        return ''

def main(opt):
    with open(opt.targets, 'r') as f:
        targets = [''.join(line.strip().split(' ')) for line in f.readlines()]

    with open(opt.predictions, 'r') as f:
        preds =  [''.join(line.strip().split(' ')) for line in f.readlines()]
 

    with open(opt.reactions, 'r') as f:
        rxs =  [''.join(line.strip().split(' ')) for line in f.readlines()]

    mistakes = []
    for i, pred in enumerate(preds):
        pred_can = canonicalize_smiles(pred)
        if pred_can != targets[i]:
            mistakes.append(rxs[i] + "." + pred)

    with open(opt.outp, 'w') as f:
        for rx in mistakes:
            f.write("%s\n" % rx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='process_preds.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-reactions', type=str, 
                       help='Path to file containing the training reaction SMILES')
    parser.add_argument('-outp', type=str,
                       help='Path to output file containing the mispredicted reactions')
    parser.add_argument('-predictions', type=str, default="",
                       help="Path to file containing the predictions")
    parser.add_argument('-targets', type=str, default="",
                       help="Path to file containing targets")

    opt = parser.parse_args()
    main(opt)
