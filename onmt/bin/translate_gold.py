#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import onmt.opts as opts
import onmt.inputters as inputters
from onmt.utils.misc import split_corpus
from onmt.utils.parse import ArgumentParser
from onmt.utils.logging import init_logger
from onmt.translate.translator_gold import build_translator

class GoldScorer(nn.Module):
    # nn to produce the probability difference between tgt1 and tgt2
    def __init__(self, opt):
        super(GoldScorer, self).__init__()
        self.gold_scorer = TranslateGoldDiff(opt)

    def forward(self, src_embed):
        return self.gold_scorer(src_embed)

class TranslateGoldDiff(object):
    # functor returning prob difference between tgt1 and tgt2 given src_embed
    def __init__(self, opt):
        self.opt = opt
    def __call__(self, src_embed):
        translator = build_translator(self.opt, report_score=True)
        src_shards = split_corpus(self.opt.src, self.opt.shard_size)
        tgt_shards = split_corpus(self.opt.tgt, self.opt.shard_size)
        tgt2_shards = split_corpus(self.opt.tgt2, self.opt.shard_size)
        shard_trips = zip(src_shards, tgt_shards, tgt2_shards)

        for i, (src_shard, tgt_shard, tgt2_shard) in enumerate(shard_trips):
            return translator.translate_gold_diff(
               src=src_shard,
               tgt=tgt_shard,
               tgt2=tgt2_shard,
               src_dir=self.opt.src_dir,
               batch_size=self.opt.batch_size,
               batch_type=self.opt.batch_type,
               attn_debug=self.opt.attn_debug,
               align_debug=self.opt.align_debug,
               src_embed=src_embed)



def translate(opt):
    '''
    Returns source and baseline embeddings
    '''
    ArgumentParser.validate_translate_opts(opt)
    logger = init_logger(opt.log_file)

    translator = build_translator(opt, report_score=True)
    src_shards = split_corpus(opt.src, opt.shard_size)
    baseline_shards = split_corpus(opt.baseline, opt.shard_size)

    print("\nEmbedding source and baseline...\n")
   
    #Loop for src_embedding
    for i, src_shard in enumerate(src_shards):
        src_data = {"reader": translator.src_reader, "data": src_shard, "dir": opt.src_dir}
        _readers, _data, _dir = inputters.Dataset.config(
            [('src', src_data)])
       
        data = inputters.Dataset(
            translator.fields, readers=_readers, data=_data, dirs=_dir,
            sort_key=inputters.str2sortkey[translator.data_type],
            filter_pred=translator._filter_pred
        )

        data_iter = inputters.OrderedIterator(
            dataset=data,
            device=translator._dev,
            batch_size=opt.batch_size,
            batch_size_fn=translator.max_tok_len if opt.batch_type == "tokens" else None,
            train=False,
            sort=False,
            sort_within_batch=True,
            shuffle=False
        )

        for batch in data_iter:
            src, src_lengths = batch.src
            src_embed = translator.model.encoder.embed(src, src_lengths)

    #Loop for baseline_embedding
    for i, src_shard in enumerate(baseline_shards):
        src_data = {"reader": translator.src_reader, "data": src_shard, "dir": opt.src_dir}
        _readers, _data, _dir = inputters.Dataset.config(
            [('src', src_data)])
       
        data = inputters.Dataset(
            translator.fields, readers=_readers, data=_data, dirs=_dir,
            sort_key=inputters.str2sortkey[translator.data_type],
            filter_pred=translator._filter_pred
        )

        data_iter = inputters.OrderedIterator(
            dataset=data,
            device=translator._dev,
            batch_size=opt.batch_size,
            batch_size_fn=translator.max_tok_len if opt.batch_type == "tokens" else None,
            train=False,
            sort=False,
            sort_within_batch=True,
            shuffle=False
        )

        for batch in data_iter:
            src, src_lengths = batch.src
            bline_embed = translator.model.encoder.embed(src, src_lengths)
    return src_embed, bline_embed


def _get_parser():
    parser = ArgumentParser(description='translate.py')

    opts.config_opts(parser)
    opts.translate_opts(parser)
    return parser

def main():
    parser = _get_parser()
    opt = parser.parse_args()

    src_embed0, bline_embed0 = translate(opt)  #Get source and baseline embeddings
    src_embed0 = src_embed0.detach().numpy() 
    bline_embed0 = bline_embed0.detach().numpy()
 
    np.save("sear_emb.npy", src_embed0)  #Save as numpy arrays and reload as torch tensors
    np.save("baseline.npy", bline_embed0)
    src_embed = torch.from_numpy(np.load("sear_emb.npy"))
    baseline_embed = torch.from_numpy(np.load("baseline.npy"))
   
    #baseline_emb = baseline_embed
    baseline_emb = torch.zeros(src_embed.size())  #repeat '.' baseline src_embed.size()[0] times
    for i in range(src_embed.size()[0]):
        baseline_emb[i][0] = baseline_embed

    gold_scorer = GoldScorer(opt)
    grads = np.zeros(src_embed.size())
    steps = 50
    gdiffs = []
    scaled_inputs = [baseline_emb + i / steps * (src_embed - baseline_emb) for i in range(0, steps + 1)]
    #scaled_inputs = [baseline_emb + np.sin(2 * np.pi * i / steps) + i / steps * (src_embed - baseline_emb) for i in range(0, steps + 1)]
    print('\nGenerating Integrated Gradients...\n')
    for c, inp in enumerate(scaled_inputs):
        inp.requires_grad = True
        gold_diff = gold_scorer(inp)
        #gdiffs.append(gold_diff)
        if c == 0:
            min_diff = gold_diff.detach().numpy()[0]
            print(gold_diff)
        elif c == steps:
            max_diff = gold_diff.detach().numpy()[0]
            print(gold_diff)
        gold_scorer.zero_grad()
        grad = torch.autograd.grad(gold_diff, inp)[0].numpy()
        gdiffs.append(grad)
        grads += grad
    avg_grads = grads / steps
    IG = (src_embed.numpy() - baseline_emb.numpy()) * avg_grads
    #print(avg_grads)
    #print(np.sum(avg_grads))
    IG_norm = np.sum(IG, axis=2).squeeze(-1)
    print('\nNumber of IG steps: {}'.format(steps))
    print('Difference in target log probs: {:.3f}'.format(max_diff-min_diff))
    print('Sum of attributions: {:.3f}'.format(np.sum(IG_norm)))
  
    np.save("sear_IGs.npy", IG_norm)
    
    print('\n')
    with open(opt.src) as file:
         for line in file:
             for i, ch in enumerate(line.split()):
                 print((ch,IG_norm[i]))
if __name__ == "__main__":
    main()
