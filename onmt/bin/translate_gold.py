#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import torch
from onmt.utils.logging import init_logger
from onmt.utils.misc import split_corpus
from onmt.translate.translator_gold import build_translator
import onmt.inputters as inputters

import onmt.opts as opts
from onmt.utils.parse import ArgumentParser
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

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
    ArgumentParser.validate_translate_opts(opt)
    logger = init_logger(opt.log_file)

    translator = build_translator(opt, report_score=True)
    src_shards = split_corpus(opt.src, opt.shard_size)
    tgt_shards = split_corpus(opt.tgt, opt.shard_size)
    tgt2_shards = split_corpus(opt.tgt2, opt.shard_size)
    shard_trips = zip(src_shards, tgt_shards, tgt2_shards)

    for i, (src_shard, tgt_shard, tgt2_shard) in enumerate(shard_trips):
        src_data = {"reader": translator.src_reader, "data": src_shard, "dir": opt.src_dir}
        tgt_data = {"reader": translator.tgt_reader, "data": tgt_shard, "dir": None}
        _readers, _data, _dir = inputters.Dataset.config(
            [('src', src_data), ('tgt', tgt_data)])

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

        logger.info("Translating shard %d." % i)

        return src_embed


"""
        return translator.translate_gold_diff(
               src=src_shard,
               tgt=tgt_shard,
               tgt2=tgt2_shard,
               src_dir=opt.src_dir,
               batch_size=opt.batch_size,
               batch_type=opt.batch_type,
               attn_debug=opt.attn_debug,
               align_debug=opt.align_debug,
               src_embed=src_embed)"""


def _get_parser():
    parser = ArgumentParser(description='translate.py')

    opts.config_opts(parser)
    opts.translate_opts(parser)
    return parser

def main():
    parser = _get_parser()
    opt = parser.parse_args()
    src_embed0 = translate(opt)
    src_embed0 = src_embed0.detach().numpy()
    np.save("sear_emb.npy", src_embed0)
    src_embed = torch.from_numpy(np.load("sear_emb.npy"))
    baseline_embed = torch.from_numpy(np.load("baseline.npy"))
    #baseline_emb = baseline_embed
    baseline_emb = torch.zeros(src_embed.size())
    for i in range(src_embed.size()[0]):
        baseline_emb[i][0] = baseline_embed
    gold_scorer = GoldScorer(opt)
    grads = np.zeros(src_embed.size())
    steps = 50
    gdiffs = []
    scaled_inputs = [baseline_emb + i / steps * (src_embed - baseline_emb) for i in range(0, steps + 1)]
    #scaled_inputs = [baseline_emb + np.sin(2 * np.pi * i / steps) + i / steps * (src_embed - baseline_emb) for i in range(0, steps + 1)]
    for c, inp in enumerate(scaled_inputs):
        inp.requires_grad = True
        gold_diff = gold_scorer(inp)
        #gdiffs.append(gold_diff)
        if c == 0:
            print(gold_diff)
        elif c == steps:
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
    print(np.sum(IG_norm))
    print(IG_norm)
    #print(np.sum(IG_norm))
    np.save("sear_IGs.npy", IG_norm)

if __name__ == "__main__":
    main()
