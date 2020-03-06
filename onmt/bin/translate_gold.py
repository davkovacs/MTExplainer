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

class GoldScorer(nn.Module):
    def __init__(self, opt):
        # tgt and tgt2: two target somethings
        super(GoldScorer, self).__init__()
        self.gold_scorer = TranslateGoldDiff(opt)

    def forward(self, src_embed):
        return self.gold_scorer(src_embed)

class TranslateGoldDiff(object):
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

#def IG_attributions(gold_diff):
    #IGs = torch.autograd.grad(gold_diff, src_embed)
    #print(IGs)

def main():
    parser = _get_parser()
    opt = parser.parse_args()
    src_embed = translate(opt)
    gold_scorer = GoldScorer(opt)
    gold_diff = gold_scorer(src_embed)
    print(gold_diff)
    #gold_diff.requires_grad = True
    gold_diff.backward()
    #print(gold_diff.grad_fn)
    print(src_embed.grad)
    #print(torch.autograd.grad(gold_diff, src_embed))
    #IG_attributions( gold_diff)


if __name__ == "__main__":
    main()
