#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals
import logging

import torch
import torch.nn as nn
import numpy as np
import onmt.opts as opts
import onmt.inputters as inputters
from onmt.utils.misc import split_corpus
from onmt.utils.parse import ArgumentParser
from onmt.utils.logging import init_logger
from onmt.translate.translator_gold import build_translator
import tqdm


class GoldScorer(nn.Module):
    # nn to produce the probability of tgt
    def __init__(self, opt):
        super(GoldScorer, self).__init__()
        self.gold_scorer = TranslateGold(opt)

    def forward(self, src_embed):
        return self.gold_scorer(src_embed)


class TranslateGold(object):
    # functor returning probability of tgt given src_embed
    def __init__(self, opt):
        self.opt = opt

    def __call__(self, src_embed, gen_hidden_states=False):
        translator = build_translator(self.opt, report_score=True)
        src_shards = split_corpus(self.opt.src, self.opt.shard_size)
        tgt_shards = split_corpus(self.opt.tgt, self.opt.shard_size)
        shard_trips = zip(src_shards, tgt_shards)

        for i, (src_shard, tgt_shard) in enumerate(shard_trips):
            return translator.translate_gold_diff(
                src=src_shard,
                tgt=tgt_shard,
                src_dir=self.opt.src_dir,
                batch_size=self.opt.batch_size,
                batch_type=self.opt.batch_type,
                attn_debug=self.opt.attn_debug,
                align_debug=self.opt.align_debug,
                src_embed=src_embed)


def translate(opt):
    """ Returns source embeddings"""

    ArgumentParser.validate_translate_opts(opt)
    logger = init_logger(opt.log_file)

    translator = build_translator(opt, report_score=True)
    src_shards = split_corpus(opt.src, opt.shard_size)

    # Loop for src_embedding
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

    return src_embed


def _get_parser():
    parser = ArgumentParser(description='translate.py')
    opts.config_opts(parser)
    opts.translate_opts(parser)
    return parser


def main():
    parser = _get_parser()
    opt = parser.parse_args()

    src_embed = translate(opt)  # Get source embeddings

    gold_scorer = GoldScorer(opt)
    score = gold_scorer(src_embed)
    try:
        score_list = np.load(opt.score_file)
    except FileNotFoundError:
        score_list=np.ndarray([])
    score_list = np.append(score_list, score.detach().numpy()[0])
    np.save(opt.score_file, score_list)

if __name__ == "__main__":
    main()
