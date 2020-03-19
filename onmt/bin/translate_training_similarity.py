#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import os

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
import tqdm


class GoldScorerHidden(nn.Module):
    # nn to produce the probability difference between tgt1 and tgt2 from enc hidden state
    def __init__(self, opt):
        super(GoldScorerHidden, self).__init__()
        self.gold_scorer = GoldHidden(opt)

    def forward(self, hidden_st):
        return self.gold_scorer(hidden_state=hidden_st)


class GoldHidden(object):
    # functor returning prob difference between tgt1 and tgt2 given encoder hidden state of src
    def __init__(self, opt):
        self.opt = opt

    def __call__(self, src_embed=None, hidden_state=None):
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
               src_embed=src_embed,
               hidden_state=hidden_state)


def return_hiddens(opt):
    """ Returns source and baseline hidden states"""

    ArgumentParser.validate_translate_opts(opt)
    logger = init_logger(opt.log_file)

    translator = build_translator(opt, report_score=True)
    src_shards = split_corpus(opt.src, opt.shard_size)
    baseline_shards = split_corpus(opt.baseline, opt.shard_size)

    print("\nGenerating source and baseline hidden states...\n")

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
            src_embed, src_hidden, src_lengths = translator.model.encoder(src, src_lengths)

    # Loop for baseline_embedding
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
            bline_embed, bline_hidden, bline_lengths = translator.model.encoder(src, src_lengths)
    return src_hidden, bline_hidden


def _get_parser():
    parser = ArgumentParser(description='translate.py')

    opts.config_opts(parser)
    opts.translate_opts(parser)
    return parser


def main():
    parser = _get_parser()
    opt = parser.parse_args()

    src_hidden0, bline_hidden0 = return_hiddens(opt)  # Get source and baseline embeddings
    src_hidden0 = src_hidden0.detach().numpy()
    bline_hidden0 = bline_hidden0.detach().numpy()
    np.save("sear_hidden.npy", src_hidden0)  # Save as numpy arrays and reload as torch tensors
    np.save("baseline_hidden.npy", bline_hidden0)

    src_hidden = torch.from_numpy(np.load("sear_hidden.npy"))
    baseline_hidden = torch.from_numpy(np.load("baseline_hidden.npy"))

    baseline_hid = torch.zeros(src_hidden.size())  # repeat '.' baseline src_embed.size()[0] times
    for i in range(src_hidden.size()[0]):
        baseline_hid[i][0] = baseline_hidden

    gold_hidden_scorer = GoldScorerHidden(opt)
    grads = np.zeros(src_hidden.size())
    steps = int(opt.n_ig_steps)
    gdiffs = []
    scaled_inputs = [baseline_hid + i / steps * (src_hidden - baseline_hid) for i in range(0, steps + 1)]
    # scaled_inputs = [baseline_emb + np.sin(2 * np.pi * i / steps) + i / steps * (src_embed - baseline_emb) for i in range(0, steps + 1)]
    print('Generating Integrated Gradients...\n')
    for c, inp in enumerate(tqdm.tqdm(scaled_inputs)):
        inp.requires_grad = True
        gold_diff = gold_hidden_scorer(inp)
        # gdiffs.append(gold_diff)
        if c == 0:
            min_diff = gold_diff.detach().numpy()[0]
            # print(gold_diff)
        elif c == steps:
            max_diff = gold_diff.detach().numpy()[0]
            # print(gold_diff)
        gold_hidden_scorer.zero_grad()
        grad = torch.autograd.grad(gold_diff, inp)[0].numpy()
        gdiffs.append(grad)
        grads += grad
    avg_grads = grads / steps
    IG = (src_hidden.numpy() - baseline_hid.numpy()) * avg_grads
    # print(avg_grads)
    # print(np.sum(avg_grads))
    #IG_norm = np.sum(IG, axis=2).squeeze(-1)
    print('\nNumber of IG steps: {}'.format(steps))
    print('Difference in target log probs: {:.3f}'.format(max_diff - min_diff))
    print('Sum of attributions: {:.3f}'.format(np.sum(IG)))

    np.save(opt.output, IG)



if __name__ == "__main__":
    main()