#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals
import logging

import torch
import torch.nn as nn
import numpy as np
import onmt.opts as opts
import torch.optim as optim
import onmt.inputters as inputters
from onmt.utils.misc import split_corpus
from onmt.utils.parse import ArgumentParser
from onmt.utils.logging import init_logger
from onmt.utils.loss import build_loss_compute
from onmt.inputters.inputter import build_dataset_iter
from onmt.translate.translator_gold import build_translator
import tqdm
import copy


def train(translator, batch, opt):
    translator.model.train()  # set to train mode
    #load training data in opt as batch?
    src, src_lengths = batch.src
    tgt = batch.tgt

    tgt_field = dict(translator.fields)["tgt"].base_field
    loss_func = build_loss_compute(translator.model, tgt_field, opt)
    params = [p for p in translator.model.parameters() if p.requires_grad]

    optimizer = optim.SGD(params, lr=opt.learning_rate)
    for i in range(opt.train_steps):
        optimizer.zero_grad()

        outputs, attns = translator.model.forward(src, tgt, src_lengths, bptt=False,
                                                  with_align=False)
        loss, batch_stats = loss_func(
            batch,
            outputs,
            attns,
            normalization=1,  # should be fine? see 159 of trainer.py
            shard_size=1,
            unlearn=True)
        #if loss is not None:
        #    optimizer.backward(loss)
        #    loss.backward()
        #else:
        #    print('Oh no, Loss is None!')
        optimizer.step()

def _get_parser():
    parser = ArgumentParser(description='translate_unlearn_fast.py')
    opts.config_opts(parser)
    opts.model_opts(parser)
    opts.untrain_opts(parser)
    return parser


def main():
    parser = _get_parser()
    opt = parser.parse_args()

    translator = build_translator(opt, report_score=True) # TODO check report_score

    # TODO delete this if the assert is working
    #checkpoint = torch.load(opt.train_from,
    #                        map_location=lambda storage, loc: storage)
    #fields = checkpoint['vocab']

    #assert fields == translator.fields
    train_iter = build_dataset_iter("train", translator.fields, opt, is_train=False)

    score_list = []
    for num, batch in enumerate(train_iter):

        translator_copy = copy.deepcopy(translator)

        train(translator_copy, batch, opt)  # should update model weights

        # Set to translate mode
        translator_copy.model.eval()
        translator_copy.model.generator.eval()

        src_shards = split_corpus(opt.src, opt.shard_size)
        tgt_shards = split_corpus(opt.tgt, opt.shard_size)
        shard_trips = zip(src_shards, tgt_shards)

        for i, (src_shard, tgt_shard) in enumerate(shard_trips):
            score = translator_copy.translate_gold_diff(
                src=src_shard,
                tgt=tgt_shard,
                src_dir=opt.src_dir,
                batch_size=opt.batch_size,
                batch_type=opt.batch_type,
                attn_debug=opt.attn_debug,
                align_debug=opt.align_debug,
                unlearn=True)
        score_list.append(score.detach()[0])
        if num % 500 == 1:
            np.save(opt.score_file, score_list)
    np.save(opt.score_file, score_list) # TODO make sure to choose different score_file to avoid overwriting previous results

if __name__ == "__main__":
    main()
