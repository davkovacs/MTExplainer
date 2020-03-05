#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals

from onmt.utils.logging import init_logger
from onmt.utils.misc import split_corpus
from onmt.translate.translator_gold import build_translator

import onmt.opts as opts
from onmt.utils.parse import ArgumentParser


def translate(opt):
    ArgumentParser.validate_translate_opts(opt)
    logger = init_logger(opt.log_file)

    translator = build_translator(opt, report_score=True)
    src_shards = split_corpus(opt.src, opt.shard_size)
    tgt_shards = split_corpus(opt.tgt, opt.shard_size)
    tgt2_shards = split_corpus(opt.tgt2, opt.shard_size)
    shard_trips = zip(src_shards, tgt_shards, tgt2_shards)

    for i, (src_shard, tgt_shard, tgt2_shard) in enumerate(shard_trips):
        logger.info("Translating shard %d." % i)
        return translator.translate_gold_diff(
               src=src_shard,
               tgt=tgt_shard,
               tgt2=tgt2_shard,
               src_dir=opt.src_dir,
               batch_size=opt.batch_size,
               batch_type=opt.batch_type,
               attn_debug=opt.attn_debug,
               align_debug=opt.align_debug
               )


def _get_parser():
    parser = ArgumentParser(description='translate.py')

    opts.config_opts(parser)
    opts.translate_opts(parser)
    return parser


def main():
    parser = _get_parser()

    opt = parser.parse_args()
    translate(opt)


if __name__ == "__main__":
    main()
