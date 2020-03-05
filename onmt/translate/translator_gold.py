#!/usr/bin/env python
""" Translator Class and builder """
from __future__ import print_function
import codecs
import os
import time
import numpy as np
from itertools import count, zip_longest

import torch

import onmt.model_builder
import onmt.inputters as inputters
import onmt.decoders.ensemble
from onmt.translate.beam_search import BeamSearch
from onmt.translate.greedy_search import GreedySearch
from onmt.utils.misc import tile, set_random_seed, report_matrix
from onmt.utils.alignment import extract_alignment, build_align_pharaoh
from onmt.modules.copy_generator import collapse_copy_scores


def build_translator(opt, report_score=True, logger=None, out_file=None):
    if out_file is None:
        out_file = codecs.open(opt.output, 'w+', 'utf-8')

    load_test_model = onmt.decoders.ensemble.load_test_model \
        if len(opt.models) > 1 else onmt.model_builder.load_test_model
    fields, model, model_opt = load_test_model(opt)

    scorer = onmt.translate.GNMTGlobalScorer.from_opt(opt)

    translator = Translator.from_opt(
        model,
        fields,
        opt,
        model_opt,
        global_scorer=scorer,
        out_file=out_file,
        report_align=opt.report_align,
        report_score=report_score,
        logger=logger
    )
    return translator


def max_tok_len(new, count, sofar):
    """
    In token batching scheme, the number of sequences is limited
    such that the total number of src/tgt tokens (including padding)
    in a batch <= batch_size
    """
    # Maintains the longest src and tgt length in the current batch
    global max_src_in_batch  # this is a hack
    # Reset current longest length at a new batch (count=1)
    if count == 1:
        max_src_in_batch = 0
        # max_tgt_in_batch = 0
    # Src: [<bos> w1 ... wN <eos>]
    max_src_in_batch = max(max_src_in_batch, len(new.src[0]) + 2)
    # Tgt: [w1 ... wM <eos>]
    src_elements = count * max_src_in_batch
    return src_elements


class Translator(object):
    """Translate a batch of sentences with a saved model.

    Args:
        model (onmt.modules.NMTModel): NMT model to use for translation
        fields (dict[str, torchtext.data.Field]): A dict
            mapping each side to its list of name-Field pairs.
        src_reader (onmt.inputters.DataReaderBase): Source reader.
        tgt_reader (onmt.inputters.TextDataReader): Target reader.
        gpu (int): GPU device. Set to negative for no GPU.
        n_best (int): How many beams to wait for.
        min_length (int): See
            :class:`onmt.translate.decode_strategy.DecodeStrategy`.
        max_length (int): See
            :class:`onmt.translate.decode_strategy.DecodeStrategy`.
        beam_size (int): Number of beams.
        random_sampling_topk (int): See
            :class:`onmt.translate.greedy_search.GreedySearch`.
        random_sampling_temp (int): See
            :class:`onmt.translate.greedy_search.GreedySearch`.
        stepwise_penalty (bool): Whether coverage penalty is applied every step
            or not.
        dump_beam (bool): Debugging option.
        block_ngram_repeat (int): See
            :class:`onmt.translate.decode_strategy.DecodeStrategy`.
        ignore_when_blocking (set or frozenset): See
            :class:`onmt.translate.decode_strategy.DecodeStrategy`.
        replace_unk (bool): Replace unknown token.
        data_type (str): Source data type.
        verbose (bool): Print/log every translation.
        report_time (bool): Print/log total time/frequency.
        copy_attn (bool): Use copy attention.
        global_scorer (onmt.translate.GNMTGlobalScorer): Translation
            scoring/reranking object.
        out_file (TextIO or codecs.StreamReaderWriter): Output file.
        report_score (bool) : Whether to report scores
        logger (logging.Logger or NoneType): Logger.
    """

    def __init__(
            self,
            model,
            fields,
            src_reader,
            tgt_reader,
            tgt2_reader,
            gpu=-1,
            n_best=1,
            min_length=0,
            max_length=100,
            ratio=0.,
            beam_size=30,
            random_sampling_topk=1,
            random_sampling_temp=1,
            stepwise_penalty=None,
            dump_beam=False,
            block_ngram_repeat=0,
            ignore_when_blocking=frozenset(),
            replace_unk=False,
            phrase_table="",
            data_type="text",
            verbose=False,
            report_time=False,
            copy_attn=False,
            global_scorer=None,
            out_file=None,
            report_align=False,
            report_score=True,
            logger=None,
            seed=-1):
        self.model = model
        self.fields = fields
        tgt_field = dict(self.fields)["tgt"].base_field
        self._tgt_vocab = tgt_field.vocab
        self._tgt_eos_idx = self._tgt_vocab.stoi[tgt_field.eos_token]
        self._tgt_pad_idx = self._tgt_vocab.stoi[tgt_field.pad_token]
        self._tgt_bos_idx = self._tgt_vocab.stoi[tgt_field.init_token]
        self._tgt_unk_idx = self._tgt_vocab.stoi[tgt_field.unk_token]
        self._tgt_vocab_len = len(self._tgt_vocab)

        self._gpu = gpu
        self._use_cuda = gpu > -1
        self._dev = torch.device("cuda", self._gpu) \
            if self._use_cuda else torch.device("cpu")

        self.n_best = n_best
        self.max_length = max_length

        self.beam_size = beam_size
        self.random_sampling_temp = random_sampling_temp
        self.sample_from_topk = random_sampling_topk

        self.min_length = min_length
        self.ratio = ratio
        self.stepwise_penalty = stepwise_penalty
        self.dump_beam = dump_beam
        self.block_ngram_repeat = block_ngram_repeat
        self.ignore_when_blocking = ignore_when_blocking
        self._exclusion_idxs = {
            self._tgt_vocab.stoi[t] for t in self.ignore_when_blocking}
        self.src_reader = src_reader
        self.tgt_reader = tgt_reader
        self.tgt2_reader = tgt2_reader
        self.replace_unk = replace_unk
        if self.replace_unk and not self.model.decoder.attentional:
            raise ValueError(
                "replace_unk requires an attentional decoder.")
        self.phrase_table = phrase_table
        self.data_type = data_type
        self.verbose = verbose
        self.report_time = report_time

        self.copy_attn = copy_attn

        self.global_scorer = global_scorer
        if self.global_scorer.has_cov_pen and \
                not self.model.decoder.attentional:
            raise ValueError(
                "Coverage penalty requires an attentional decoder.")
        self.out_file = out_file
        self.report_align = report_align
        self.report_score = report_score
        self.logger = logger

        self.use_filter_pred = False
        self._filter_pred = None

        # for debugging
        self.beam_trace = self.dump_beam != ""
        self.beam_accum = None
        if self.beam_trace:
            self.beam_accum = {
                "predicted_ids": [],
                "beam_parent_ids": [],
                "scores": [],
                "log_probs": []}

        set_random_seed(seed, self._use_cuda)

    @classmethod
    def from_opt(
            cls,
            model,
            fields,
            opt,
            model_opt,
            global_scorer=None,
            out_file=None,
            report_align=False,
            report_score=True,
            logger=None):
        """Alternate constructor.

        Args:
            model (onmt.modules.NMTModel): See :func:`__init__()`.
            fields (dict[str, torchtext.data.Field]): See
                :func:`__init__()`.
            opt (argparse.Namespace): Command line options
            model_opt (argparse.Namespace): Command line options saved with
                the model checkpoint.
            global_scorer (onmt.translate.GNMTGlobalScorer): See
                :func:`__init__()`..
            out_file (TextIO or codecs.StreamReaderWriter): See
                :func:`__init__()`.
            report_align (bool) : See :func:`__init__()`.
            report_score (bool) : See :func:`__init__()`.
            logger (logging.Logger or NoneType): See :func:`__init__()`.
        """

        src_reader = inputters.str2reader[opt.data_type].from_opt(opt)
        tgt_reader = inputters.str2reader["text"].from_opt(opt)
        tgt2_reader = inputters.str2reader["text"].from_opt(opt)
        return cls(
            model,
            fields,
            src_reader,
            tgt_reader,
            tgt2_reader,
            gpu=opt.gpu,
            n_best=opt.n_best,
            min_length=opt.min_length,
            max_length=opt.max_length,
            ratio=opt.ratio,
            beam_size=opt.beam_size,
            random_sampling_topk=opt.random_sampling_topk,
            random_sampling_temp=opt.random_sampling_temp,
            stepwise_penalty=opt.stepwise_penalty,
            dump_beam=opt.dump_beam,
            block_ngram_repeat=opt.block_ngram_repeat,
            ignore_when_blocking=set(opt.ignore_when_blocking),
            replace_unk=opt.replace_unk,
            phrase_table=opt.phrase_table,
            data_type=opt.data_type,
            verbose=opt.verbose,
            report_time=opt.report_time,
            copy_attn=model_opt.copy_attn,
            global_scorer=global_scorer,
            out_file=out_file,
            report_align=report_align,
            report_score=report_score,
            logger=logger,
            seed=opt.seed)

    def _log(self, msg):
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)

    def _gold_score(self, batch, memory_bank, src_lengths, src_vocabs,
                    use_src_map, enc_states, batch_size, src):
        if "tgt" in batch.__dict__:
            #gs = self._score_target(
            #    batch, memory_bank, src_lengths, src_vocabs,
            #    batch.src_map if use_src_map else None)
            gs_1, gs_2 = self._score_targets(
                batch, memory_bank, src_lengths, src_vocabs,
                batch.src_map if use_src_map else None)
            self.model.decoder.init_state(src, memory_bank, enc_states)
        else:
            gs = [0] * batch_size
        return gs_1, gs_2

    def translate_gold_diff(
            self,
            src,
            tgt=None,
            tgt2=None,
            src_dir=None,
            batch_size=None,
            batch_type="sents",
            attn_debug=False,
            align_debug=False,
            phrase_table=""):
        """Translate content of ``src`` and get gold scores from ``tgt``.

        Args:
            src: See :func:`self.src_reader.read()`.
            tgt: See :func:`self.tgt_reader.read()`.
            src_dir: See :func:`self.src_reader.read()` (only relevant
                for certain types of data).
            batch_size (int): size of examples per mini-batch
            attn_debug (bool): enables the attention logging
            align_debug (bool): enables the word alignment logging

        Returns:
            (`list`, `list`)

            * all_scores is a list of `batch_size` lists of `n_best` scores
            * all_predictions is a list of `batch_size` lists
                of `n_best` predictions
        """

        if batch_size is None:
            raise ValueError("batch_size must be set")

        src_data = {"reader": self.src_reader, "data": src, "dir": src_dir}
        tgt_data = {"reader": self.tgt_reader, "data": tgt, "dir": None}
        tgt2_data = {"reader": self.tgt2_reader, "data": tgt2, "dir": None}
        _readers, _data, _dir = inputters.Dataset.config(
            [('src', src_data), ('tgt', tgt_data), ('tgt2',tgt2_data)])

        data = inputters.Dataset(
            self.fields, readers=_readers, data=_data, dirs=_dir,
            sort_key=inputters.str2sortkey[self.data_type],
            filter_pred=self._filter_pred
        )

        data_iter = inputters.OrderedIterator(
            dataset=data,
            device=self._dev,
            batch_size=batch_size,
            batch_size_fn=max_tok_len if batch_type == "tokens" else None,
            train=False,
            sort=False,
            sort_within_batch=True,
            shuffle=False
        )

        xlation_builder = onmt.translate.TranslationBuilder(
            data, self.fields, self.n_best, self.replace_unk, tgt,
            self.phrase_table
        )

        # Statistics
        counter = count(1)
        pred_score_total, pred_words_total = 0, 0
        gold_score_total, gold_words_total = 0, 0

        all_scores = []
        all_predictions = []

        start_time = time.time()

        for batch in data_iter:
            gold_diff = self.translate_batch(
                batch, data.src_vocabs, attn_debug
            )
            return gold_diff


    def _align_pad_prediction(self, predictions, bos, pad):
        """
        Padding predictions in batch and add BOS.

        Args:
            predictions (List[List[Tensor]]): `(batch, n_best,)`, for each src
                sequence contain n_best tgt predictions all of which ended with
                eos id.
            bos (int): bos index to be used.
            pad (int): pad index to be used.

        Return:
            batched_nbest_predict (torch.LongTensor): `(batch, n_best, tgt_l)`
        """
        dtype, device = predictions[0][0].dtype, predictions[0][0].device
        flatten_tgt = [best.tolist() for bests in predictions
                       for best in bests]
        paded_tgt = torch.tensor(
            list(zip_longest(*flatten_tgt, fillvalue=pad)),
            dtype=dtype, device=device).T
        bos_tensor = torch.full([paded_tgt.size(0), 1], bos,
                                dtype=dtype, device=device)
        full_tgt = torch.cat((bos_tensor, paded_tgt), dim=-1)
        batched_nbest_predict = full_tgt.view(
            len(predictions), -1, full_tgt.size(-1))  # (batch, n_best, tgt_l)
        return batched_nbest_predict

    def _align_forward(self, batch, predictions):
        """
        For a batch of input and its prediction, return a list of batch predict
        alignment src indice Tensor in size ``(batch, n_best,)``.
        """
        # (0) add BOS and padding to tgt prediction
        if hasattr(batch, 'tgt'):
            batch_tgt_idxs = batch.tgt.transpose(1, 2).transpose(0, 2)
        else:
            batch_tgt_idxs = self._align_pad_prediction(
                predictions, bos=self._tgt_bos_idx, pad=self._tgt_pad_idx)
        tgt_mask = (batch_tgt_idxs.eq(self._tgt_pad_idx) |
                    batch_tgt_idxs.eq(self._tgt_eos_idx) |
                    batch_tgt_idxs.eq(self._tgt_bos_idx))

        n_best = batch_tgt_idxs.size(1)
        # (1) Encoder forward.
        src, enc_states, memory_bank, src_lengths = self._run_encoder(batch)

        # (2) Repeat src objects `n_best` times.
        # We use batch_size x n_best, get ``(src_len, batch * n_best, nfeat)``
        src = tile(src, n_best, dim=1)
        enc_states = tile(enc_states, n_best, dim=1)
        if isinstance(memory_bank, tuple):
            memory_bank = tuple(tile(x, n_best, dim=1) for x in memory_bank)
        else:
            memory_bank = tile(memory_bank, n_best, dim=1)
        src_lengths = tile(src_lengths, n_best)  # ``(batch * n_best,)``

        # (3) Init decoder with n_best src,
        self.model.decoder.init_state(src, memory_bank, enc_states)
        # reshape tgt to ``(len, batch * n_best, nfeat)``
        tgt = batch_tgt_idxs.view(-1, batch_tgt_idxs.size(-1)).T.unsqueeze(-1)
        dec_in = tgt[:-1]  # exclude last target from inputs
        _, attns = self.model.decoder(
            dec_in, memory_bank, memory_lengths=src_lengths, with_align=True)

        alignment_attn = attns["align"]  # ``(B, tgt_len-1, src_len)``
        # masked_select
        align_tgt_mask = tgt_mask.view(-1, tgt_mask.size(-1))
        prediction_mask = align_tgt_mask[:, 1:]  # exclude bos to match pred
        # get aligned src id for each prediction's valid tgt tokens
        alignement = extract_alignment(
            alignment_attn, prediction_mask, src_lengths, n_best)
        return alignement

    def translate_batch(self, batch, src_vocabs, attn_debug):
        """Translate a batch of sentences."""
        with torch.no_grad():
            if self.beam_size == 1:
                decode_strategy = GreedySearch(
                    pad=self._tgt_pad_idx,
                    bos=self._tgt_bos_idx,
                    eos=self._tgt_eos_idx,
                    batch_size=batch.batch_size,
                    min_length=self.min_length, max_length=self.max_length,
                    block_ngram_repeat=self.block_ngram_repeat,
                    exclusion_tokens=self._exclusion_idxs,
                    return_attention=attn_debug or self.replace_unk,
                    sampling_temp=self.random_sampling_temp,
                    keep_topk=self.sample_from_topk)
            else:
                # TODO: support these blacklisted features
                assert not self.dump_beam
                decode_strategy = BeamSearch(
                    self.beam_size,
                    batch_size=batch.batch_size,
                    pad=self._tgt_pad_idx,
                    bos=self._tgt_bos_idx,
                    eos=self._tgt_eos_idx,
                    n_best=self.n_best,
                    global_scorer=self.global_scorer,
                    min_length=self.min_length, max_length=self.max_length,
                    return_attention=attn_debug or self.replace_unk,
                    block_ngram_repeat=self.block_ngram_repeat,
                    exclusion_tokens=self._exclusion_idxs,
                    stepwise_penalty=self.stepwise_penalty,
                    ratio=self.ratio)
            return self._return_gold_diff(batch, src_vocabs,
                                                       decode_strategy)

    def _run_encoder(self, batch):
        src, src_lengths = batch.src if isinstance(batch.src, tuple) \
                           else (batch.src, None)

        enc_states, memory_bank, src_lengths = self.model.encoder(
            src, src_lengths)
        if src_lengths is None:
            assert not isinstance(memory_bank, tuple), \
                'Ensemble decoding only supported for text data'
            src_lengths = torch.Tensor(batch.batch_size) \
                               .type_as(memory_bank) \
                               .long() \
                               .fill_(memory_bank.size(0))
        return src, enc_states, memory_bank, src_lengths


    def _return_gold_diff(
            self,
            batch,
            src_vocabs,
            decode_strategy):
        """Translate a batch of sentences step by step using cache.

        Args:
            batch: a batch of sentences, yield by data iterator.
            src_vocabs (list): list of torchtext.data.Vocab if can_copy.
            decode_strategy (DecodeStrategy): A decode strategy to use for
                generate translation step by step.

        Returns:
            results (dict): The translation results.
        """
        # (0) Prep the components of the search.
        use_src_map = self.copy_attn
        parallel_paths = decode_strategy.parallel_paths  # beam_size
        batch_size = batch.batch_size

        # (1) Run the encoder on the src.
        src, enc_states, memory_bank, src_lengths = self._run_encoder(batch)
        self.model.decoder.init_state(src, memory_bank, enc_states)

        gold_scores_1, gold_scores_2 = self._gold_score(
                                       batch, memory_bank, src_lengths, src_vocabs, use_src_map,
                                       enc_states, batch_size, src)
        return gold_scores_1 - gold_scores_2

    def _score_targets(self, batch, memory_bank, src_lengths,
                      src_vocabs, src_map):
        tgt = batch.tgt
        tgt_in = tgt[:-1]

        log_probs, attn = self._decode_and_generate(
            tgt_in, memory_bank, batch, src_vocabs,
            memory_lengths=src_lengths, src_map=src_map)

        log_probs[:, :, self._tgt_pad_idx] = 0
        gold = tgt[1:]
        gold_scores = log_probs.gather(2, gold)
        gold_scores = gold_scores.sum(dim=0).view(-1)

        tgt2 = batch.tgt2
        tgt2_in = tgt2[:-1]

        log_probs, attn = self._decode_and_generate(
            tgt2_in, memory_bank, batch, src_vocabs,
            memory_lengths=src_lengths, src_map=src_map)

        log_probs[:, :, self._tgt_pad_idx] = 0
        gold = tgt2[1:]
        gold_scores = log_probs.gather(2, gold)
        gold_scores_2 = gold_scores.sum(dim=0).view(-1)
        return gold_scores, gold_scores_2

    def _report_score(self, name, score_total, words_total):
        if words_total == 0:
            msg = "%s No words predicted" % (name,)
        else:
            avg_score = score_total / words_total
            ppl = np.exp(-score_total.item() / words_total)
            msg = ("%s AVG SCORE: %.4f, %s PPL: %.4f" % (
                name, avg_score,
                name, ppl))
        return msg
