import itertools
import collections
import random
import re
import glob
import math
import os

import pyranges as pr
import gzip
import kipoiseq
from kipoiseq import Interval
import pyfaidx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

import tensorflow_hub as hub
import tensorflow as tf

# length of sequence which basenji2 gets as input
# ═════┆═════┆════════════════════════┆═════┆═════
SEQUENCE_LENGTH = 131072
# length of central sequence which basenji2 actually sees (1024 bins)
# ─────┆═════┆════════════════════════┆═════┆─────
SEEN_SEQUENCE_LENGTH = 1024*128
# length of central sequence for which enformer gives predictions (1024 bins)
# ─────┆─────┆════════════════════════┆─────┆─────
PRED_SEQUENCE_LENGTH = 1024*128
# padding (only one side!) until the PRED_SEQUENCE_LENGTH window
# ═════┆═════┆────────────────────────┆═════┆═════
PADDING = (SEQUENCE_LENGTH - PRED_SEQUENCE_LENGTH)//2
# padding (only one side!) until the SEEN_SEQUENCE_LENGTH window
# ═════┆─────┆────────────────────────┆─────┆═════
PADDING_UNTIL_SEEN = (SEQUENCE_LENGTH - SEEN_SEQUENCE_LENGTH)//2
# padding (only one side!) from PADDING_UNTIL_SEEN to PRED_SEQUENCE_LENGTH
# ─────┆═════┆────────────────────────┆═════┆─────
PADDING_SEEN = PADDING - PADDING_UNTIL_SEEN

assert 2*(PADDING_UNTIL_SEEN + PADDING_SEEN) + PRED_SEQUENCE_LENGTH == SEQUENCE_LENGTH, \
    "All parts should add up to SEQUENCE_LENGTH"
assert PADDING_UNTIL_SEEN + PADDING_SEEN == PADDING, \
    "All padding parts should add up to PADDING"
assert PRED_SEQUENCE_LENGTH + 2*(PADDING_SEEN) == SEEN_SEQUENCE_LENGTH, \
    "All SEEN_SEQUENCE parts should add up to SEEN_SEQUENCE_LENGTH"

import Code.Utilities.seq_utils as seq_utils

"""Custom layers"""

class StochasticShift(tf.keras.layers.Layer):
    """Stochastically shift a one hot encoded DNA sequence."""
    def __init__(self, shift_max=0, symmetric=True, pad='uniform', **params):
        super(StochasticShift, self).__init__()
        self.shift_max = shift_max
        self.symmetric = symmetric
        if self.symmetric:
            self.augment_shifts = tf.range(-self.shift_max, self.shift_max+1)
        else:
            self.augment_shifts = tf.range(0, self.shift_max+1)
        self.pad = pad

    def call(self, seq_1hot, training=None):
        if training:
            shift_i = tf.random.uniform(shape=[], minval=0, dtype=tf.int64,
                                      maxval=len(self.augment_shifts))
            shift = tf.gather(self.augment_shifts, shift_i)
            sseq_1hot = tf.cond(tf.not_equal(shift, 0),
                              lambda: shift_sequence(seq_1hot, shift),
                              lambda: seq_1hot)
            return sseq_1hot
        else:
            return seq_1hot

    def get_config(self):
        config = super().get_config().copy()
        config.update({
          'shift_max': self.shift_max,
          'symmetric': self.symmetric,
          'pad': self.pad
        })
        return config

def shift_sequence(seq, shift, pad_value=0):
    """Shift a sequence left or right by shift_amount.
    Args:
    seq: [batch_size, seq_length, seq_depth] sequence
    shift: signed shift value (tf.int32 or int)
    pad_value: value to fill the padding (primitive or scalar tf.Tensor)
    """
    if seq.shape.ndims != 3:
        raise ValueError('input sequence should be rank 3')
    input_shape = seq.shape

    pad = pad_value * tf.ones_like(seq[:, 0:tf.abs(shift), :])

    def _shift_right(_seq):
        # shift is positive
        sliced_seq = _seq[:, :-shift:, :]
        return tf.concat([pad, sliced_seq], axis=1)

    def _shift_left(_seq):
        # shift is negative
        sliced_seq = _seq[:, -shift:, :]
        return tf.concat([sliced_seq, pad], axis=1)

    sseq = tf.cond(tf.greater(shift, 0),
                 lambda: _shift_right(seq),
                 lambda: _shift_left(seq))
    sseq.set_shape(input_shape)

    return sseq

class GELU(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(GELU, self).__init__(**kwargs)
    def call(self, x):
        # return tf.keras.activations.sigmoid(1.702 * x) * x
        return tf.keras.activations.sigmoid(tf.constant(1.702) * x) * x

"""Code to handle intervals"""

def position_to_bin(pos, 
                    pos_type="relative",
                    target_interval=None):
    if pos_type == "absolute":
        pos = pos - target_interval.start
    elif pos_type == "relative_padded":
        pos = pos - PADDING
    return pos//128

def bin_to_position(bin,
                    pos_type="relative",
                    target_interval=None):
    pos = bin*128 + 64
    if pos_type == "absolute":
        pos = pos + target_interval.start
    elif pos_type == "relative_padded":
        pos = pos + PADDING
    return pos

"""Code to handle inserting sequences into the genome"""

def insert_variant_centred_on_tss(tss_interval,
                                  variant,
                                  allele,
                                  fasta_extractor, 
                                  shift_five_end=0,
                                  rev_comp=False):
    return seq_utils.insert_variant_centred_on_tss(tss_interval,
                                                   variant,
                                                   allele,
                                                   fasta_extractor,
                                                   shift_five_end=shift_five_end,
                                                   SEQUENCE_LENGTH=SEQUENCE_LENGTH,
                                                   PADDING=PADDING,
                                                   binsize=128,
                                                   rev_comp=rev_comp)

def pad_sequence(insert,
                shift_five_end=0,
                landmark=0,
                rev_comp=False):
    return seq_utils.pad_sequence(insert,
                                shift_five_end=shift_five_end,
                                SEQUENCE_LENGTH=SEQUENCE_LENGTH,
                                PADDING=PADDING,
                                binsize=128,
                                landmark=landmark,
                                rev_comp=rev_comp)

def insert_sequence_at_landing_pad(insert,
                                   lp_interval,
                                   fasta_extractor,
                                   mode="center",
                                   shift_five_end=0,
                                   landmark=0,
                                   rev_comp=False,
                                   shuffle=False):
    return seq_utils.insert_sequence_at_landing_pad(insert,
                                                    lp_interval,
                                                    fasta_extractor,
                                                    mode=mode,
                                                    shift_five_end=shift_five_end,
                                                    SEQUENCE_LENGTH=SEQUENCE_LENGTH,
                                                    PADDING=PADDING,
                                                    binsize=128,
                                                    landmark=landmark,
                                                    rev_comp=rev_comp,
                                                    shuffle=shuffle)
