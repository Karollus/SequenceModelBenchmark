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

# length of sequence which enformer gets as input
# ═════┆═════┆════════════════════════┆═════┆═════
SEQUENCE_LENGTH = 393216
# length of central sequence which enformer actually sees (1536 bins)
# ─────┆═════┆════════════════════════┆═════┆─────
SEEN_SEQUENCE_LENGTH = 1536*128
# length of central sequence for which enformer gives predictions (896 bins)
# ─────┆─────┆════════════════════════┆─────┆─────
PRED_SEQUENCE_LENGTH = 896*128
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

class Enformer:

    def __init__(self, tfhub_url):
        self._model = hub.load(tfhub_url).model

    def predict_on_batch(self, inputs):
        predictions = self._model.predict_on_batch(inputs)
        return {k: v.numpy() for k, v in predictions.items()}

    @tf.function
    def contribution_input_grad(self, input_sequence,
                              target_mask, output_head='human'):
        input_sequence = input_sequence[tf.newaxis]

        target_mask_mass = tf.reduce_sum(target_mask)
        with tf.GradientTape() as tape:
            tape.watch(input_sequence)
            prediction = tf.reduce_sum(
              target_mask[tf.newaxis] *
              self._model.predict_on_batch(input_sequence)[output_head]) / target_mask_mass

        input_grad = tape.gradient(prediction, input_sequence) * input_sequence
        input_grad = tf.squeeze(input_grad, axis=0)
        return tf.reduce_sum(input_grad, axis=-1)

"""The fasta string extractor for enformer"""
    
class FastaStringExtractor:
    
    def __init__(self, fasta_file):
        self.fasta = pyfaidx.Fasta(fasta_file)
        self._chromosome_sizes = {k: len(v) for k, v in self.fasta.items()}

    def extract(self, interval: Interval, safe_mode=True, **kwargs) -> str:
        chromosome_length = self._chromosome_sizes[interval.chrom]
        # if interval is completely outside chromosome boundaries ...
        if (interval.start < 0 and interval.end < 0 or
            interval.start >= chromosome_length and interval.end > chromosome_length):
            if safe_mode:  # if safe mode is on: fail
                raise ValueError("Interval outside chromosome boundaries")
            else:  # if it's off: return N-sequence
                return interval.width() * 'N'
        # ... else interval (at least!) overlaps chromosome boundaries
        else:
            # Truncate interval if it extends beyond the chromosome lengths.
            trimmed_interval = Interval(interval.chrom,
                                        max(interval.start, 0),
                                        min(interval.end, chromosome_length),
                                        )
            # pyfaidx wants a 1-based interval
            sequence = str(self.fasta.get_seq(trimmed_interval.chrom,
                                            trimmed_interval.start + 1,
                                            trimmed_interval.stop).seq).upper()
            # Fill truncated values with N's.
            pad_upstream = 'N' * max(-interval.start, 0)
            pad_downstream = 'N' * max(interval.end - chromosome_length, 0)
            return pad_upstream + sequence + pad_downstream

    def close(self):
        return self.fasta.close()


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

def extract_refseq_centred_at_landmark(landmark_interval, 
                                       fasta_extractor, 
                                       shift_five_end=0,
                                       rev_comp=False):
    return seq_utils.extract_refseq_centred_at_landmark(landmark_interval,
                                                        fasta_extractor,
                                                        shift_five_end=shift_five_end,
                                                        SEQUENCE_LENGTH=SEQUENCE_LENGTH,
                                                        PADDING=PADDING,
                                                        binsize=128,
                                                        rev_comp=rev_comp)

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
    
""" Visualization """

def plot_tracks(tracks, interval, height=1.5):
    fig, axes = plt.subplots(len(tracks), 1, figsize=(20, height * len(tracks)), sharex=True)
    for ax, (title, y) in zip(axes, tracks.items()):
        ax.fill_between(np.linspace(interval.start, interval.end, num=len(y)), y)
        ax.set_title(title)
        sns.despine(top=True, right=True, bottom=True)
    ax.set_xlabel(str(interval))
    plt.tight_layout()
