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

import kipoi

import tensorflow_hub as hub
import tensorflow as tf

# length of sequence which basenji1 gets as input
# ═════┆═════┆════════════════════════┆═════┆═════
SEQUENCE_LENGTH = 131072
# length of central sequence which basenji1 actually sees (1024 bins)
# ─────┆═════┆════════════════════════┆═════┆─────
SEEN_SEQUENCE_LENGTH = 1024*128
# length of central sequence for which enformer gives predictions (960 bins)
# ─────┆─────┆════════════════════════┆─────┆─────
PRED_SEQUENCE_LENGTH = 960*128
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
