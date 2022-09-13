import itertools
import collections
import random
import re
import glob
import math
import os
import pickle


import pyranges as pr
import gzip
import kipoiseq
import pyfaidx
import pandas as pd
import numpy as np

import Code.Utilities.utils as utils
import Code.Utilities.seq_utils as seq_utils
import Code.Utilities.enformer_utils as enformer_utils

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--fasta-path")
parser.add_argument("--data-path") # "Data/Cohen_genomic_environments/"
parser.add_argument("--number-of-jobs", type=int) 
parser.add_argument("--output-dir")
args = parser.parse_args()

"""This script creates samples for the Cohen PatchMPRA study"""

fasta_extractor = enformer_utils.FastaStringExtractor(args.fasta_path)

# get the insert sequence components
with open(os.path.join(args.data_path, "mScarlet.txt")) as file:
    lines = file.readlines()[1:]
    mScarlet_seq = "".join(lines).replace('\n', '').upper()
enhancer = "TGCCCCCCTTCTTCCTATGTCTGATGGAGTTTCCTCTCTAAGTAGCCATTTTATTCTGCTGACTCACCCTCTAACTCCCGGTCTTATTCCATCCTGCCTCAGGGTCTGTGGTGTAGTCATAGCAC"

# get the landing pad and expression data
landing_pads = pd.read_csv(os.path.join(args.data_path, "landing_pad_locations.tsv"), sep="\t").dropna()
expression = pd.read_csv(os.path.join(args.data_path, "patchMPRA_expression.tsv"), sep="\t")
prom_sequences = pd.read_csv(os.path.join(args.data_path, "sequences.tsv"), sep="\t")
prom_sequences["Orientation"] = prom_sequences["strand"].apply(lambda x: "RC" if x == "-" else "SS")

lp_info = []
for row in landing_pads.iterrows():
    row = row[1]
    interval_str = row["Liftover"]  
    lp_interval = kipoiseq.Interval(chrom=interval_str.split(":")[0],
                               start=int(interval_str.split(":")[1].split("-")[0].replace(",","")),
                               end=int(interval_str.split(":")[1].split("-")[1].replace(",","")))
    lp_interval_mid = (lp_interval.start + lp_interval.stop)//2
    lp_info.append({"LP_name":row["Hong"],
                  "LP_interval":lp_interval})
lp_info = pd.DataFrame(lp_info).sort_values('LP_name')

crs_sequences = expression.merge(prom_sequences, on="oligo_id")

def cohen_sample_generator_factory(fasta_extractor):
    for crs_row in crs_sequences.iterrows():
        crs_row = crs_row[1]
        lp = crs_row["LP"]
        lp_row = lp_info.query('LP_name == @lp').iloc[0]
        crs = crs_row["sequence"]
        insert = enhancer + crs + mScarlet_seq
        landmark = len(enhancer) + len(crs)//2
        ideal_offset = seq_utils.compute_offset_to_center_landmark(landmark, insert)
        for offset in [-64, -32, 0]:
            modified_sequence, minbin, maxbin, landmarkbin = \
                enformer_utils.insert_sequence_at_landing_pad(insert,
                                                              lp_row["LP_interval"],
                                                              fasta_extractor,
                                                              shift_five_end=ideal_offset + offset,
                                                              landmark=landmark)
            yield modified_sequence, {"oligo_id":crs_row["oligo_id"],
                                      "LP_name":lp_row["LP_name"],
                                      "minbin":minbin,
                                      "maxbin":maxbin,
                                      "landmarkbin":landmarkbin,
                                      "offset":offset
                                      }

# prepare generator
sample_generator = \
    cohen_sample_generator_factory(fasta_extractor=fasta_extractor)
# write jobs
num_of_samples = len(crs_sequences.index)*3

utils.batcherator(sample_generator, num_of_samples, args.number_of_jobs, args.output_dir)

# assert that we produced the right number of files
# should have number_of_jobs many pkls
assert args.number_of_jobs == len(glob.glob(os.path.join(args.output_dir,"*.pkl"))), \
       "Number of output files does not match the expected number."
