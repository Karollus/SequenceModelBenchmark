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
parser.add_argument("--data-path") # "Data/Segal_promoter/"
parser.add_argument("--number-of-jobs", type=int)
parser.add_argument("--output-dir")
args = parser.parse_args()


"""This script does an SIM study of some segal promoters"""

fasta_extractor = enformer_utils.FastaStringExtractor(args.fasta_path)

# collect the data
segal_df = pd.read_csv(
    os.path.join(args.data_path, "GSM3323461_oligos_measurements_processed_data.tab"),
    sep="\t")
elements_df = pd.read_csv(os.path.join(args.data_path, "synthetic_configurations_of_core_promoters.tsv"),sep="\t").rename(columns={"GFP_RFP_ratio(mean exp)":"mean_exp"})
tata_shift_df = pd.read_csv(os.path.join(args.data_path, "tata_inr_shift.tsv"),sep="\t").rename(columns={"GFP_RFP_ratio(mean exp)":"mean_exp"})
tf_activity_df = pd.read_csv(os.path.join(args.data_path, "tf_activity_screen.tsv"), sep="\t").rename(columns={"GFP_RFP_ratio(mean exp)":"mean_exp"})

#collect sequences
# AAVS1: (PPP1R12C-201, intron 1)
aavs1 = kipoiseq.Interval(chrom="chr19", start=55_115_750, end=55_115_780)
with open(os.path.join(args.data_path, "PZDONOR EF1alpha Cherry ACTB GFP synthetic core promoter library (Ns).gbk")) as file:
    plasmid_sequence = ""
    started_reading = False
    for line in file.readlines():
        if started_reading:
            line = line.strip().replace(" ", "").replace("/", "")
            line = re.sub(r'[0-9]+', '', line)
            plasmid_sequence += line
        if line.startswith("ORIGIN"):
            started_reading = True
    plasmid_sequence = plasmid_sequence.upper()
full_insert = plasmid_sequence[658:6742]
egfp_seq = seq_utils.rev_comp_sequence(plasmid_sequence[4189:4909])

cols = ['Oligo_index', 'Background', 'Oligo_sequence']
dnase_example = segal_df.query('Oligo_index == 12885').copy()
dnase_example["Background"] = "DNAse_example"
background_seq_df = pd.concat(
    [elements_df.query('Configuration_summary == 0')[cols],
     tf_activity_df.query('Motif_info == "-"')[cols],
     dnase_example[cols]]
).reset_index(drop=True)
background_seq_df["pos"] = -1
background_seq_df["alt"] = -1
background_seq_df["idx"] = -1


def ism(background_df):
    options = ["A", "C", "G", "T"]
    rows = []
    for _,crs_row in background_df.iterrows():
        seq = crs_row["Oligo_sequence"]
        for pos in range(len(seq)):
            for i in range(3):
                ref = seq[pos]
                alt = options[(options.index(ref) + i) % 4]
                new_seq = ''.join([seq[:pos], alt, seq[pos+1:]])
                rows.append({"Oligo_index":crs_row["Oligo_index"],
                             "Oligo_sequence":new_seq,
                             "Background":crs_row["Background"],
                             "pos":pos,
                             "alt":alt,
                             "idx":options.index(alt)
                            })
    return pd.DataFrame(rows)


ism_df = pd.concat([ism(background_seq_df), background_seq_df]).reset_index(drop=True)

def segal_ism_sample_generator_factory(fasta_extractor):
    for _, crs_row in ism_df.iterrows():
        crs = crs_row["Oligo_sequence"]
        for insert_type in ["full", "full_rv", "minimal"]:
            if insert_type in ["full"]:
                insert = full_insert[:4658] + crs + full_insert[4822:]
                landmark = len(full_insert[:4658]) + len(crs)//2
            elif insert_type in ["full_rv"]:
                insert = seq_utils.rev_comp_sequence(full_insert[:4658] + crs + full_insert[4822:])
                landmark = len(full_insert[4822:]) + len(crs)//2
            else:
                insert = crs + egfp_seq
                landmark = len(crs)//2
            ideal_offset = seq_utils.compute_offset_to_center_landmark(landmark, insert)
            for offset in [-43, 0, 43]:
                modified_sequence, minbin, maxbin, landmarkbin = \
                    enformer_utils.insert_sequence_at_landing_pad(insert, aavs1,
                                                                  fasta_extractor,
                                                                  shift_five_end=ideal_offset + offset,
                                                                  landmark=landmark)
                yield modified_sequence, {"Oligo_index": crs_row["Oligo_index"],
                                          "offset": offset,
                                          "insert_type": insert_type,
                                          "minbin": minbin,
                                          "maxbin": maxbin,
                                          "landmarkbin": landmarkbin}


# prepare generator
sample_generator = \
    segal_ism_sample_generator_factory(fasta_extractor=fasta_extractor)
# we have 3 insert types and 3 offsets
num_of_samples = len(ism_df.index)*3*3

# write jobs
utils.batcherator(sample_generator, num_of_samples, args.number_of_jobs, args.output_dir)

# assert that we produced the right number of files
# should have number_of_jobs many pkls
assert args.number_of_jobs == len(glob.glob(os.path.join(args.output_dir,"*.pkl"))), \
       "Number of output files does not match the expected number."
