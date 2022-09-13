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
import Code.Utilities.basenji2_utils as basenji2_utils

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--fasta-path")
parser.add_argument("--data-path") # "Data/Bergman_compatibility_logic/"
parser.add_argument("--number-of-jobs", type=int) 
parser.add_argument("--model", choices=["enformer", "basenji2"])
parser.add_argument("--output-dir")
args = parser.parse_args()

if args.model == "enformer":
    insert_sequence_at_landing_pad = enformer_utils.insert_sequence_at_landing_pad
    pad_sequence = enformer_utils.pad_sequence
elif args.model == "basenji2":
    insert_sequence_at_landing_pad = basenji2_utils.insert_sequence_at_landing_pad
    pad_sequence = basenji2_utils.pad_sequence

"""This script creates samples for the Bergmann enhancer x promoter study"""

fasta_extractor = enformer_utils.FastaStringExtractor(args.fasta_path)

# prepare sequences
exp_df = pd.read_csv(os.path.join(args.data_path,"GSE184426_ExP_1Kx1K_counts.txt"),sep="\t")
promoters = pd.read_csv(os.path.join(args.data_path,"promoters.txt"),sep="\t").rename(columns={"fragment":"promoter","seq":"promoter_seq"})
enhancers = pd.read_csv(os.path.join(args.data_path,"enhancers.txt"),sep="\t").rename(columns={"fragment":"enhancer","seq":"enhancer_seq"})
# sum over replicates
exp_df["RNA_sum"] = (exp_df["weighted_reads_rep1"] + exp_df["weighted_reads_rep2"] + exp_df["weighted_reads_rep3"] + exp_df["weighted_reads_rep4"])
# subset DNA > 25 and RNA > 1
exp_df =  exp_df.query('DNA_input > 25 & RNA_sum > 0')
# subset if < 2 barcodes
bc_count = (exp_df[["promoter","enhancer"]]
                  .groupby(["promoter","enhancer"])
                  .size()
                  .reset_index()
                  .rename(columns={0:"bc_count"})
                 ).query('bc_count > 1')
exp_df = exp_df.merge(bc_count, on=["promoter","enhancer"])
# aggregate over barcodes
exp_df = (exp_df[["promoter","enhancer","DNA_input","RNA_sum"]]
          .groupby(["promoter","enhancer"])
          .sum()
          .reset_index()
         )
# merge with sequences
exp_df = exp_df.merge(promoters[["promoter","promoter_seq"]], on="promoter")
exp_df = exp_df.merge(enhancers[["enhancer","enhancer_seq"]], on="enhancer")


# get the plasmid sequence
with open(os.path.join(args.data_path,"hstarr_seq.gbk")) as file:
    plasmid_sequence = ""
    started_reading = False
    for line in file.readlines():
        if started_reading:
            line = line.strip().replace(" ","").replace("/","")
            line = re.sub(r'[0-9]+', '', line)
            plasmid_sequence += line
        if line.startswith("ORIGIN"):
            started_reading = True
    plasmid_sequence = plasmid_sequence.upper()

plasmid_left = plasmid_sequence[17:]
tGFP = plasmid_sequence[119:372]
plasmid_right = plasmid_sequence[1838:]
five_prime_prom = "TCATGTGGGACATCAAGC"
three_prime_prom = "GCATAGTGAGTCCACCTT"
five_prime_enhancer = "GCTAACTTCTACCCATGC"
three_prime_enhancer = "GCAAGTTAAGTAGGTCGT"
bc_left = "TAGATTGATCTAGAGCATGCA" 
bc_right = "GAGTACTGGTATGTTCA"
bc = "N"*16 #dummy barcode

aavs1 = kipoiseq.Interval(chrom="chr19",start=55_115_750,end=55_115_780)

def bergmann_sample_generator_factory(fasta_extractor):
    for _,crs_row in exp_df.iterrows():
        promoter = crs_row["promoter_seq"]
        enhancer = crs_row["enhancer_seq"]
        for insert_type in ["full_plasmid", "min_plasmid", "aavs1"]:
            if insert_type == "full_plasmid":
                insert = plasmid = plasmid_left + five_prime_prom + promoter + three_prime_prom + tGFP + bc_left + bc + bc_right + five_prime_enhancer + enhancer + three_prime_enhancer + plasmid_right
                landmark = len(plasmid_left + five_prime_prom) + len(promoter)//2
            else:
                insert = plasmid_min = five_prime_prom + promoter + three_prime_prom + tGFP + bc_left + bc + bc_right + five_prime_enhancer + enhancer + three_prime_enhancer 
                landmark = len(five_prime_prom) + len(promoter)//2
            ideal_offset = seq_utils.compute_offset_to_center_landmark(landmark, insert)
            for offset in [-64,-32,0]:
                if insert_type in ["full_plasmid", "min_plasmid"]:
                    modified_sequence, minbin, maxbin, landmarkbin = \
                        pad_sequence(insert,
                                     shift_five_end=ideal_offset + offset,
                                     landmark=landmark)
                else:
                    modified_sequence, minbin, maxbin, landmarkbin = \
                        insert_sequence_at_landing_pad(insert,aavs1,
                                                       fasta_extractor,
                                                       shift_five_end=ideal_offset + offset,
                                                       landmark=landmark)
                yield modified_sequence, {"promoter":crs_row["promoter"],
                                          "enhancer":crs_row["enhancer"],
                                          "offset":offset,
                                          "insert_type":insert_type,
                                          "minbin":minbin,
                                          "maxbin":maxbin,
                                          "landmarkbin":landmarkbin}

# prepare generator
sample_generator = \
    bergmann_sample_generator_factory(fasta_extractor=fasta_extractor)
# write jobs
num_of_samples = len(exp_df.index)*9

utils.batcherator(sample_generator, num_of_samples, args.number_of_jobs, args.output_dir)

# assert that we produced the right number of files
# should have number_of_jobs many pkls
assert args.number_of_jobs == len(glob.glob(os.path.join(args.output_dir,"*.pkl"))), \
       "Number of output files does not match the expected number."
