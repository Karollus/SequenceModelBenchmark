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
parser.add_argument("--data-path") # "Data/Bergman_compatibility_logic/"
parser.add_argument("--number-of-jobs", type=int) 
parser.add_argument("--output-dir")
args = parser.parse_args()


"""This script creates samples for the Bergmann enhancer x promoter study, but treating everything as a promoter"""

fasta_extractor = enformer_utils.FastaStringExtractor(args.fasta_path)

# prepare sequences
promoters = pd.read_csv(os.path.join(args.data_path,"promoters.txt"),sep="\t")[["fragment","seq"]]
enhancers = pd.read_csv(os.path.join(args.data_path,"enhancers.txt"),sep="\t")[["fragment","seq"]]
crs_df = pd.concat([promoters, enhancers])

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
    for _,crs_row in crs_df.iterrows():
        fragment = crs_row["seq"]
        for insert_type in ["full_plasmid", "min_plasmid", "aavs1"]:
            if insert_type == "full_plasmid":
                insert = plasmid = plasmid_left + five_prime_prom + fragment + three_prime_prom + tGFP + plasmid_right
                landmark = len(plasmid_left + five_prime_prom) + len(fragment)//2
            else:
                insert = plasmid_min = five_prime_prom + fragment + three_prime_prom + tGFP
                landmark = len(five_prime_prom) + len(fragment)//2
            ideal_offset = seq_utils.compute_offset_to_center_landmark(landmark, insert)
            for offset in [-64,-32,0]:
                if insert_type in ["full_plasmid", "min_plasmid"]:
                    modified_sequence, minbin, maxbin, landmarkbin = \
                        enformer_utils.pad_sequence(insert,
                                                  shift_five_end=ideal_offset + offset,
                                                  landmark=landmark)
                else:
                    modified_sequence, minbin, maxbin, landmarkbin = \
                        enformer_utils.insert_sequence_at_landing_pad(insert,aavs1,
                                                                      fasta_extractor,
                                                                      shift_five_end=ideal_offset + offset,
                                                                      landmark=landmark)
                yield modified_sequence, {"fragment":crs_row["fragment"],
                                          "offset":offset,
                                          "insert_type":insert_type,
                                          "minbin":minbin,
                                          "maxbin":maxbin,
                                          "landmarkbin":landmarkbin}

# prepare generator
sample_generator = \
    bergmann_sample_generator_factory(fasta_extractor=fasta_extractor)
# write jobs
num_of_samples = len(crs_df.index)*9

utils.batcherator(sample_generator, num_of_samples, args.number_of_jobs, args.output_dir)

# assert that we produced the right number of files
# should have number_of_jobs many pkls
assert args.number_of_jobs == len(glob.glob(os.path.join(args.output_dir,"*.pkl"))), \
       "Number of output files does not match the expected number."