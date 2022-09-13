import re
import glob
import os

import kipoiseq
import pandas as pd

import Code.Utilities.utils as utils
import Code.Utilities.seq_utils as seq_utils
import Code.Utilities.enformer_utils as enformer_utils
import Code.Utilities.basenji1_utils as basenji1_utils
import Code.Utilities.basenji2_utils as basenji2_utils

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--fasta-path")
parser.add_argument("--data-path") # "Data/Segal_promoter/"
parser.add_argument("--number-of-jobs", type=int) 
parser.add_argument("--model", choices=["enformer", "basenji1", "basenji2"])
parser.add_argument("--output-dir")
args = parser.parse_args()

"""This script creates samples for the Segal human promoter study"""

if args.model == "enformer":
    insert_sequence_at_landing_pad = enformer_utils.insert_sequence_at_landing_pad
elif args.model == "basenji1":
    insert_sequence_at_landing_pad = basenji1_utils.insert_sequence_at_landing_pad
elif args.model == "basenji2":
    insert_sequence_at_landing_pad = basenji2_utils.insert_sequence_at_landing_pad

fasta_extractor = enformer_utils.FastaStringExtractor(args.fasta_path)

# collect the data
segal_df = pd.read_csv(
    os.path.join(args.data_path, "GSM3323461_oligos_measurements_processed_data.tab"),
    sep="\t")

#collect sequences
# AAVS1: (PPP1R12C-201, intron 1)
aavs1 = kipoiseq.Interval(chrom="chr19",start=55_115_750,end=55_115_780)
with open(os.path.join(args.data_path, "PZDONOR EF1alpha Cherry ACTB GFP synthetic core promoter library (Ns).gbk")) as file:
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
full_insert = plasmid_sequence[658:6742]
egfp_seq = seq_utils.rev_comp_sequence(plasmid_sequence[4189:4909])


def segal_sample_generator_factory(fasta_extractor):
    for crs_row in segal_df.iterrows():
        crs_row = crs_row[1]
        crs = crs_row["Oligo_sequence"]
        for insert_type in ["full", "full_rv", "minimal"]:
            if insert_type in ["full"]:
                insert = full_insert[:4658] + crs + full_insert[4822:]
                landmark = len(full_insert[:4658]) + len(crs)//2
            elif insert_type in ["full_rv"]:
                insert = seq_utils.rev_comp_sequence(full_insert[:4658] + crs + full_insert[4822:])
                landmark = len(full_insert[4822:]) + len(crs)//2
            elif insert_type in ["minimal"]:
                insert = crs + egfp_seq
                landmark = len(crs)//2
            ideal_offset = seq_utils.compute_offset_to_center_landmark(landmark, insert)
            for rev_comp in [False, True]:
                for offset in [-43, 0, 43]:
                    modified_sequence, minbin, maxbin, landmarkbin = \
                        insert_sequence_at_landing_pad(insert,
                                                       aavs1,
                                                       fasta_extractor,
                                                       shift_five_end=ideal_offset + offset,
                                                       landmark=landmark,
                                                       rev_comp=rev_comp)
                    yield modified_sequence, {"Oligo_index": crs_row["Oligo_index"],
                                              "offset": offset,
                                              "orient": "rv" if rev_comp else "fw",
                                              "insert_type": insert_type,
                                              "minbin": minbin,
                                              "maxbin": maxbin,
                                              "landmarkbin": landmarkbin}

# prepare generator
sample_generator = \
    segal_sample_generator_factory(fasta_extractor=fasta_extractor)
# we have 3 insert types, 3 offsets and 2 orientations
num_of_samples = len(segal_df.index)*3*3*2

# write jobs
utils.batcherator(sample_generator, num_of_samples, args.number_of_jobs, args.output_dir)

# assert that we produced the right number of files
# should have number_of_jobs many pkls
assert args.number_of_jobs == len(glob.glob(os.path.join(args.output_dir,"*.pkl"))), \
       "Number of output files does not match the expected number."
