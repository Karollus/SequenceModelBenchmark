import pandas as pd
import os
import Code.Utilities.enformer_utils as enformer_utils
import Code.Utilities.utils as utils
import Code.Utilities.seq_utils as seq_utils
import kipoiseq
import numpy as np
import argparse
import glob
from copy import deepcopy

parser = argparse.ArgumentParser()
parser.add_argument("--fasta-path")
parser.add_argument("--data-path")  # "Data/Fulco_CRISPRi/"
parser.add_argument("--number-of-jobs", type=int)
parser.add_argument("--output-dir")
args = parser.parse_args()

df = pd.read_csv(os.path.join(args.data_path, "avsec_fulco_merge_full.tsv"),
                 sep="\t")


def join_orient_seq(five_end, enhancer, three_end, orient):
    seq = "".join([five_end, enhancer, three_end])
    if orient == "rv":
        seq = seq_utils.rev_comp_sequence(seq)
    return seq


def shuffle_string(string):
    string_as_list = list(string)
    np.random.shuffle(string_as_list)
    return "".join(string_as_list)


def assemble_seq(five_end, enhancer, three_end, orient, enh_mod):
    if enh_mod == "none":
        seq = join_orient_seq(five_end, enhancer, three_end, orient)
    elif enh_mod == "replace_with_n":
        mod_enh_seq = "N" * len(enhancer)
        seq = join_orient_seq(five_end, mod_enh_seq, three_end, orient)
    elif enh_mod == "shuffle":
        rnd_enhancer = shuffle_string(enhancer)
        seq = join_orient_seq(five_end, rnd_enhancer, three_end, orient)
    assert len(seq) == enformer_utils.SEQUENCE_LENGTH
    return seq


def fulco_crispri_sample_generator_factory(fasta_extractor):
    for _, row in df.iterrows():

        # first get the enhancer
        enhancer_interval = kipoiseq.Interval(
            chrom=row["chromosome"],
            start=row["enhancer_wide_start"],
            end=row["enhancer_wide_end"]
        )
        enhancer_seq = fasta_extractor.extract(enhancer_interval)
        # start metadata dict with all info relevant to this key
        metadata_dict = {
            "key": row["key"],
            "chromosome": row["chromosome"],
            "gene": row["Gene"],
        }

        for enh_mod in ["none", "replace_with_n", "shuffle"]:
            metadata_dict["enhancer_modification"] = enh_mod
            for offset in [-43, 0, +43]:
                metadata_dict["offset"] = offset
                for orient in ["fw", "rv"]:
                    metadata_dict["orient"] = orient

                    # extract 5' and 3' flanks to attach to enhancer
                    five_end_interval = kipoiseq.Interval(
                         chrom=row["chromosome"],
                         start=row["sequence_start"] + offset,
                         end=row["enhancer_wide_start"]
                     )
                    three_end_interval = kipoiseq.Interval(
                         chrom=row["chromosome"],
                         start=row["enhancer_wide_end"],
                         end=row["sequence_end"] + offset
                     )
                    five_end_seq = fasta_extractor.extract(five_end_interval)
                    three_end_seq = fasta_extractor.extract(three_end_interval)

                    # "package" seqs
                    seqs = (five_end_seq, enhancer_seq, three_end_seq)

                    # calculate and set the tss bins & enh coordinates accordingly
                    # as TSS is centered in bin, +/-43 shift doesn't change it
                    if orient == "fw":
                        tss_bin = row["main_tss_bin"]
                        enh_mid_in_seq = len(five_end_seq) + (len(enhancer_seq) // 2)
                    elif orient == "rv":
                        tss_bin = 896 - row["main_tss_bin"] - 1
                        enh_mid_in_seq = len(three_end_seq) + (len(enhancer_seq) // 2)
                    metadata_dict.update(
                        {"landmarkbin": tss_bin,
                         "minbin": tss_bin,
                         "maxbin": tss_bin,
                         "enhancer_mid_in_seq": enh_mid_in_seq})

                    # assemble seq depending on enhancer modification type
                    if enh_mod == "shuffle":
                        for i in range(100):
                            seq = assemble_seq(*seqs, orient, enh_mod)
                            metadata_dict["randomization"] = i
                            yield seq, deepcopy(metadata_dict)
                    else:
                        seq = assemble_seq(*seqs, orient, enh_mod)
                        metadata_dict["randomization"] = -1
                        yield seq, deepcopy(metadata_dict)


fasta_extractor = enformer_utils.FastaStringExtractor(args.fasta_path)
# prepare generator
sample_generator = fulco_crispri_sample_generator_factory(
    fasta_extractor=fasta_extractor)
# * 2 (rev_comp) * 3 (offset) * 102 (1 none, 1 replace_with_n, 100 shuffle)
num_of_samples = len(df.index) * 2 * 3 * 102

# write jobs
utils.batcherator(sample_generator, num_of_samples, args.number_of_jobs, args.output_dir)

# assert that we produced the right number of files
# should have number_of_jobs many pkls
assert args.number_of_jobs == len(glob.glob(os.path.join(args.output_dir,"*.pkl"))), \
       "Number of output files does not match the expected number."
