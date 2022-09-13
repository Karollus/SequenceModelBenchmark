import glob
import kipoiseq
import argparse
import os
import pandas as pd
import math

import Code.Utilities.utils as utils
import Code.Utilities.enformer_utils as enformer_utils
import Code.Utilities.basenji1_utils as basenji1_utils
import Code.Utilities.basenji2_utils as basenji2_utils

parser = argparse.ArgumentParser()
parser.add_argument("--fasta-path")
parser.add_argument("--data-path")  # "Data/TSS_sim/"
parser.add_argument("--number-of-jobs", type=int)
parser.add_argument("--model", choices=["enformer", "basenji1", "basenji2"])
parser.add_argument("--output-dir")
args = parser.parse_args()

if args.model == "enformer":
    insert_sequence_at_landing_pad = enformer_utils.insert_sequence_at_landing_pad
    pad_sequence = enformer_utils.pad_sequence
elif args.model == "basenji1":
    insert_sequence_at_landing_pad = basenji1_utils.insert_sequence_at_landing_pad
    pad_sequence = basenji1_utils.pad_sequence
elif args.model == "basenji2":
    insert_sequence_at_landing_pad = basenji2_utils.insert_sequence_at_landing_pad
    pad_sequence = basenji2_utils.pad_sequence

name_list = [
    "ts_id", "ts_ver", "ts_type", "chr", "strand", "ts_start", "ts_end", "tss",
    "ensembl_canonical", "mane", "tsl", "gene_id", "gene_name"
]
# conversion functions for some tsv columns
conv_funs = {
    "strand": lambda x: "+" if x == "1" else "-",
    "ensembl_canonical": lambda x: True if x == "1" else False,
    "mane": lambda x: x if not x == "" else pd.NA,
    "tsl": lambda x: x.split(" ")[0] if not x.split(" ")[0] == "" else pd.NA
}
# strand, ensembl_canonical, mane and tsl are handled by their converters
dtype_dict = {
    "ts_id": str,
    "ts_ver": int,
    "ts_type": str,
    "chr": str,
    "ts_start": int,
    "ts_end": int,
    "tss": int,
    "gene_id": str,
    "gene_name": str
}
# create the dataframe
tss_df = pd.read_csv(os.path.join(args.data_path,
                                  "bigly_tss_from_biomart.txt"),
                     sep="\t",
                     dtype=dtype_dict,
                     names=name_list,
                     converters=conv_funs,
                     header=0)
# get protein_coding ensembl_canonical transcripts on the standard chromosomes
tss_df = tss_df[(tss_df["ts_type"] == "protein_coding")
                & (tss_df["ensembl_canonical"])
                & (tss_df["chr"].isin(
                    list(map(lambda x: str(x), range(1, 23))) +
                    ["MT", "X", "Y"]))]

# needed to translate chromosome symbols from biomart output to fasta
chromosome_dict = {
    "1": "chr1",
    "2": "chr2",
    "3": "chr3",
    "4": "chr4",
    "5": "chr5",
    "6": "chr6",
    "7": "chr7",
    "8": "chr8",
    "9": "chr9",
    "10": "chr10",
    "11": "chr11",
    "12": "chr12",
    "13": "chr13",
    "14": "chr14",
    "15": "chr15",
    "16": "chr16",
    "17": "chr17",
    "18": "chr18",
    "19": "chr19",
    "20": "chr20",
    "21": "chr21",
    "22": "chr22",
    "MT": "chrM",
    "X": "chrX",
    "Y": "chrY"
}


def ingenome_sample_from_row(row, fasta_extractor, offset, rev_comp,
                             window_type):
    interval = kipoiseq.Interval(chrom=chromosome_dict[row["chr"]],
                                 start=int(row["tss"]) - 1,
                                 end=int(row["tss"]))
    nuc = fasta_extractor.extract(interval)
    modified_sequence, minbin, maxbin, landmarkbin = \
        insert_sequence_at_landing_pad(nuc,
                                       interval,
                                       fasta_extractor,
                                       mode="replace",
                                       landmark=0,
                                       shift_five_end=offset,
                                       rev_comp=rev_comp)
    metadata_dict = {"chromosome": chromosome_dict[row["chr"]],
                     "landmark": row["tss"],
                     "transcript_id": row["ts_id"],
                     "strand": row["strand"],
                     "gene_id": row["gene_id"],
                     "gene_name": row["gene_name"],
                     "window_type": window_type,
                     "window_size": -1,
                     "offset": offset,
                     "orient": "rv" if rev_comp else "fw",
                     "minbin": minbin,
                     "maxbin": maxbin,
                     "landmarkbin": landmarkbin}
    return modified_sequence, metadata_dict


def windowed_sample_from_row(row,
                             fasta_extractor,
                             offset,
                             rev_comp,
                             window_type,
                             window_size,
                             percent_window_shift=0):
    """Returns seq and metadata with the TSS shifted away from the seq center.

    percent_window_shift: percent of window size the center of the window
    should be shifted away from the TSS. If positive, shift in 3' direction, if
    negative, shift in 5' direction. If set to 0 (default), the TSS will remain
    in the center. The function is strand-aware, so it is not necessary to
    manually adapt this value for the negative strand.
    """
    window_5_end = math.ceil(window_size / 2)
    window_3_end = math.floor(window_size / 2)
    absolute_window_shift = math.floor(percent_window_shift * window_size)
    if row["strand"] == "+":
        center = row["tss"] + absolute_window_shift
        landmark = window_5_end - 1 - absolute_window_shift
    elif row["strand"] == "-":
        center = row["tss"] - absolute_window_shift
        landmark = window_5_end - 1 + absolute_window_shift
    interval = kipoiseq.Interval(chrom=chromosome_dict[row["chr"]],
                                 start=center-window_5_end,
                                 end=center+window_3_end)
    insert = fasta_extractor.extract(interval)
    modified_sequence, minbin, maxbin, landmarkbin = \
        pad_sequence(insert=insert,
                     shift_five_end=offset,
                     landmark=landmark,
                     rev_comp=rev_comp)
    metadata_dict = {"chromosome": chromosome_dict[row["chr"]],
                     "landmark": center,
                     "transcript_id": row["ts_id"],
                     "strand": row["strand"],
                     "gene_id": row["gene_id"],
                     "gene_name": row["gene_name"],
                     "window_type": window_type,
                     "window_size": window_size,
                     "offset": offset,
                     "orient": "rv" if rev_comp else "fw",
                     "minbin": minbin,
                     "maxbin": maxbin,
                     "landmarkbin": landmarkbin}
    return modified_sequence, metadata_dict


def enformer_tss_sim_sample_generator_factory(fasta_extractor):
    for _, row in tss_df.iterrows():
        for rev_comp in [False, True]:
            for offset in [-43, 0, 43]:
                for window_type in ["ingenome", "window_center_shift"]:
                    if window_type == "ingenome":
                        yield ingenome_sample_from_row(row, fasta_extractor,
                                                       offset, rev_comp,
                                                       window_type)
                    elif window_type == "window_center_shift":
                        for window_size in [1_001, 3_001, 12_501, 34_501,
                                           enformer_utils.SEEN_SEQUENCE_LENGTH//5,
                                           enformer_utils.SEEN_SEQUENCE_LENGTH//4 + 1,
                                           enformer_utils.SEEN_SEQUENCE_LENGTH//3 + 1,
                                           enformer_utils.SEEN_SEQUENCE_LENGTH//2 + 1,
                                            131_073]:
                            yield windowed_sample_from_row(
                                row, fasta_extractor, offset, rev_comp,
                                window_type, window_size, 0)

def basenji2_tss_sim_sample_generator_factory(fasta_extractor):
    for _, row in tss_df.iterrows():
        for rev_comp in [False, True]:
            for offset in [-43, 0, 43]:
                for window_type in ["ingenome", "window_center_shift"]:
                    if window_type == "ingenome":
                        yield ingenome_sample_from_row(row, fasta_extractor,
                                                       offset, rev_comp,
                                                       window_type)
                    elif window_type == "window_center_shift":
                        for window_size in [1_001, 3_001, 12_501, 34_501,
                                           enformer_utils.SEEN_SEQUENCE_LENGTH//5,
                                           enformer_utils.SEEN_SEQUENCE_LENGTH//4 + 1,
                                           enformer_utils.SEEN_SEQUENCE_LENGTH//3 + 1,
                                           enformer_utils.SEEN_SEQUENCE_LENGTH//2 + 1]:
                            yield windowed_sample_from_row(
                                row, fasta_extractor, offset, rev_comp,
                                window_type, window_size, 0)


def basenji1_tss_sim_sample_generator_factory(fasta_extractor):
    for _, row in tss_df.iterrows():
        for rev_comp in [False, True]:
            for offset in [-43, 0, 43]:
                yield ingenome_sample_from_row(row, fasta_extractor,
                                               offset, rev_comp,
                                               "ingenome")


fasta_extractor = enformer_utils.FastaStringExtractor(args.fasta_path)

if args.model == "enformer":
    # prepare generator
    sample_generator = enformer_tss_sim_sample_generator_factory(
        fasta_extractor=fasta_extractor)
    # * 2 (rev_comp) * 3 (offset) * 10 (ingenome + 9 window sizes)
    num_of_samples = len(tss_df.index) * 2 * 3 * 10
elif args.model == "basenji2":
    sample_generator = basenji2_tss_sim_sample_generator_factory(
        fasta_extractor=fasta_extractor)
    # * 2 (rev_comp) * 3 (offset) * 9 (ingenome + 8 window sizes)
    num_of_samples = len(tss_df.index) * 2 * 3 * 10
elif args.model == "basenji1":
    sample_generator = basenji1_tss_sim_sample_generator_factory(
        fasta_extractor=fasta_extractor)
    # * 2 (rev_comp) * 3 (offset)
    num_of_samples = len(tss_df.index) * 2 * 3

# write jobs
utils.batcherator(sample_generator, num_of_samples, args.number_of_jobs, args.output_dir)

# assert that we produced the right number of files
# should have number_of_jobs many pkls
assert args.number_of_jobs == len(glob.glob(os.path.join(args.output_dir,"*.pkl"))), \
       "Number of output files does not match the expected number."
