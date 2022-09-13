import pandas as pd
import os
import Code.Utilities.enformer_utils as enformer_utils
import Code.Utilities.utils as utils
import Code.Utilities.seq_utils as seq_utils
import kipoiseq
import math
import argparse
import glob

parser = argparse.ArgumentParser()
parser.add_argument("--fasta-path")
parser.add_argument("--data-path")  # "Data/Fulco_CRISPRi/"
parser.add_argument("--number-of-jobs", type=int)
parser.add_argument("--output-dir")
args = parser.parse_args()

df = pd.read_csv(
    os.path.join(args.data_path, "bergmann_in_fulco_candidate_locs_enhancers_merge.tsv"),
    sep="\t"
)

def calculate_enhancer_interval(row):
    """Calculates genomic start and end of the new enhancer"""
    # calculate mid of (old) fulco enhancer
    enh_len = abs(row["enhancer_end"] - row["enhancer_start"])
    assert enh_len == row["enhancer_size"]
    enh_mid = row["enhancer_start"] + (enh_len // 2)
    # now we need to calculate new start/end for the bergmann enhancer
    bm_enh_total_len = len(row["seq"])
    assert bm_enh_total_len == 264
    # split enhancer length into a 5' and a 3' part
    bm_enh_5_end_len = math.ceil(bm_enh_total_len / 2)
    bm_enh_3_end_len = math.floor(bm_enh_total_len / 2)
    start = enh_mid - bm_enh_5_end_len
    end = enh_mid + bm_enh_3_end_len
    return start, end

def join_orient_seq(five_end, enhancer, three_end, orient):
    seq = "".join([five_end, enhancer, three_end])
    if orient == "rv":
        seq = seq_utils.rev_comp_sequence(seq)
    return seq

def bergmann_in_fulco_generator_factory(fasta_extractor):
    for _, row in df.iterrows():
        bm_enh_seq = row["seq"]  # always 264 bp long
        bm_enh_start, bm_enh_end = calculate_enhancer_interval(row)
        for offset in [-43, 0, +43]:
            for orient in ["fw", "rv"]:
                five_end_interval = kipoiseq.Interval(
                    chrom=row["chromosome"],
                    start=row["sequence_start"] + offset,
                    end=bm_enh_start
                )
                three_end_interval = kipoiseq.Interval(
                    chrom=row["chromosome"],
                    start=bm_enh_end,
                    end=row["sequence_end"] + offset
                )
                five_end_seq = fasta_extractor.extract(five_end_interval)
                three_end_seq = fasta_extractor.extract(three_end_interval)
                # "package" seqs
                seqs = (five_end_seq, bm_enh_seq, three_end_seq)
                if orient == "fw":
                    tss_bin = row["main_tss_bin"]
                elif orient == "rv":
                    tss_bin = 896 - row["main_tss_bin"] - 1
                seq = join_orient_seq(*seqs, orient)
                assert len(seq) == enformer_utils.SEQUENCE_LENGTH
                metadata = {
                    "candidate_id": row["ziga_key"],
                    "bergmann_id": row["fragment"],
                    "candidate_chr": row["chromosome"],
                    "gene": row["gene"],
                    "offset": offset,
                    "orient": orient,
                    "landmarkbin": tss_bin,
                    "minbin": tss_bin,
                    "maxbin": tss_bin,
                }
                yield seq, metadata


fasta_extractor = enformer_utils.FastaStringExtractor(args.fasta_path)
# prepare generator
sample_generator = bergmann_in_fulco_generator_factory(
    fasta_extractor=fasta_extractor)
# * 2 (rev_comp) * 3 (offset)
num_of_samples = len(df.index) * 2 * 3

# write jobs
utils.batcherator(sample_generator, num_of_samples, args.number_of_jobs, args.output_dir)

# assert that we produced the right number of files
# should have number_of_jobs many pkls
assert args.number_of_jobs == len(glob.glob(os.path.join(args.output_dir,"*.pkl"))), \
       "Number of output files does not match the expected number."
