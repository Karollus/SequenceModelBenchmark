import glob
import kipoiseq
import argparse
import os
import pandas as pd
import itertools as it

import Code.Utilities.utils as utils
import Code.Utilities.enformer_utils as enformer_utils
import Code.Utilities.seq_utils as seq_utils

parser = argparse.ArgumentParser()
parser.add_argument("--fasta-path")
parser.add_argument("--data-path")  # "Data/Fulco_CRISPRi/"
parser.add_argument("--number-of-jobs", type=int)
parser.add_argument("--output-dir")
args = parser.parse_args()

iscl_df = pd.read_csv(
    os.path.join(args.data_path, "in_silico_candidate_locs.tsv"), sep="\t"
)

# check that the TSS end and TSS start always have a fixed size
assert ((iscl_df["main_tss_end"] - iscl_df["main_tss_start"]) == 128).all()
# calculate windows around TSS
iscl_df["main_tss_center"] = iscl_df["main_tss_start"] + 64
iscl_df["tss_window_start"] = iscl_df["main_tss_center"] - 500
iscl_df["tss_window_end"] = iscl_df["main_tss_center"] + 500
# extract unique TSSs
tss_df = iscl_df.drop_duplicates(
    subset=["main_tss_start", "main_tss_end"]
).copy()
tss_df["tss_id"] = tss_df.agg(
    lambda x: f"{x['chromosome']}:{x['main_tss_start']}-{x['main_tss_end']}",
    axis=1
)
tss_df = tss_df[["tss_id", "chromosome", "tss_window_start", "tss_window_end", "main_tss_start", "main_tss_end"]]
# extract unique enhancers
enh_df = iscl_df.drop_duplicates(
    subset=["fix_enhancer_wide_start", "fix_enhancer_wide_end"]
).copy()
enh_df["enh_id"] = enh_df.agg(
    lambda x: f"{x['chromosome']}:{x['fix_enhancer_wide_start']}-{x['fix_enhancer_wide_end']}",
    axis=1
)
enh_df = enh_df[["enh_id", "chromosome", "fix_enhancer_wide_start", "fix_enhancer_wide_end"]]
# extract *t*est *loc*ations
tloc_df = iscl_df[iscl_df["test_location"]]
# compute the "correct" id's at each test location
tloc_df["actual_tss_id"] = tloc_df.agg(
    lambda x: f"{x['chromosome']}:{x['main_tss_start']}-{x['main_tss_end']}",
    axis=1
)
tloc_df["actual_enh_id"] = tloc_df.agg(
    lambda x: f"{x['chromosome']}:{x['fix_enhancer_wide_start']}-{x['fix_enhancer_wide_end']}",
    axis=1
)


def tss_to_right_of_enh(tloc_row):
    if (tloc_row["fix_enhancer_wide_start"] < tloc_row["tss_window_start"] and
        tloc_row["fix_enhancer_wide_end"] < tloc_row["tss_window_start"]):
        return True  # tss is to the left of enhancer
    elif (tloc_row["tss_window_start"] < tloc_row["fix_enhancer_wide_start"] and
          tloc_row["tss_window_end"] < tloc_row["fix_enhancer_wide_start"]):
        return False  # tss is to the right of enhancer
    else:
        assert False, "Overlap should not be possible"


def make_seq(tloc_row, tss_row, enh_row, offset, orient, fasta_extractor):
    if tss_to_right_of_enh(tloc_row):
        # 5'---enh---tss---3'
        five_chunk_start = tloc_row["fix_enhancer_wide_start"]
        five_chunk_end = tloc_row["fix_enhancer_wide_end"]
        three_chunk_start = tloc_row["tss_window_start"]
        three_chunk_end = tloc_row["tss_window_end"]
        five_insert_chrom = enh_row["chromosome"]
        five_insert_start = enh_row["fix_enhancer_wide_start"]
        five_insert_end = enh_row["fix_enhancer_wide_end"]
        three_insert_chrom = tss_row["chromosome"]
        three_insert_start = tss_row["tss_window_start"]
        three_insert_end = tss_row["tss_window_end"]
    else:
        # 5'---tss---enh---3'
        five_chunk_start = tloc_row["tss_window_start"]
        five_chunk_end = tloc_row["tss_window_end"]
        three_chunk_start = tloc_row["fix_enhancer_wide_start"]
        three_chunk_end = tloc_row["fix_enhancer_wide_end"]
        five_insert_chrom = tss_row["chromosome"]
        five_insert_start = tss_row["tss_window_start"]
        five_insert_end = tss_row["tss_window_end"]
        three_insert_chrom = enh_row["chromosome"]
        three_insert_start = enh_row["fix_enhancer_wide_start"]
        three_insert_end = enh_row["fix_enhancer_wide_end"]
    # 5'---ins---ins---3'
    #   ^^^
    five_flank = kipoiseq.Interval(
        chrom=tloc_row["chromosome"],
        start=tloc_row["sequence_start"] + offset,
        end=five_chunk_start
    )
    # 5'---ins---ins---3'
    #      ^^^
    five_insert = kipoiseq.Interval(
        chrom=five_insert_chrom,
        start=five_insert_start,
        end=five_insert_end
    )
    # 5'---ins---ins---3'
    #         ^^^
    mid = kipoiseq.Interval(
        chrom=tloc_row["chromosome"],
        start=five_chunk_end,
        end=three_chunk_start
    )
    # 5'---ins---ins---3'
    #            ^^^
    three_insert = kipoiseq.Interval(
        chrom=three_insert_chrom,
        start=three_insert_start,
        end=three_insert_end
    )
    # 5'---ins---ins---3'
    #               ^^^
    three_flank = kipoiseq.Interval(
        chrom=tloc_row["chromosome"],
        start=three_chunk_end,
        end=tloc_row["sequence_end"] + offset
    )
    seq = [five_flank, five_insert, mid, three_insert, three_flank]
    seq = map(lambda intvl: fasta_extractor.extract(intvl), seq)
    seq = "".join(seq)
    if orient == "rv":
        seq = seq_utils.rev_comp_sequence(seq)
    assert len(seq) == enformer_utils.SEQUENCE_LENGTH
    return seq


def make_metadata_dict(tloc_row, tss_row, enh_row, offset, orient):
    if orient == "fw":
        tss_bin = tloc_row["main_tss_bin"]
    else:
        tss_bin = 896 - tloc_row["main_tss_bin"] - 1
    metadata = {
        "location_key": tloc_row["ziga_key"],
        "promoter_id": tss_row["tss_id"],
        "enhancer_id": enh_row["enh_id"],
        "landmarkbin": tss_bin,
        "minbin": tss_bin,
        "maxbin": tss_bin,
        "offset": offset,
        "orient": orient,
    }
    return metadata

def extract_refseq_to_test(tloc_row, offset, orient, fasta_extractor):
    ref_interval = kipoiseq.Interval(
                    chrom=tloc_row["chromosome"],
                    start=tloc_row["sequence_start"] + offset,
                    end=tloc_row["sequence_end"] + offset)
    ref_seq = fasta_extractor.extract(ref_interval)
    if orient == "rv":
        ref_seq = seq_utils.rev_comp_sequence(ref_seq)
    return ref_seq

def fulco_in_fulco_sample_generator_factory(fasta_extractor):
    itr = it.product(tloc_df.iterrows(), tss_df.iterrows(), enh_df.iterrows())
    for (_, tloc_row), (_, tss_row), (_, enh_row) in itr:
        for offset in [-43, 0, +43]:
            for orient in ["fw", "rv"]:
                seq = make_seq(tloc_row, tss_row, enh_row, offset, orient, fasta_extractor)
                # if we have an endogenous pair, test that we have the correct sequence
                if tloc_row["actual_tss_id"] == tss_row["tss_id"] and tloc_row["actual_enh_id"] == enh_row["enh_id"]:
                        assert extract_refseq_to_test(tloc_row, offset, orient, fasta_extractor) == seq, \
                        "Wrong seq for loc:{}, prom:{} and enh:{}. Is {}, should be {}".format(tloc_row["ziga_key"],
                                                                                               tss_row["tss_id"],
                                                                                               enh_row["enh_id"],
                                                                                              seq, ref_seq)
                metadata = make_metadata_dict(tloc_row, tss_row, enh_row, offset, orient)
                yield seq, metadata


fasta_extractor = enformer_utils.FastaStringExtractor(args.fasta_path)
# prepare generator
sample_generator = fulco_in_fulco_sample_generator_factory(fasta_extractor=fasta_extractor)
# tlocs * tss * enh * 2 (rev_comp) * 3 (offset)
num_of_samples = len(tloc_df.index) * len(tss_df.index) * len(enh_df.index) * 2 * 3

# write jobs
utils.batcherator(sample_generator, num_of_samples, args.number_of_jobs, args.output_dir)

# assert that we produced the right number of files
# should have number_of_jobs many pkls
assert args.number_of_jobs == len(glob.glob(os.path.join(args.output_dir,"*.pkl"))), \
       "Number of output files does not match the expected number."
