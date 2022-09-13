import Code.Utilities.enformer_utils as enformer_utils
import Code.Utilities.seq_utils as seq_utils
import Code.Utilities.utils as utils
import glob
import itertools as it
import pandas as pd
import numpy as np
import os
import math
import kipoiseq
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--fasta-path")
parser.add_argument("--data-path")  # "Data/TSS_sim/"
parser.add_argument("--number-of-jobs", type=int)
parser.add_argument("--output-dir")
parser.add_argument("--shuffles", type=int)
args = parser.parse_args()

cre_df = pd.read_csv(
    os.path.join(args.data_path, "cre_positions.tsv"), 
    sep="\t", index_col=0, dtype={"chr": str}
)
can_df = pd.read_csv(
    os.path.join(args.data_path, "only_protein_coding_ensembl_canonical_tss.tsv"), 
    sep="\t"
)
can_df["chr"] = "chr" + can_df["chr"]
can_df = can_df[~(can_df["chr"] == "chrY")]

def cre_to_local(c_start, c_end, r_start, r_end):
    """Translates CRE coordinates in a region to coordinates where region starts at 0"""
    assert c_start >= r_start or c_end <= r_end, \
        f"CRE needs to be within region! CRE: ({c_start}, {c_end}) - region: ({r_start}, {r_end})"
    c_start_local = max(0, c_start - r_start)
    c_end_local = min(c_end - r_start, enformer_utils.SEEN_SEQUENCE_LENGTH)
    return (c_start_local, c_end_local)

def transform_cre_coord_to_local(crelist, r_start, r_end):
    """Translates a list of CRE coord tuples into coordinates within r_egion starting at 0"""
    return [cre_to_local(*cre, r_start, r_end) for cre in crelist]

def dist_cre_to_tss(cre_start, cre_end, tss):
    """Get distance from cre to tss"""
    if cre_start < tss and cre_end < tss:
        distance = tss - cre_start
        return distance
    elif cre_start > tss and cre_end > tss:
        distance = cre_end - tss
        return distance
    else:
        assert False, "CRE should not intersect TSS!"

def filter_cre_to_outside_of_window(crelist, tss, window):
    """Filter CREs to exclude everything within window from TSS"""
    return [cre for cre in crelist if dist_cre_to_tss(*cre, tss) > window]

def n_cres_in_seq(crelist, seq):
    """Replace all given CREs in seq with N's"""
    idx_to_n = it.chain.from_iterable([range(start, end) for (start, end) in crelist])
    idx_to_n = np.fromiter(idx_to_n, dtype="int")
    npseq = np.array(list(seq))
    npseq[idx_to_n] = 'N'
    return "".join(npseq)

def shuf_cres_in_seq(crelist, seq):
    """Shuffles all given CREs in seq"""
    npseq = np.array(list(seq))
    for (start, end) in crelist:
        np.random.shuffle(npseq[start:end])
    return "".join(npseq)

def ingenome_predict(row, fasta_extractor):
    """Generator for ingenome predictions. Does not change CREs."""
    tss_interval = kipoiseq.Interval(
        chrom=row["chr"],
        start=row["tss"],
        end=row["tss"]+1
    )
    for offset in [-43, 0, +43]:
        for orient in ["fw", "rv"]:
            seq, minbin, maxbin, landmarkbin = \
                enformer_utils.extract_refseq_centred_at_landmark(
                    landmark_interval=tss_interval,
                    fasta_extractor=fasta_extractor,
                    shift_five_end=offset,
                    rev_comp=True if orient == "rv" else False
                )
            metadata = {
                "gene_id": row["gene_id"],
                "offset": offset,
                "orient": orient,
                "landmarkbin": landmarkbin,
                "minbin": minbin,
                "maxbin": maxbin,
                "mode": "ingenome",
                "window_size": -1,
                "shuffle_num": -1,
            }
            assert len(seq) == enformer_utils.SEQUENCE_LENGTH, \
                f"{metadata}"
            yield seq, metadata

def window_predict(subdf, fasta_extractor):
    """Generator for CRE-window predictions. N's and shuffles CREs."""
    assert (subdf[["gene_id", "Chromosome", "Start", "End", "tss"]].nunique() == 1).all(), \
        "All CREs should be the same wrt. chromosome, start, end and tss"
    # extract info relevant for all sequence segments
    gene_id = subdf["gene_id"].unique()[0]
    chromosome = subdf["Chromosome"].unique()[0]
    start = subdf["Start"].unique()[0]
    end = subdf["End"].unique()[0]
    tss = subdf["tss"].unique()[0]
    # transform CREs into a list of (start, end) coordinate tuples
    crelist = list(subdf[["Start_cre", "End_cre"]].to_records(index=False))
    for window in [1_000, 10_000, 20_000, 50_000]:
        # filter CRE list - remove CRE's *within* window and ...
        crelist_tmp = filter_cre_to_outside_of_window(crelist, tss, window)
        # ... transform them into "local" coordinates for the seen sequence segment
        crelist_tmp = transform_cre_coord_to_local(crelist_tmp, start, end)
        for offset in [-43, 0, +43]:
            five_flank_interval = kipoiseq.Interval(
                chrom=chromosome,
                start=(start - enformer_utils.PADDING_UNTIL_SEEN + offset),
                end=start
            )
            three_flank_interval = kipoiseq.Interval(
                chrom=chromosome,
                start=end,
                end=(end + enformer_utils.PADDING_UNTIL_SEEN + offset)
            )
            seen_seq_interval = kipoiseq.Interval(
                chrom=chromosome,
                start=start,
                end=end
            )
            five_flank = fasta_extractor.extract(five_flank_interval)
            three_flank = fasta_extractor.extract(three_flank_interval)
            for mode in ["N", "shuffle"]:
                for orient in ["fw", "rv"]:
                    seen_seq = fasta_extractor.extract(seen_seq_interval)
                    if mode == "N":
                        seen_seq = n_cres_in_seq(crelist_tmp, seen_seq)
                        seq = "".join([five_flank, seen_seq, three_flank])
                        if orient == "rv": 
                            seq = seq_utils.rev_comp_sequence(seq)
                            landmarkbin = \
                                (len(three_flank)+math.floor(len(seen_seq)/2)-enformer_utils.PADDING)//128
                            minbin = landmarkbin
                            maxbin = \
                                (len(three_flank)+math.floor(len(seen_seq)/2)+1-enformer_utils.PADDING)//128

                        else:
                            landmarkbin = \
                                (len(five_flank)+math.ceil(len(seen_seq)/2)-enformer_utils.PADDING)//128
                            minbin = landmarkbin
                            maxbin = \
                                (len(five_flank)+math.floor(len(seen_seq)/2)+1-enformer_utils.PADDING)//128
                        metadata = {
                            "gene_id": gene_id,
                            "offset": offset,
                            "orient": orient,
                            "landmarkbin": landmarkbin,
                            "minbin": minbin,
                            "maxbin": maxbin,
                            "mode": mode,
                            "window_size": window,
                            "shuffle_num": -1,
                        }
                        assert len(seq) == enformer_utils.SEQUENCE_LENGTH, \
                            f"{metadata}"
                        yield seq, metadata
                    elif mode == "shuffle":
                        for shuf_num in range(0, args.shuffles):
                            seen_seq = shuf_cres_in_seq(crelist_tmp, seen_seq)
                            seq = "".join([five_flank, seen_seq, three_flank])
                            if orient == "rv": 
                                seq = seq_utils.rev_comp_sequence(seq)
                                landmarkbin = \
                                    (len(three_flank)+math.floor(len(seen_seq)/2)-enformer_utils.PADDING)//128
                                minbin = landmarkbin
                                maxbin = \
                                    (len(three_flank)+math.floor(len(seen_seq)/2)+1-enformer_utils.PADDING)//128

                            else:
                                landmarkbin = \
                                    (len(five_flank)+math.ceil(len(seen_seq)/2)-enformer_utils.PADDING)//128
                                minbin = landmarkbin
                                maxbin = \
                                    (len(five_flank)+math.floor(len(seen_seq)/2)+1-enformer_utils.PADDING)//128
                            metadata = {
                                "gene_id": gene_id,
                                "offset": offset,
                                "orient": orient,
                                "landmarkbin": landmarkbin,
                                "minbin": minbin,
                                "maxbin": maxbin,
                                "mode": mode,
                                "window_size": window,
                                "shuffle_num": shuf_num,
                            }
                            assert len(seq) == enformer_utils.SEQUENCE_LENGTH, \
                                f"{metadata}"
                            yield seq, metadata
                    else: assert False, "Unknown mode"

def windowed_cre_generator_factory(fasta_extractor):
    for _, row in can_df.iterrows():
        # subset to CREs belonging to canonical TSS/gene
        gene_id = row["gene_id"]
        sub_cre_df = cre_df.query("gene_id == @gene_id")
        if sub_cre_df.shape[0] == 0:  # case: we don't have any CREs for gene
            yield from ingenome_predict(row, fasta_extractor)
        else:  # case: there *are* CREs for gene!
            yield from ingenome_predict(row, fasta_extractor)
            yield from window_predict(sub_cre_df, fasta_extractor)

fasta_extractor = enformer_utils.FastaStringExtractor(args.fasta_path)
# prepare generator
sample_generator = windowed_cre_generator_factory(fasta_extractor=fasta_extractor)

def num_of_samples():
    num = 0
    for _, row in can_df.iterrows():
        gene_id = row["gene_id"]
        num_cres = cre_df.query("gene_id == @gene_id").shape[0]
        if num_cres == 0:  # case: we don't have any CREs for gene
            num += 3 * 2
        else:  # case: there *are* CREs for gene!
            num += 3 * 2
            num += num_cres * 3 * 2 * (1 + args.shuffles)
    return num

num_of_samples = num_of_samples()

# write jobs
utils.batcherator(sample_generator, num_of_samples, args.number_of_jobs, args.output_dir)

# assert that we produced the right number of files
# should have number_of_jobs many pkls
assert args.number_of_jobs == len(glob.glob(os.path.join(args.output_dir,"*.pkl"))), \
       "Number of output files does not match the expected number."
