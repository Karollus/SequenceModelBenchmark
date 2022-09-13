import argparse
import glob
import os
import pandas as pd
import kipoiseq
import Code.Utilities.enformer_utils as enformer_utils
import Code.Utilities.basenji2_utils as basenji2_utils
import Code.Utilities.seq_utils as seq_utils
import Code.Utilities.utils as utils

parser = argparse.ArgumentParser()
parser.add_argument("--fasta-path")
parser.add_argument("--data-path")
parser.add_argument("--number-of-jobs", type=int)
parser.add_argument("--model", choices=["enformer", "basenji2"])
parser.add_argument("--output-dir")
args = parser.parse_args()

if args.model == "enformer":
    insert_sequence_at_landing_pad = enformer_utils.insert_sequence_at_landing_pad
elif args.model == "basenji2":
    insert_sequence_at_landing_pad = basenji2_utils.insert_sequence_at_landing_pad

df = pd.read_csv(os.path.join(args.data_path, "ziga_additional_columns.tsv"),
                 sep="\t")
df = df.drop_duplicates(
    subset=["chromosome", "enhancer_start", "enhancer_end"]
)
df = df[["chromosome", "enhancer_start", "enhancer_end"]]
df = df.reset_index()

fasta_extractor = enformer_utils.FastaStringExtractor(args.fasta_path)


def avsec_enhancercentered_sample_generator_factory(fasta_extractor):
    for _, row in df.iterrows():
        enh_len = (row["enhancer_end"] - row["enhancer_start"])
        enh_mid = row["enhancer_start"] + (enh_len // 2)
        interval = kipoiseq.Interval(chrom=row["chromosome"],
                                     start=enh_mid,
                                     end=enh_mid+1)
        nuc = fasta_extractor.extract(interval)
        for rev_comp in [False, True]:
            for offset in [-43, 0, 43]:
                modified_sequence, minbin, maxbin, landmarkbin = \
                    insert_sequence_at_landing_pad(
                        insert=nuc,
                        lp_interval=interval,
                        fasta_extractor=fasta_extractor,
                        mode="replace",
                        shift_five_end=offset,
                        landmark=0,
                        rev_comp=rev_comp
                    )
                metadata_dict = {"chr": row["chromosome"],
                                 # old_index links to first occurrence
                                 # of enhancer in original avsec DF
                                 "old_index": row["index"],
                                 "enh_start": row["enhancer_start"],
                                 "enh_end": row["enhancer_end"],
                                 "enh_mid": enh_mid,
                                 "orient": "rv" if rev_comp else "fw",
                                 "offset": offset,
                                 "landmarkbin": landmarkbin,
                                 "minbin": minbin,
                                 "maxbin": maxbin}
                # start sanity checks
                seqmid = len(modified_sequence) // 2
                if not rev_comp:
                    assert (m := modified_sequence[seqmid+offset]) == nuc, \
                        f"Mismatch at center - expected {nuc}, got {m}"
                else:
                    revnuc = seq_utils.rev_comp_sequence(nuc)
                    assert (m := modified_sequence[seqmid-offset-1]) == revnuc, \
                        f"Mismatch at center - expected {revnuc}, got {m}"
                # end sanity checks
                yield modified_sequence, metadata_dict


# prepare generator
sample_generator = \
    avsec_enhancercentered_sample_generator_factory(fasta_extractor=fasta_extractor)
# we have 3 offsets and 2 orientations
num_of_samples = len(df.index)*3*2

# write jobs
utils.batcherator(sample_generator, num_of_samples, args.number_of_jobs, args.output_dir)

# assert that we produced the right number of files
# should have number_of_jobs many pkls
assert args.number_of_jobs == len(glob.glob(os.path.join(args.output_dir,"*.pkl"))), \
       "Number of output files does not match the expected number."
