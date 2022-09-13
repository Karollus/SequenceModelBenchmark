import argparse
import glob
import os
import pandas as pd
import kipoiseq
import Code.Utilities.enformer_utils as enformer_utils
import Code.Utilities.utils as utils

parser = argparse.ArgumentParser()
parser.add_argument("--fasta-path")
parser.add_argument("--data-path")
parser.add_argument("--number-of-jobs", type=int)
parser.add_argument("--output-dir")
args = parser.parse_args()

df = pd.read_csv(os.path.join(args.data_path, "ziga_additional_columns.tsv"),
                 sep="\t")
# extract relevant enhancer data
enh_df = df[
    ["chromosome", "fix_enhancer_wide_start", "fix_enhancer_wide_end",
     "enhancer_start", "enhancer_end"]
].drop_duplicates(
    keep="first",
    ignore_index=True,
).rename(
    columns={
        "fix_enhancer_wide_start": "wide_start",
        "fix_enhancer_wide_end": "wide_end",
        "enhancer_start": "start",
        "enhancer_end": "end",
    }
)
enh_df["mid"] = enh_df["start"] + ((enh_df["end"] - enh_df["start"]) // 2)
enh_df["type"] = "enhancer"
# same for promoters
prm_df = df[
    ["chromosome", "main_tss_start", "main_tss_end"]
].drop_duplicates(
    keep="first",
    ignore_index=True
).rename(
    columns={
        "main_tss_start": "start",
        "main_tss_end": "end",
    }
)
prm_df["mid"] = prm_df["start"] + ((prm_df["end"] - prm_df["start"]) // 2)
prm_df["wide_start"] = prm_df["mid"] - 1_000
prm_df["wide_end"] = prm_df["mid"] + 1_000
prm_df["type"] = "promoter"
prm_df = prm_df[enh_df.columns]
# merge them into the final dataframe
df = pd.concat([enh_df, prm_df], ignore_index=True)

fasta_extractor = enformer_utils.FastaStringExtractor(args.fasta_path)


def fulco_localeffects_sample_generator_factory(fasta_extractor):
    # interval for the aavs1 site
    aavs1 = kipoiseq.Interval(chrom="chr19", start=55_115_750, end=55_115_780)
    for _, row in df.iterrows():
        row_id = "_".join([str(row["chromosome"]), str(row["mid"]), str(row["type"])])
        interval = kipoiseq.Interval(chrom=row["chromosome"],
                                     start=row["wide_start"],
                                     end=row["wide_end"])
        insert = fasta_extractor.extract(interval)
        for rev_comp in [False, True]:
            for offset in [-43, 0, 43]:
                for seqtype in ["npad", "aavs1"]:
                    if seqtype == "npad":
                        modified_sequence, minbin, maxbin, landmarkbin = \
                            enformer_utils.pad_sequence(
                                insert=insert,
                                shift_five_end=offset,
                                landmark=1000,
                                rev_comp=rev_comp
                            )
                    else:  # seqtype == "aavs1"
                        modified_sequence, minbin, maxbin, landmarkbin = \
                            enformer_utils.insert_sequence_at_landing_pad(
                                insert=insert,
                                lp_interval=aavs1,
                                fasta_extractor=fasta_extractor,
                                mode="replace",
                                shift_five_end=offset,
                                landmark=1000,
                                rev_comp=rev_comp
                            )
                    metadata_dict = {
                        "id": row_id,
                        "chr": row["chromosome"],
                        "wide_start": row["wide_start"],
                        "wide_end": row["wide_end"],
                        "start": row["start"],
                        "end": row["end"],
                        "type": row["type"],
                        "insert_seq": insert,
                        "orient": "rv" if rev_comp else "fw",
                        "offset": offset,
                        "seqtype": seqtype,
                        "landmarkbin": landmarkbin,
                        "minbin": minbin,
                        "maxbin": maxbin
                    }
                    yield modified_sequence, metadata_dict

# prepare generator
sample_generator = \
    fulco_localeffects_sample_generator_factory(fasta_extractor=fasta_extractor)
# we have 2 orientations, 3 offsets and 2 seqtypes
num_of_samples = len(df.index)*2*3*2

# write jobs
utils.batcherator(sample_generator, num_of_samples, args.number_of_jobs, args.output_dir)

# assert that we produced the right number of files
# should have number_of_jobs many pkls
assert args.number_of_jobs == len(glob.glob(os.path.join(args.output_dir,"*.pkl"))), \
       "Number of output files does not match the expected number."
