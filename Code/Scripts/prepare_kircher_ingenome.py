import glob
import kipoiseq
import argparse
import os
import pandas as pd

import Code.Utilities.utils as utils
import Code.Utilities.seq_utils as seq_utils
import Code.Utilities.enformer_utils as enformer_utils
import Code.Utilities.basenji1_utils as basenji1_utils
import Code.Utilities.basenji2_utils as basenji2_utils

parser = argparse.ArgumentParser()
parser.add_argument("--fasta-path")
parser.add_argument("--data-path") # "Data/Kircher_saturation_mutagenesis/"
parser.add_argument("--number-of-jobs", type=int)
parser.add_argument("--model", choices=["enformer", "basenji1", "basenji2"])
parser.add_argument("--output-dir")
args = parser.parse_args()

if args.model == "enformer":
    insert_sequence_at_landing_pad = enformer_utils.insert_sequence_at_landing_pad
elif args.model == "basenji1":
    insert_sequence_at_landing_pad = basenji1_utils.insert_sequence_at_landing_pad
elif args.model == "basenji2":
    insert_sequence_at_landing_pad = basenji2_utils.insert_sequence_at_landing_pad

fasta_extractor = enformer_utils.FastaStringExtractor(args.fasta_path)

# aggregate data from several input files
data_folder_name = "CAGI5_train_test_hg19_to_hg38_liftover"
locus_df_list = []
for path in glob.glob(os.path.join(args.data_path, data_folder_name, "*.tsv")):
    locus_df = pd.read_csv(path, sep="\t", names=["Chromosome",
                                                  "Pos_hg38",
                                                  "Ref",
                                                  "Alt",
                                                  "log2FC",
                                                  "Variance"])
    # extract locus identifier from filename
    locus = path.split("/")[-1].split("_")[-1].split(".")[0]
    locus_df["Locus"] = locus
    locus_df_list.append(locus_df)
# concatenate all dataframes into a single one
kircher_df = pd.concat(locus_df_list)
kircher_df.reset_index(drop=True, inplace=True)
# construct a long df out of kircher_df
already_handled_refs = set()
rearranged_df_data = []
for row in kircher_df.iterrows():
    row = row[1]
    ref_uid = (row["Chromosome"], row["Pos_hg38"], row["Locus"])
    if not ref_uid in already_handled_refs:
        ref_row = {"Chromosome": row["Chromosome"],
                   "Pos_hg38": row["Pos_hg38"],
                   "variant_type": "ref",
                   "nucleotide": row["Ref"],
                   "log2FC": 0,
                   "Variance": 0,
                   "Locus": row["Locus"]}
        rearranged_df_data.append(ref_row)
        already_handled_refs.add(ref_uid)
    alt_row = {"Chromosome": row["Chromosome"],
               "Pos_hg38": row["Pos_hg38"],
               "variant_type": "alt",
               "nucleotide": row["Alt"],
               "log2FC": row["log2FC"],
               "Variance": row["Variance"],
               "Locus": row["Locus"]}
    rearranged_df_data.append(alt_row)
kircher_df = pd.DataFrame(rearranged_df_data)

def kircher_ingenome_sample_generator_factory(fasta_extractor):
    for row in kircher_df.iterrows():
        row = row[1] # drop the part we don't need
        genomic_interval = kipoiseq.Interval(chrom=row["Chromosome"],
                                                start=int(row["Pos_hg38"])-1,
                                                end=int(row["Pos_hg38"]))
        landmark = 0
        for offset in [-43,0,43]:
            for orient in ["fw","rv"]:
                # fw variant will be @ seq[393216 // 2 + offset]
                # rv variant will be @ seq[393216 // 2 - offset - 1]
                if orient == "fw":
                    rev_comp = False
                elif orient == "rv":
                    rev_comp = True
                modified_sequence, minbin, maxbin, landmarkbin = \
                    insert_sequence_at_landing_pad(
                        row["nucleotide"],
                        genomic_interval,
                        fasta_extractor,
                        mode="replace",
                        landmark=0,
                        shift_five_end=offset,
                        rev_comp=rev_comp
                    )
                yield modified_sequence, {"chromosome":row["Chromosome"],
                                          "variant_pos":row["Pos_hg38"],
                                          "locus":row["Locus"],
                                          "variant_type":row["variant_type"],
                                          "nucleotide":row["nucleotide"],
                                          "offset":offset,
                                          "orient":orient,
                                          "minbin":minbin,
                                          "maxbin":maxbin,
                                          "landmarkbin":landmarkbin}
                    
# prepare generator
sample_generator = \
    kircher_ingenome_sample_generator_factory(fasta_extractor=fasta_extractor)
num_of_samples = len(kircher_df.index)*3*2

# write jobs
utils.batcherator(sample_generator, num_of_samples, args.number_of_jobs, args.output_dir)

# assert that we produced the right number of files
# should have number_of_jobs many pkls
assert args.number_of_jobs == len(glob.glob(os.path.join(args.output_dir,"*.pkl"))), \
       "Number of output files does not match the expected number."
