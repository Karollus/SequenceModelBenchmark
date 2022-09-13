import glob
import kipoiseq
import argparse
import os
import pandas as pd
import itertools as it

import Code.Utilities.utils as utils
import Code.Utilities.seq_utils as seq_utils
import Code.Utilities.enformer_utils as enformer_utils
import Code.Utilities.basenji2_utils as basenji2_utils

parser = argparse.ArgumentParser()
parser.add_argument("--fasta-path")
parser.add_argument("--data-path") # "Data/gtex_aFC/"
parser.add_argument("--number-of-jobs", type=int)
parser.add_argument("--model", choices=["enformer", "basenji2"])
parser.add_argument("--output-dir")
args = parser.parse_args()

if args.model == "enformer":
    insert_variant_centred_on_tss = enformer_utils.insert_variant_centred_on_tss
elif args.model == "basenji2":
    insert_variant_centred_on_tss = basenji2_utils.insert_variant_centred_on_tss

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

fasta_extractor = enformer_utils.FastaStringExtractor(args.fasta_path)

eqtl_df = pd.read_csv(
    os.path.join(args.data_path, "susie_df_small_blocks.tsv"),
    sep="\t"
)

if args.model == "basenji2":
    maxdist = basenji2_utils.SEEN_SEQUENCE_LENGTH // 2
    eqtl_df = eqtl_df.query("abs_tss_distance_max < @maxdist")

def generate_gtex_eqtl_variants(eqtl_df):
    for _,row in eqtl_df.iterrows():
        variant = kipoiseq.dataclasses.Variant(chrom=chromosome_dict[row["chromosome"]],
                                               pos=row["position"],
                                               ref=row["ref"], 
                                               alt=row["alt"])
        tss_interval = kipoiseq.Interval(chrom=chromosome_dict[row["chromosome"]],
                                         start=row["tss"],
                                         end=row["tss"]+1)
        for allele in ["ref","alt"]:
            for offset in [-43,0,43]:
                for orient in ["fw","rv"]:
                    if orient == "fw":
                        rev_comp = False
                    elif orient == "rv":
                        rev_comp = True
                    modified_sequence, minbin, maxbin, landmarkbin = \
                        insert_variant_centred_on_tss(tss_interval,
                                                      variant,
                                                      allele,
                                                      fasta_extractor,
                                                      shift_five_end=offset,
                                                      rev_comp=False)
                    yield modified_sequence, {"chromosome":chromosome_dict[row["chromosome"]],
                                              "variant_pos":row["position"],
                                              "variant_id":row["variant_id"],
                                              "cs_id":row["cs_id"],
                                              "tissue":row["tissue"],
                                              "variant_seq":row[allele],
                                              "allele":allele,
                                              "offset":offset,
                                              "orient":orient,
                                              "minbin":minbin,
                                              "maxbin":maxbin,
                                              "landmarkbin":landmarkbin}

sample_generator = generate_gtex_eqtl_variants(eqtl_df)
num_of_samples = len(eqtl_df.index) * 2 * 3 * 2

utils.batcherator(sample_generator, num_of_samples, args.number_of_jobs, args.output_dir)

# assert that we produced the right number of files
# should have number_of_jobs many pkls
assert args.number_of_jobs == len(glob.glob(os.path.join(args.output_dir,"*.pkl"))), \
       "Number of output files does not match the expected number."
