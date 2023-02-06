import pandas as pd
import os
import Code.Utilities.enformer_utils as enformer_utils
import Code.Utilities.utils as utils
import Code.Utilities.seq_utils as seq_utils
import kipoiseq
import numpy as np
import argparse
import glob
import re
from copy import deepcopy


parser = argparse.ArgumentParser()
parser.add_argument("--fasta-path")
parser.add_argument("--data-path")  # "Data/Fulco_CRISPRi/"
parser.add_argument("--number-of-jobs", type=int)
parser.add_argument("--output-dir")
#parser.add_argument("--enh-wide-type", choices=["avsec", "fixed"])
#parser.add_argument("--shuffles", type=int)
args = parser.parse_args()

enh_shifts = pd.read_csv(os.path.join(args.data_path, "all_enhancer_shifts_design.tsv"),
                         sep="\t")

enhancers = enh_shifts[['ziga_key', 'enhancer_id', 'chrom', 
                        'tss_genome_pos','enh_genome_start', 
                        'enh_genome_end', "enh_is_3prime"]].drop_duplicates()

SEQUENCE_LENGTH = enformer_utils.SEQUENCE_LENGTH

def build_sequence(key, chrom, seq_start, seq_end, 
                   enh_start, enh_end, tss_pos,
                   enh_seq, prom_seq, offset, strand,
                   insert = None, insert_pos = -1):
    base_interval = kipoiseq.Interval(chrom, 
                                      seq_start, 
                                      seq_end)
    base_seq = fasta_extractor.extract(base_interval)
    #print(base_interval)
    #print(base_seq)
    # cut out the actual enhancer
    assert enh_seq == base_seq[enh_start:enh_end], "{}\n{}".format(enh_seq,  base_seq[enh_start:enh_end])
    base_seq = base_seq[:enh_start] + base_seq[enh_end:]
    if insert and insert_pos >= 0:
        # insert at the new position
        base_seq = base_seq[:insert_pos] + insert + base_seq[insert_pos:]
    # check that the tss is indeed at the center
    assert(len([x for x in re.finditer(prom_seq,base_seq)]) > 0), "{}:{}-{}".format(chrom, seq_start, seq_end)
    assert prom_seq == base_seq[tss_pos-500:tss_pos+500], "{}\n{}\n{}\n{}".format(key,
                                                                                  prom_seq,  
                                                                                  base_seq[tss_pos-500:tss_pos+500],
                                                                                  [x for x in re.finditer(prom_seq,base_seq)])
    # perform the shift and the rc 
    base_seq = "N"*max(0,offset) + base_seq[max(0,offset):len(base_seq) + min(0,offset)] + "N"*np.abs(min(0,offset))
    assert len(base_seq) == SEQUENCE_LENGTH, "{} vs. {}".format(len(base_seq), SEQUENCE_LENGTH)
    if strand == "rc":
        base_seq = rev_comp_sequence(base_seq)
    return base_seq

def shuffle_string(string):
    string_as_list = list(string)
    np.random.shuffle(string_as_list)
    return "".join(string_as_list)

def enh_shift_sample_gen_factory(fasta_extractor, num_shuffles=10):
    for _, enh in enhancers.iterrows():
        key = enh["ziga_key"]
        print(key)
        chrom = enh["chrom"]
        enh_is_3prime = enh["enh_is_3prime"]
        enh_interval = kipoiseq.Interval(chrom, 
                                          enh['enh_genome_start'], 
                                          enh['enh_genome_end'])
        enh_seq = fasta_extractor.extract(enh_interval)
        prom_interval = kipoiseq.Interval(chrom, 
                                         enh['tss_genome_pos'] - 500, 
                                         enh['tss_genome_pos'] + 500)
        prom_seq = fasta_extractor.extract(prom_interval)
        # create the possible inserts
        insert_dict = {enh_seq:"enhancer",
                       "N"*2000:"neutral"} # always test these two
        for i in range(num_shuffles):
            insert_dict[shuffle_string(enh_seq)] = "shuffle"
        # predict once without the enhancer
        for offset in [-43, 0, 43]:
            for strand in ["fw", "rv"]:
                seq_start = enh['tss_genome_pos'] - (SEQUENCE_LENGTH//2 + 2000*(not enh_is_3prime))
                seq_end = enh['tss_genome_pos'] + (SEQUENCE_LENGTH//2 + 2000*(enh_is_3prime))
                seq = build_sequence(key=key, chrom=chrom, 
                                     seq_start=seq_start, 
                                     seq_end=seq_end, 
                                     enh_start=enh['enh_genome_start']-seq_start, 
                                     enh_end=enh['enh_genome_end']-seq_start, 
                                     tss_pos=SEQUENCE_LENGTH//2,
                                     enh_seq=enh_seq, 
                                     prom_seq=prom_seq, 
                                     offset=offset, strand=strand)
                yield seq, {"ziga_key": key,
                            "enhancer_id":enh["enhancer_id"],
                            "offset": offset,
                            "orient": strand,
                            "insert_type": "None",
                            "distance": -1,
                            "minbin": 447,
                            "maxbin": 449,
                            "landmarkbin": 448}
        # predict for the inserts
        for _, row in enh_shifts.query('ziga_key == @key').iterrows():
            for insert in insert_dict:
                for offset in [-43, 0, 43]:
                    for strand in ["fw", "rv"]:
                        seq = build_sequence(key=key, chrom=chrom, 
                                             seq_start=row['sequence_start'], 
                                             seq_end=row['sequence_end'], 
                                             enh_start=row['enh_seq_start'], 
                                             enh_end=row['enh_seq_end'], 
                                             tss_pos=row['tss_seq_pos'],
                                             enh_seq=enh_seq, 
                                             prom_seq=prom_seq, 
                                             offset=offset, strand=strand,
                                             insert=insert, 
                                             insert_pos=row['enh_insert_pos'])
                        yield seq, {"ziga_key": key,
                                    "enhancer_id":enh["enhancer_id"],
                                    "offset": offset,
                                    "orient": strand,
                                    "insert_type": insert_dict[insert],
                                    "distance": row['signed_distance'],
                                    "minbin": 447,
                                    "maxbin": 449,
                                    "landmarkbin": 448}

fasta_extractor = enformer_utils.FastaStringExtractor(args.fasta_path)
num_shuffles = 10
num_enh = len(enhancers)
# prepare generator
sample_generator = enh_shift_sample_gen_factory(
    fasta_extractor=fasta_extractor, num_shuffles=num_shuffles)
num_of_samples = num_enh*6 + num_enh*6*2*(num_shuffles+2)*6
# for each enhancer:
# - predict once without insert (with 6 augments)
# - for each distance (20) on each side (2)
#   - predict for each random shuffle (num_shuffles), "N" sequence and the actual enhancer
#     - with 6 augments each time

# write jobs
utils.batcherator(sample_generator, num_of_samples, args.number_of_jobs, args.output_dir)

# assert that we produced the right number of files
# should have number_of_jobs many pkls
assert args.number_of_jobs == len(glob.glob(os.path.join(args.output_dir,"*.pkl"))), \
       "Number of output files does not match the expected number."
