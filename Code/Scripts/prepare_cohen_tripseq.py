import re
import glob
import math
import os

import kipoiseq
import pandas as pd

import Code.Utilities.utils as utils
import Code.Utilities.seq_utils as seq_utils
import Code.Utilities.enformer_utils as enformer_utils
import Code.Utilities.basenji1_utils as basenji1_utils
import Code.Utilities.basenji2_utils as basenji2_utils

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--fasta-path")
parser.add_argument("--data-path")  # "Data/Cohen_genomic_environments/"
parser.add_argument("--number-of-jobs", type=int)
parser.add_argument("--model", choices=["enformer", "basenji1", "basenji2"])
parser.add_argument("--output-dir")
args = parser.parse_args()

if args.model == "enformer":
    insert_sequence_at_landing_pad = enformer_utils.insert_sequence_at_landing_pad
    SEQUENCE_LENGTH = enformer_utils.SEQUENCE_LENGTH
elif args.model == "basenji1":
    insert_sequence_at_landing_pad = basenji1_utils.insert_sequence_at_landing_pad
    SEQUENCE_LENGTH = basenji1_utils.SEQUENCE_LENGTH
elif args.model == "basenji2":
    insert_sequence_at_landing_pad = basenji2_utils.insert_sequence_at_landing_pad
    SEQUENCE_LENGTH = basenji2_utils.SEQUENCE_LENGTH

"""This script creates samples for the Cohen Trip-Seq study"""

fasta_extractor = enformer_utils.FastaStringExtractor(args.fasta_path)

# get the plasmid sequence
with open(os.path.join(args.data_path,"trip_plasmid.gb")) as file:
    plasmid_sequence = ""
    started_reading = False
    for line in file.readlines():
        if started_reading:
            line = line.strip().replace(" ","").replace("/","")
            line = re.sub(r'[0-9]+', '', line)
            plasmid_sequence += line
        if line.startswith("ORIGIN"):
            started_reading = True
    plasmid_sequence = plasmid_sequence.upper()

tdTomato = plasmid_sequence[4432:5137]
plasmid_left = plasmid_sequence[1809:4294]
plasmid_right = plasmid_sequence[4426:6128]

# coordinates here already seem to be hg38 according to methods section
trip_expression = pd.read_csv(os.path.join(args.data_path, "trip_seq.tsv"), sep="\t").dropna()
trip_sequences = pd.read_csv(os.path.join(args.data_path, "trip_seq_sequences.tsv"),sep="\t").rename(columns={"Name":"promoter"})
trip_sequences["Orientation"] = trip_sequences["Oligo_id"].apply(lambda x: "RC" if x.split("_")[-1] == "-" else "SS")

# add insertion interval
trip_expression["LP_interval"] = trip_expression.apply(lambda x: kipoiseq.Interval(chrom=x["chr"],
                                                                                  start=x["location"],
                                                                                  end=x["location"]+1)
                                                       ,axis=1)

trip = trip_expression.merge(trip_sequences, on="promoter")


def get_seen_sequence_part(seq):
    sliced_seq = seq[
        enformer_utils.PADDING_UNTIL_SEEN:
        (enformer_utils.PADDING_UNTIL_SEEN+enformer_utils.SEEN_SEQUENCE_LENGTH)
    ]
    assert len(sliced_seq) == enformer_utils.SEEN_SEQUENCE_LENGTH
    return sliced_seq


def calc_flank_and_window_lens(seq_len, window_size):
    flank_len_total = seq_len - window_size
    left_flank_len = math.ceil(flank_len_total / 2)
    right_flank_len = math.floor(flank_len_total / 2)
    assert (left_flank_len + right_flank_len) == flank_len_total
    assert (flank_len_total + window_size) == seq_len
    return left_flank_len, right_flank_len


def assemble_windowed_seq(seq, window_size):
    assert len(seq) == enformer_utils.SEQUENCE_LENGTH
    slicedseq = get_seen_sequence_part(seq)
    left_flank_len, right_flank_len = calc_flank_and_window_lens(len(slicedseq), window_size)
    left_flank = 'N' * (enformer_utils.PADDING_UNTIL_SEEN + left_flank_len)
    right_flank = 'N' * (enformer_utils.PADDING_UNTIL_SEEN + right_flank_len)
    window = slicedseq[left_flank_len:(left_flank_len+window_size)]
    assert len(window) == window_size
    windowed_seq = "".join([left_flank, window, right_flank])
    assert len(windowed_seq) == enformer_utils.SEQUENCE_LENGTH
    assert seq[enformer_utils.SEQUENCE_LENGTH // 2] == \
        windowed_seq[enformer_utils.SEQUENCE_LENGTH // 2]
    return windowed_seq


def enformer_trip_sample_generator_factory(fasta_extractor=fasta_extractor,
                                           orient_promoter=False):
    assert args.model == "enformer"
    for _, trip_row in trip.iterrows():
        lp_interval = trip_row["LP_interval"]
        crs = trip_row["Sequence"]
        for window_type in ["full", "windowed"]:
            for insert_type in ["full", "minimal"]:
                if insert_type == "full":
                    insert = plasmid_left + crs + plasmid_right
                    landmark = len(plasmid_left) + len(crs)//2
                    landmark_rv = len(plasmid_right) + len(crs)//2
                else:
                    insert = crs + tdTomato
                    landmark = len(crs)//2
                    landmark_rv = len(tdTomato) + len(crs)//2
                # reverse if the insert is integrated in rev orientation
                if trip_row["strand"] == "-":
                    insert = seq_utils.rev_comp_sequence(insert)
                    landmark = landmark_rv
                ideal_offset = seq_utils.compute_offset_to_center_landmark(landmark, insert)
                for orient in ["fw", "rv"]:
                    for offset in [-43, 0, 43]:
                        modified_sequence, minbin, maxbin, landmarkbin = \
                            insert_sequence_at_landing_pad(insert,
                                                           lp_interval,
                                                           fasta_extractor,
                                                           shift_five_end=ideal_offset + offset,
                                                           landmark=landmark,
                                                           rev_comp=False if orient == "fw" else True)
                        mid_nuc = modified_sequence[SEQUENCE_LENGTH // 2]
                        if window_type == "full":
                            yield modified_sequence, {"promoter":trip_row["promoter"],
                                                      "LP_interval":lp_interval,
                                                      "orient": orient,
                                                      "window_type": window_type,
                                                      "window_size": -1,
                                                      "insert_type":insert_type,
                                                      "minbin":minbin,
                                                      "maxbin":maxbin,
                                                      "landmarkbin":landmarkbin,
                                                      "offset":offset
                                                      }
                        else:
                            for window_size in [1_001, 3_001, 12_501, 34_501,
                                                enformer_utils.SEEN_SEQUENCE_LENGTH//5,
                                                enformer_utils.SEEN_SEQUENCE_LENGTH//4 + 1,
                                                enformer_utils.SEEN_SEQUENCE_LENGTH//3 + 1,
                                                enformer_utils.SEEN_SEQUENCE_LENGTH//2 + 1,
                                                131_073]:
                                windowed_sequence = assemble_windowed_seq(modified_sequence, window_size)
                                assert mid_nuc == windowed_sequence[enformer_utils.SEQUENCE_LENGTH // 2]
                                yield windowed_sequence, {"promoter":trip_row["promoter"],
                                                          "LP_interval":lp_interval,
                                                          "orient": orient,
                                                          "window_type": window_type,
                                                          "window_size": window_size,
                                                          "insert_type":insert_type,
                                                          "minbin":minbin,
                                                          "maxbin":maxbin,
                                                          "landmarkbin":landmarkbin,
                                                          "offset":offset
                                                          }


def basenji_trip_sample_generator_factory(fasta_extractor=fasta_extractor,
                                          orient_promoter=False):
    assert args.model in ["basenji1", "basenji2"]
    for _, trip_row in trip.iterrows():
        lp_interval = trip_row["LP_interval"]
        crs = trip_row["Sequence"]
        for insert_type in ["full", "minimal"]:
            if insert_type == "full":
                insert = plasmid_left + crs + plasmid_right
                landmark = len(plasmid_left) + len(crs)//2
                landmark_rv = len(plasmid_right) + len(crs)//2
            else:
                insert = crs + tdTomato
                landmark = len(crs)//2
                landmark_rv = len(tdTomato) + len(crs)//2
            # reverse if the insert is integrated in rev orientation
            if trip_row["strand"] == "-":
                insert = seq_utils.rev_comp_sequence(insert)
                landmark = landmark_rv
            ideal_offset = seq_utils.compute_offset_to_center_landmark(landmark, insert)
            for orient in ["fw", "rv"]:
                for offset in [-43, 0, 43]:
                    modified_sequence, minbin, maxbin, landmarkbin = \
                        insert_sequence_at_landing_pad(insert,
                                                       lp_interval,
                                                       fasta_extractor,
                                                       shift_five_end=ideal_offset + offset,
                                                       landmark=landmark,
                                                       rev_comp=False if orient == "fw" else True)
                    yield modified_sequence, {"promoter": trip_row["promoter"],
                                              "LP_interval": lp_interval,
                                              "orient": orient,
                                              "window_type": "full",
                                              "window_size": -1,
                                              "insert_type": insert_type,
                                              "minbin": minbin,
                                              "maxbin": maxbin,
                                              "landmarkbin": landmarkbin,
                                              "offset": offset
                                              }


if args.model == "enformer":
    # prepare generator
    sample_generator = enformer_trip_sample_generator_factory(fasta_extractor=fasta_extractor)
    num_of_samples = len(trip.index)*2*2*3*10
elif args.model in ["basenji1", "basenji2"]:
    sample_generator = basenji_trip_sample_generator_factory(fasta_extractor=fasta_extractor)
    num_of_samples = len(trip.index)*2*2*3

utils.batcherator(sample_generator, num_of_samples, args.number_of_jobs, args.output_dir)

# assert that we produced the right number of files
# should have number_of_jobs many pkls
assert args.number_of_jobs == len(glob.glob(os.path.join(args.output_dir,"*.pkl"))), \
       "Number of output files does not match the expected number."
