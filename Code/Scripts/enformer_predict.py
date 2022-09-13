# Given a datastructure of samples (sequence, metadata), this script lets enformer predict for all of them and writes the results

import itertools
import collections
import random
import re
import glob
import math
import os
import pickle
import yaml

import gzip
import pandas as pd
import numpy as np
import tensorflow_hub as hub
import tensorflow as tf
assert tf.config.list_physical_devices('GPU')

# from Code.Utilities.seq_utils import *
# from Code.Utilities.enformer_utils import *
import Code.Utilities.seq_utils as seq_utils
import Code.Utilities.enformer_utils as enformer_utils

import argparse
import subprocess as sp

parser = argparse.ArgumentParser()
parser.add_argument("--data-path")
parser.add_argument("--output-file")
parser.add_argument("--track-dict-path")
parser.add_argument("--verbose", action="store_true")
args = parser.parse_args()


# as per https://stackoverflow.com/a/59571639
def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(
        command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [
        int(x.split()[0]) for i, x in enumerate(memory_free_info)
    ]
    return memory_free_values


batch_size = 8 if get_gpu_memory()[0] < 45000 else 14

transform_path = 'gs://dm-enformer/models/enformer.finetuned.SAD.robustscaler-PCA500-robustscaler.transform.pkl'
model_path = 'https://tfhub.dev/deepmind/enformer/1'

job_name = os.path.splitext(os.path.basename(args.data_path))

with open(args.track_dict_path, 'r') as stream:
    track_dict = yaml.safe_load(stream)

with open(args.data_path, "rb") as data:
    sample_generator = (x for x in pickle.load(data))

model = enformer_utils.Enformer(model_path)


def enformer_batcherator(sample_generator, track_dict, batch_size):
    results = []
    done = False
    batch_idx = 0
    while True:
        batch = []
        # fill batch
        for i in range(batch_size):
            try:
                sample = next(sample_generator)
                batch.append(sample)
            except StopIteration:
                done = True
                break
        if len(batch) == 0:
            break
        if len(batch) == 1:
            batch_seq = seq_utils.one_hot_encode(batch[0][0])[np.newaxis]
        else:
            batch_seq = np.stack(
                [seq_utils.one_hot_encode(x[0]) for x in batch])
        # predict
        predictions = model.predict_on_batch(batch_seq)['human']
        # collect relevant track data
        for idx, sample in enumerate(batch):
            _, metadata = sample
            result_dict = dict()
            result_dict["metadata"] = metadata
            result_dict["metadata"][
                "sample_idx"] = batch_idx * batch_size + idx
            result_dict["predictions"] = {}
            for track in track_dict:
                result_dict["predictions"][track] = np.copy(
                    predictions[idx, :, track_dict[track]])
            results.append(result_dict)
        batch_idx += 1
        if args.verbose and (batch_idx % 5 == 0):
            print(batch_idx * batch_size)
        if done:
            break
    return results


results = enformer_batcherator(sample_generator,
                               track_dict=track_dict,
                               batch_size=batch_size)

rows = []
for result in results:
    # extract metadata/predictions and assemble a row for each
    row = {k: v for k, v in result["metadata"].items()}
    for track in result["predictions"]:
        row[track+"_max"] = \
            np.max(result["predictions"][track][max(row["minbin"]-1,0):row["maxbin"]+2])
        row[track+"_sum"] = \
            np.sum(result["predictions"][track][max(row["minbin"]-1,0):row["maxbin"]+2])
        row[track+"_landmark"] = \
            result["predictions"][track][row["landmarkbin"]]
        row[track+"_landmark_sum"] = \
            np.sum(result["predictions"][track][max(row["landmarkbin"]-1,0):row["landmarkbin"]+2])
        row[track+"_landmark_sum_wide"] = \
            np.sum(result["predictions"][track][max(row["landmarkbin"]-2,0):row["landmarkbin"]+3])
        row[track+"_center_sum"] = \
            np.sum(result["predictions"][track][446:450])
    rows.append(row)

pd.DataFrame(rows).to_csv(args.output_file,
                          mode="w",  # overwrite
                          sep="\t",
                          index=None,
                          header=True)
