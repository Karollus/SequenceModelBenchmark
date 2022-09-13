import kipoi
import argparse
import numpy as np
import pandas as pd
import Code.Utilities.seq_utils as seq_utils
import pickle
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--data-path")
parser.add_argument("--output-file")
parser.add_argument("--track-dict-path")
parser.add_argument("--verbose", action="store_true")
args = parser.parse_args()

model = kipoi.get_model("../Models/Basenji", source="dir")

with open(args.track_dict_path, 'r') as stream:
    track_dict = yaml.safe_load(stream)

with open(args.data_path, "rb") as data:
    sample_generator = (x for x in pickle.load(data))


def basenji1_batcherator(sample_generator, track_dict):
    # basenji1 always has a batchsize of exactly 2!!
    # it is a weird tensorflow graph, so no idea how to fix this
    # so we will work around it
    batch_size = 2
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
            # so we have to pad the batch if we have only one
            batch_seq = seq_utils.one_hot_encode(batch[0][0])[np.newaxis]
            batch_seq = np.concatenate([batch_seq, batch_seq], axis=0)
        else:
            batch_seq = np.stack(
                [seq_utils.one_hot_encode(x[0]) for x in batch])
        # predict
        predictions = model.predict_on_batch(batch_seq)
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
            if len(batch) == 1:
                break  # ignore the padding introduced earlier
        batch_idx += 1
        if args.verbose and (batch_idx % 5 == 0):
            print(batch_idx * batch_size)
        if done:
            break
    return results


results = basenji1_batcherator(sample_generator,
                               track_dict=track_dict)

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
