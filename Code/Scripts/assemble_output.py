# concatenates all tsvs generated in the prediction run

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--input-files", nargs="+")
parser.add_argument("--output-file")
args = parser.parse_args()

with open(args.output_file, "wb") as output_tsv:
    with open(args.input_files[0], "rb") as first_tsv:
        output_tsv.write(first_tsv.read())
    for tsv_path in args.input_files[1:]:
        with open(tsv_path, "rb") as tsv:
            next(tsv)
            output_tsv.write(tsv.read())
