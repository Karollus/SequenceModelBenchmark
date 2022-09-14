#!/bin/bash

# rename arguments to something human readable
OUTPUT=$1
DATASET_NAME=$2
MODEL_NAME=$3
RESULT_PATH=$4

# check whether file named like the RESULT_PATH already exists
if [[ ! -e $RESULT_PATH ]]; then
    mkdir $RESULT_PATH
fi

# construct filename of the timestamped output
TIMESTAMP=$(date +%Y-%m-%d-%T)
TIMESTAMPED_FILE="$RESULT_PATH/$DATASET_NAME-$MODEL_NAME-$TIMESTAMP.tsv"
# construct filename of the link
LINK_NAME="$RESULT_PATH/$DATASET_NAME-$MODEL_NAME-latest_results.tsv"

# copy the created predictions over there
cp $OUTPUT $TIMESTAMPED_FILE
# create a symbolic link with a fixed name
ln --symbolic --relative --force $TIMESTAMPED_FILE $LINK_NAME
# always return positive exit status so that snakemake doesn't freak
# out and delete output files if this script fails
exit 0
