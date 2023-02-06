# Sequence Model Benchmark

## Quick start

### Download Data from Zenodo

Download `SequenceBenchmark.zip` from our [Zenodo](https://zenodo.org/record/7076228#.YyGd6vexVhE), and extract it into the top level of this project (such that the contained `Data` and `Result` folders are on the same level as the `Code` and `Pipes` folders).

### Set up Conda Environment

Recreate our conda environment using the `environment.yml` file:
```
conda env create -f environment.yml
```
This creates a `sequencebenchmark` conda environment, which needs to be activated before running the predictions:
```
conda activate sequencebenchmark
```

### Running a Prediction Task

Running a prediction task is as simple as supplying the `target_name` as a command line argument to `snakemake` (see below for a [list of available targets](#available-targets)). 
However, as you will most likely have to run this on a cluster things might get a little more complicated than that, as each cluster is set up differently.
For reference, here is a small example script which *could* be used for a cluster running Slurm:
```
#!/bin/bash

# define cluster command for snakemake
SBATCH_CMD="sbatch \
    --nodes=1 \
    --ntasks={resources.ntasks} \
    --cpus-per-task={threads} \
    --mem={resources.mem_mb}M \
    --gres=gpu:{resources.gpu} \
    --parsable \
    --requeue \
    --output=\"logs/$JOBNAME-%A.out\" \
    --job-name=sequence_expression_benchmark"
# run snakemake with said cluster command
snakemake \
    --keep-going \
    --default-resources ntasks=1 mem_mb=1000 gpu=0 \
    --cluster "${SBATCH_CMD}" \
    --cores 64 \
    --jobs 64 \
    --latency-wait 180 \
    $@
```
Note that number of tasks, CPUs/threads, RAM and GPUs are passed to the cluster command via `{resources.ntasks}`, `{threads}`, `{resources.mem_mb}` and `{resources.gpu}`.
To find out the appropriate arguments needed to run this in your specific case, please refer to the [Snakemake documentation](https://snakemake.readthedocs.io/en/stable) and/or ask your sysadmin.

After the prediction task has finished, a `<dataset_name>-<model_name>-latest_results.tsv` link will be made in the `Results` folder. This file is then used in the analysis notebook.

If you can't run the pipeline yourself but need the prediction file for a certain target, please contact us.

## Available Targets
`Snakefile` currently contains the following targets to generate predictions, each of which corresponds to a different dataset, and if available, a certain model (in case no model is specified, Enformer is used):

- `segal_promoters_<model>` for `<model>`: `enformer`, `basenji1` and `basenji2`
- `cohen_tripseq_<model>` for `<model>`: `enformer`, `basenji1` and `basenji2`
- `cohen_patchmpra`
- `findlay_brca`
- `bergmann_exp_<model>` for `<model>`: `enformer` and `basenji2`
- `bergmann_promoteronly`
- `bergmann_enhancercentered`
- `kircher_ingenome_<model>` for `<model>`: `enformer`, `basenji1` and `basenji2`
- `tss_sim_<model>` for `<model>`: `enformer`, `basenji1` and `basenji2`
- `fulco_crispri`
- `avsec_fulltable`
- `avsec_fulltable_fixed`
- `avsec_enhancercentered_<model>` for `<model>`: `enformer` and `basenji2`
- `segal_ism`
- `gtex_eqtl_at_tss_<model>` for `<model>`: `enformer` and `basenji2`
- `ful_gas_localeffects`
- `fulco_in_fulco`

## Directory tour
- `Data`: data needed as input for generating our samples etc.
- `environment.yml`: file to reproduce the pipeline conda environment
- `Enformer_experiments.ipynb`: notebook containing all analyses from the main text
- `Enhancer_shift.ipynb`: notebook containing all analyses pertaining to the in-silico enhancer shift
- `GTEX_manual_match.ipynb`: notebook containing the analyses for Additional File 2.
- `Track_file_prep.ipynb`: notebook used to generate track files
- `Pipes`: pipeline data
  - `Snakefile`: the file defining all pipeline steps
  - `config`: configuration files, contains only a single YAML file describing paths to genome files, prediction tracks used for a sample generator, and the number of jobs to split a dataset into.
  - `scripts`: folder for tiny helper scripts
  - `pickles`: output directory, datasets are split and pickled into job files before the prediction, those pickles are placed here.
  - `predictions`: output directory, predictions generated from the job pickle splits are placed here
  - `result`: output directory, tsvs assembled from prediction splits are placed here
- `Results`: final directory into which results get copied 

The pipeline output directories each contain subdirectories named after the corresponding dataset and the used model, e.g. `segal_promoters/basenji2/`.
