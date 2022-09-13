# Expression Benchmark

## Quickstart

`Snakefile` currently contains the following targets to generate predictions, each of which corresponds to a different dataset:
```
rule segal_promoters:
    input: "result/segal_promoters/all_predictions.tsv",

rule cohen_tripseq:
    input: "result/cohen_tripseq/all_predictions.tsv",

rule cohen_patchmpra:
    input: "result/cohen_patchmpra/all_predictions.tsv"

rule findlay_brca:
    input: "result/findlay_brca/all_predictions.tsv"

rule weiss_constructs:
    input: "result/weiss_constructs/all_predictions.tsv"
```

To run one of those targets, simply run `run_slurm.sh` with the target name as argument.

Example:
```
./run_slurm_jobs.sh findlay_brca
```

If you want to force Snakemake to regenerate _all_ files which are needed to create the target file, add the `--forceall` flag:
```
./run_slurm_jobs.sh findlay_brca --forceall
```
Note that previous results will be overwritten this way!

## Pipeline directory tour

- `Pipes`: pipeline data
  - `config`: configuration files, currently only a single YAML
  - `envs`: currently empty, for Snakemake specific conda environments
  - `logs`: log files generated by `run_slurm_jobs.sh` are placed here
  - `pickles`: output directory, datasets are split into smaller jobs for the model, those job pickles are placed here
  - `predictions`: output directory, predictions generated from the job pickle splits are placed here
  - `result`: output directory, tsvs assembled from prediction splits are placed here
  - `run_slurm_jobs.sh` & `slurm-status.py`: scripts used to launch slurm jobs. Note that some slurm configs are hardcoded into the former file!
  - `Snakefile`: the actual pipeline

The output directories each contain subdirectories named after the corresponding dataset, e.g. `segal_promoters`, `cohen_tripseq`, etc.