#!/usr/bin/env bash
# =============================================================================
# scGRAM — central configuration. Edit the values below ONCE, then run:
#
#     bash apply_config.sh
#
# which fills these paths into every script (replacing the /path/to/... and
# <partition>/<node> placeholders). Shell scripts can also `source config.sh`
# directly to pick up the exported variables.
# =============================================================================

# This repository / working directory (absolute path to the scgram folder)
export SCGRAM_ROOT="/path/to/scgram"

# Root of the input & output data tree (training data, predictions, model_output, ...)
export DATA_ROOT="/path/to/data"

# Reference genome (GRCh38 FASTA) and GENCODE v44 GTF live here
export RESOURCES_DIR="/path/to/resources"

# Conda installation base (the directory containing etc/profile.d/conda.sh)
export CONDA_BASE="/path/to/conda"

# Conda environment name used by the main pipeline
export CONDA_ENV="scgram-env"

# HuggingFace cache (large disk; used only by the foundation-model benchmarks)
export HF_HOME="${DATA_ROOT}/.cache/huggingface"

# SLURM settings (only used by the *.sbatch templates and benchmarks/run_benchmarks.sh)
export SLURM_PARTITION="<partition>"
export COMPUTE_NODE="<node>"
