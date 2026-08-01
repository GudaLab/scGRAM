#!/bin/bash
#SBATCH --job-name=MGC1_step1
#SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --output=/path/to/scgram/step1_logs/MGC_1_step1_slurm_%j.log
#SBATCH --error=/path/to/scgram/step1_logs/MGC_1_step1_slurm_%j.log

set -e
source /path/to/conda/etc/profile.d/conda.sh
conda activate scgram-env
cd /path/to/scgram

echo "=============================================="
echo "MGC_1 STEP 1 (SLURM) start: $(date)"
echo "  jobid=$SLURM_JOB_ID  cpus=$SLURM_CPUS_PER_TASK  node=$SLURMD_NODENAME"
echo "  python: $(which python)"
echo "  input beds: $(find /path/to/data/MGC_1_sorted_scBAMs_merged_beds -name '*.bed' | wc -l)"
echo "  already-done beds: $(find /path/to/data/MGC_1_sorted_scBAMs_UNKNOWN_TFBS -name '*_uncharacterized.bed' 2>/dev/null | wc -l)"
echo "=============================================="

# Uses ~11k already-done beds — the script skips cells whose output already exists (verify).
python 01_derive_uncharacterized.py --sample MGC_1 --max-workers 20

echo "=============================================="
echo "MGC_1 STEP 1 complete: $(date)"
echo "  final beds: $(find /path/to/data/MGC_1_sorted_scBAMs_UNKNOWN_TFBS -name '*_uncharacterized.bed' | wc -l)"
echo "=============================================="
