#!/bin/bash
#SBATCH --job-name=MGC1_step5
#SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=/path/to/scgram/step5_logs/MGC_1_step5_slurm_%j.log
#SBATCH --error=/path/to/scgram/step5_logs/MGC_1_step5_slurm_%j.log

set -e
source /path/to/conda/etc/profile.d/conda.sh
conda activate scgram-env
cd /path/to/scgram

echo "=============================================="
echo "MGC_1 STEP 5 (SLURM) start: $(date)"
echo "  jobid=$SLURM_JOB_ID  cpus=$SLURM_CPUS_PER_TASK  gpus=$CUDA_VISIBLE_DEVICES"
echo "=============================================="

python 05_predict_uncharacterized.py --sample MGC_1 --ensemble --model-dir model_output_v2 --workers-per-gpu 2 --batch-size 1024

echo "=============================================="
echo "MGC_1 STEP 5 complete: $(date)"
echo "  predictions written: $(find predictions/MGC_1 -name '*_predictions.csv' 2>/dev/null | wc -l)"
echo "=============================================="
