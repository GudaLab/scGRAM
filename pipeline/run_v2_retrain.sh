#!/bin/bash
#SBATCH --partition=<partition>
#SBATCH --nodelist=<node>
#SBATCH --gres=gpu:8
#SBATCH --time=120:00:00
#SBATCH --mem=200G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=v2_retrain
#SBATCH --output=v2_retrain_%j.out
#SBATCH --error=v2_retrain_%j.err

# v2 retrain — addresses reviewer comments (1) + (2):
#
#  (1) Val loss never converged in v1: train loss fell while val loss rose
#      monotonically — the model kept becoming more over-confident even
#      after AUROC saturated (see figures/05_calibration.png). Fix:
#        - early-stop on val LOSS (not val AUROC)
#        - cap BCE pos_weight at 5.0 (silencer had ~20×, inflating logits)
#        - label-smoothing ε=0.05 (kills the "push logits to ±∞" gradient)
#
#  (2) Train against every annotation in the source data, not just two
#      heads. Switch to --multilabel-set full → 4 sigmoid heads:
#        is_enhancer, is_promoter, is_genic, is_silencer
#      Downstream code re-buckets as needed (e.g. reg = enh ∨ prom ∨ sil).
#
# Uses ALL 8 GPUs, ALL 128 CPUs, 900 GB RAM (max on <node>).

set -euo pipefail
source /path/to/conda/etc/profile.d/conda.sh
conda activate scgram-env

PIPE=/path/to/scgram
cd "$PIPE"
export TORCH_NCCL_WATCHDOG_TIMEOUT_SEC=7200

N_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
N_CPUS=$(nproc)
# DataLoader workers / GPU — heavy disk-bound work, give it plenty.
DL_WORKERS=$(( (N_CPUS - 8) / N_GPUS ))
[ "$DL_WORKERS" -lt 2 ] && DL_WORKERS=2
[ "$DL_WORKERS" -gt 14 ] && DL_WORKERS=14

BATCH_SIZE="${BATCH_SIZE:-2048}"
LR=$(python -c "import math; print(f'{1e-3*math.sqrt(${N_GPUS}):.4f}')")
# High-confidence convergence per reviewer feedback:
#   epochs: ceiling raised to 40 so cosine schedule has room to fully decay
#   patience: 8 epochs without improvement before stopping
#   min_delta: 1e-3 — an "improvement" must actually move the needle
EPOCHS="${EPOCHS:-40}"
PATIENCE="${PATIENCE:-8}"
MIN_DELTA="${MIN_DELTA:-0.001}"
POS_WEIGHT_CAP="${POS_WEIGHT_CAP:-5.0}"
LABEL_SMOOTHING="${LABEL_SMOOTHING:-0.05}"
EARLY_STOP_METRIC="${EARLY_STOP_METRIC:-loss}"
MULTILABEL_SET="${MULTILABEL_SET:-full}"
MODELS="${MODELS:-cnn,hybrid}"
OUT_ROOT="${OUT_ROOT:-model_output_v2}"

echo "==== v2 retrain ===="
echo "  GPUs=$N_GPUS  CPUs=$N_CPUS  dl_workers/gpu=$DL_WORKERS"
echo "  models=$MODELS  multilabel_set=$MULTILABEL_SET"
echo "  bs=$BATCH_SIZE  lr=$LR  epochs<=$EPOCHS  patience=$PATIENCE  min_delta=$MIN_DELTA"
echo "  early_stop=$EARLY_STOP_METRIC  pos_weight_cap=$POS_WEIGHT_CAP  "
echo "  label_smoothing=$LABEL_SMOOTHING"
echo "  out=$OUT_ROOT"

# ---- Step 2e: derive lab4 if not already present ----
if [[ "$MULTILABEL_SET" == "full" ]]; then
    LAB4_SAMPLE="$PIPE/training_data/mmap/lab4_0000.npy"
    if [[ ! -f "$LAB4_SAMPLE" ]]; then
        echo "==== Step 2e (full) — deriving lab4 from parquet ===="
        python 02e_build_multilabel.py --label-set full --workers "$N_CPUS"
    else
        echo "lab4 already present, skipping Step 2e."
    fi
fi

# ---- Training (with auto-retry) ----
mkdir -p "$OUT_ROOT"
ATTEMPT=0
while [ "$ATTEMPT" -lt 5 ]; do
    ATTEMPT=$((ATTEMPT+1))
    if torchrun --nproc_per_node="$N_GPUS" 04_train.py \
            --data-dir "$PIPE/training_data" \
            --output-dir "$OUT_ROOT" \
            --model-type "$MODELS" \
            --task multilabel \
            --multilabel-set "$MULTILABEL_SET" \
            --early-stop-metric "$EARLY_STOP_METRIC" \
            --pos-weight-cap "$POS_WEIGHT_CAP" \
            --label-smoothing "$LABEL_SMOOTHING" \
            --epochs "$EPOCHS" --patience "$PATIENCE" \
            --min-delta "$MIN_DELTA" \
            --batch-size "$BATCH_SIZE" --lr "$LR" \
            --num-workers "$DL_WORKERS" --amp-dtype bf16 \
            --save-every-n-batches 500; then
        echo "v2 retrain succeeded."
        break
    fi
    echo "  attempt $ATTEMPT failed; sleeping 60s before retry"
    sleep 60
done

echo
echo "==== v2 retrain complete ===="
ls -la "$OUT_ROOT"
