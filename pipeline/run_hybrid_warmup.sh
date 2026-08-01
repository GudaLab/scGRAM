#!/bin/bash
# Hybrid-only retrain with LR warmup, gradient clip, and conservative LR.
# Addresses the epoch-1-best overfit pattern observed in the first v2 run,
# diagnosed as Transformer optimization instability (NOT memorization
# overfit) — train loss barely moved (0.512 → 0.497 over 9 epochs) while
# val loss drifted up from epoch 1.
#
# Plan A — first attempt (this script):
#   --lr 7e-4              (explicit, not sqrt-scaled; conservative)
#   --warmup-steps 1000    (linear warmup → cosine; per-batch step)
#   --max-grad-norm 1.0    (attention spike insurance)
#   --label-smoothing 0.05 (unchanged from CNN — already tuned)
#   --pos-weight-cap 5.0   (unchanged)
#   --adamw-beta2 0.999    (default — see Plan A.5 below)
#
# Plan A.5 — if Plan A's train loss still doesn't drop, set:
#   --adamw-beta2 0.95
# (faster-adapting variance estimate, helps during warmup of attention)
#
# Plan B — only if A+A.5 fail:
#   --label-smoothing 0.10  --pos-weight-cap 3.0
#
# Output dir is SEPARATE (model_output_v2_hybrid_v2) so the existing
# model_output_v2/hybrid/ is preserved for comparison.

set -uo pipefail
cd /path/to/scgram
source /path/to/conda/etc/profile.d/conda.sh && conda activate scgram-env

# Debug envs (same as last successful v2 retrain)
export NCCL_DEBUG=INFO
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_FAULT_HANDLER=1
export TORCH_NCCL_WATCHDOG_TIMEOUT_SEC=7200
export OMP_NUM_THREADS=1
export PYTHONMALLOC=malloc
ulimit -n 131072

NUM_WORKERS="${NUM_WORKERS:-4}"
LR="${LR:-7e-4}"
WARMUP_STEPS="${WARMUP_STEPS:-1000}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
ADAMW_BETA2="${ADAMW_BETA2:-0.999}"
LABEL_SMOOTH="${LABEL_SMOOTH:-0.05}"
POS_WEIGHT_CAP="${POS_WEIGHT_CAP:-5.0}"
OUT_DIR="${OUT_DIR:-model_output_v2_hybrid_v2}"

# Pre-flight: ensure no other v2 training is running
if pgrep -f "torchrun.*04_train" >/dev/null; then
    echo "Another torchrun is running. Aborting to avoid GPU contention."
    pgrep -af "torchrun.*04_train"
    exit 1
fi

mkdir -p "$OUT_DIR"
TS=$(date +%Y%m%d_%H%M%S)
LOG=hybrid_warmup_${TS}.log

echo "=== Hybrid retrain Plan A ==="
echo "  lr=$LR warmup_steps=$WARMUP_STEPS max_grad_norm=$MAX_GRAD_NORM beta2=$ADAMW_BETA2"
echo "  label_smooth=$LABEL_SMOOTH pos_weight_cap=$POS_WEIGHT_CAP"
echo "  output_dir=$OUT_DIR"
echo "  log=$LOG"

nohup torchrun --nproc_per_node=8 04_train.py \
    --data-dir training_data --output-dir "$OUT_DIR" \
    --model-type hybrid --task multilabel --multilabel-set full \
    --early-stop-metric loss --pos-weight-cap "$POS_WEIGHT_CAP" \
    --label-smoothing "$LABEL_SMOOTH" \
    --epochs 40 --patience 8 --min-delta 0.001 \
    --batch-size 2048 --lr "$LR" \
    --warmup-steps "$WARMUP_STEPS" \
    --max-grad-norm "$MAX_GRAD_NORM" \
    --adamw-beta2 "$ADAMW_BETA2" \
    --num-workers "$NUM_WORKERS" --amp-dtype bf16 \
    --save-every-n-batches 500 \
    > "$LOG" 2>&1 &

NEW_PID=$!
disown -h $NEW_PID
echo "Hybrid retrain PID: $NEW_PID"
echo "Watch:  tail -f $LOG"
