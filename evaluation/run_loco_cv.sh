#!/bin/bash
#SBATCH --partition=<partition>
#SBATCH --gres=gpu:8
#SBATCH --time=700:00:00
#SBATCH --mem=300G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --job-name=loco_cv
#SBATCH --output=loco_cv_%j.out
#SBATCH --error=loco_cv_%j.err

# 10-fold chromosome-rotation cross-validation for the multilabel CNN.
# Trains a fresh CNN per fold (held-out chromosome group = test), records
# per-fold test AUROC. Resumable: each fold writes its own done.marker; a
# re-submit skips completed folds.
#
# Uses the fast layout (seq8 + tfb), bf16, capped epochs (models plateau
# early) to keep the 10-fold sweep tractable.

set -euo pipefail
source /path/to/conda/etc/profile.d/conda.sh
conda activate scgram-env

PIPE=/path/to/scgram
cd "$PIPE"
export TORCH_NCCL_WATCHDOG_TIMEOUT_SEC=7200

N_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
DL_WORKERS=$(python -c "import os,torch; print(max(2, os.cpu_count()//max(torch.cuda.device_count(),1)//2))")
BATCH_SIZE="${BATCH_SIZE:-2048}"
LR=$(python -c "import math; print(f'{1e-3*math.sqrt(${N_GPUS}):.4f}')")
EPOCHS="${EPOCHS:-20}"
PATIENCE="${PATIENCE:-4}"
NFOLDS=$(python 10_loco_cv_folds.py --n-folds)

LOCO_ROOT="$PIPE/model_output_loco"
mkdir -p "$LOCO_ROOT"

echo "==== LOCO-CV: ${NFOLDS}-fold chromosome rotation ===="
echo "GPUs=$N_GPUS  bs=$BATCH_SIZE  lr=$LR  epochs<=$EPOCHS  patience=$PATIENCE"
python 10_loco_cv_folds.py --list

for ((k=0; k<NFOLDS; k++)); do
    FOLD_DIR="$LOCO_ROOT/fold_${k}"
    if [[ -f "$FOLD_DIR/cnn/done.marker" ]]; then
        echo "Fold $k already complete, skipping."
        continue
    fi
    SPEC=$(python 10_loco_cv_folds.py --fold "$k")
    TR="${SPEC%%|*}"; REST="${SPEC#*|}"; VA="${REST%%|*}"; TE="${REST##*|}"
    echo
    echo "==== FOLD $k/$((NFOLDS-1)) ===="
    echo "  TEST=$TE"
    echo "  VAL=$VA"
    mkdir -p "$FOLD_DIR"

    # Per-fold auto-retry (transient-failure resilience)
    ATTEMPT=0; OK=0
    while [ "$ATTEMPT" -lt 5 ]; do
        ATTEMPT=$((ATTEMPT+1))
        if torchrun --nproc_per_node="$N_GPUS" 04_train.py \
                --data-dir "$PIPE/training_data" \
                --output-dir "$FOLD_DIR" \
                --model-type cnn \
                --task multilabel \
                --train-chroms "$TR" --val-chroms "$VA" --test-chroms "$TE" \
                --epochs "$EPOCHS" --patience "$PATIENCE" \
                --batch-size "$BATCH_SIZE" --lr "$LR" \
                --num-workers "$DL_WORKERS" --amp-dtype bf16 \
                --save-every-n-batches 500; then
            OK=1; break
        fi
        if [[ -f "$FOLD_DIR/cnn/done.marker" ]]; then OK=1; break; fi
        echo "  fold $k attempt $ATTEMPT failed; retrying in 60s"; sleep 60
    done
    [ "$OK" -eq 1 ] || { echo "FOLD $k FAILED after retries"; exit 1; }
done

echo
echo "==== All folds done — aggregating ===="
python 11_aggregate_loco.py
echo "LOCO-CV complete."
