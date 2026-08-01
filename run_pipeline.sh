#!/bin/bash
#SBATCH --partition=<partition>
#SBATCH --gres=gpu:8
#SBATCH --time=700:00:00
#SBATCH --mem=300G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --job-name=regulatory_classifier
#SBATCH --output=regulatory_classifier_%j.out
#SBATCH --error=regulatory_classifier_%j.err

set -euo pipefail
trap 'echo "ERROR on line $LINENO. Exiting." >&2' ERR

export PS1="${PS1-}"

echo "==== Regulatory Region Classification Pipeline ===="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)"
echo

# ------------------------------------------------
# 0) Environment setup
# ------------------------------------------------
module purge || true

set +u
source /path/to/conda/etc/profile.d/conda.sh
set -u
conda activate scgram-env

echo "Python: $(which python)"
echo "Conda env: ${CONDA_DEFAULT_ENV:-unknown}"

# Install PyTorch if not available
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')" 2>/dev/null || {
    echo "Installing PyTorch with CUDA support..."
    pip install torch --index-url https://download.pytorch.org/whl/cu121
}

# Install other dependencies if missing
pip install scikit-learn umap-learn 2>/dev/null || true

# Verify
python -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
"

PIPELINE_DIR="/path/to/scgram"
cd "$PIPELINE_DIR"

# ------------------------------------------------
# Dynamic resource detection — leave 8 CPUs and 20GB RAM for system
# ------------------------------------------------
RESERVE_CPUS=8
RESERVE_MEM_GB=20

TOTAL_CPUS=$(nproc)
# For Step 1 (file I/O) we can use many workers.
# For Step 2 (memory-heavy — each worker holds ~2-3 GB of parsed overlap regions),
# cap at 20 workers to keep in-flight memory bounded to ~60 GB regardless of CPU count.
STEP1_WORKERS=$(( TOTAL_CPUS - RESERVE_CPUS ))
[ "$STEP1_WORKERS" -lt 1 ] && STEP1_WORKERS=1
STEP2_WORKERS=20
[ "$STEP2_WORKERS" -gt "$STEP1_WORKERS" ] && STEP2_WORKERS="$STEP1_WORKERS"
MAX_WORKERS="$STEP1_WORKERS"  # kept for monitor compatibility

# Memory in GB available to the pipeline (informational; Python uses what it needs)
TOTAL_MEM_KB=$(awk '/MemTotal:/ {print $2}' /proc/meminfo)
TOTAL_MEM_GB=$(( TOTAL_MEM_KB / 1024 / 1024 ))
USABLE_MEM_GB=$(( TOTAL_MEM_GB - RESERVE_MEM_GB ))

N_GPUS=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")

echo "Detected resources:"
echo "  CPUs:   ${TOTAL_CPUS} total, reserving ${RESERVE_CPUS}"
echo "  Step 1 workers: ${STEP1_WORKERS} (I/O bound, many workers OK)"
echo "  Step 2 workers: ${STEP2_WORKERS} (memory bound, capped to prevent OOM)"
echo "  Memory: ${TOTAL_MEM_GB} GB total, reserving ${RESERVE_MEM_GB} GB, ${USABLE_MEM_GB} GB usable"
echo "  GPUs:   ${N_GPUS}"

# ------------------------------------------------
# Background monitor log — captures progress snapshots every 60s
# Tail this file to check progress without grepping the main output:
#   tail -f ${PIPELINE_DIR}/monitor.log
# ------------------------------------------------
MONITOR_LOG="${PIPELINE_DIR}/monitor.log"
: > "$MONITOR_LOG"  # truncate
PARENT_PID=$$

(
    while kill -0 "$PARENT_PID" 2>/dev/null; do
        ts=$(date '+%Y-%m-%d %H:%M:%S')
        free_mem_gb=$(awk '/MemAvailable:/ {printf "%.0f", $2/1024/1024}' /proc/meminfo)
        load=$(awk '{print $1, $2, $3}' /proc/loadavg)
        # Active python processes from the pipeline
        py_procs=$(pgrep -u "$(id -u)" -f "0[0-9]_.*\.py" 2>/dev/null | wc -l)
        py_rss_gb=$(ps -u "$(id -u)" -o rss --no-headers 2>/dev/null | awk -v p="$(pgrep -u "$(id -u)" -f "0[0-9]_.*\.py" 2>/dev/null | tr '\n' ' ')" 'BEGIN{n=split(p,a," "); for(i=1;i<=n;i++) pids[a[i]]=1} {sum+=$1} END{printf "%.1f", sum/1024/1024}')
        # Per-step file-level progress for Step 1/2
        s1_done=0
        for d in /path/to/data/*_sorted_scBAMs_UNKNOWN_TFBS; do
            [ -d "$d" ] && s1_done=$(( s1_done + $(find "$d" -name "*_uncharacterized.bed" 2>/dev/null | wc -l) ))
        done
        s2_csvs=$(ls "${PIPELINE_DIR}/training_data/_peaks_"*.csv 2>/dev/null | wc -l)
        s2_chunks=$(ls "${PIPELINE_DIR}/training_data/chunk_"*.npz 2>/dev/null | wc -l)
        # Current step from main log
        cur_step=$(grep -E "^==== STEP" "${PIPELINE_DIR}/pipeline_run.out" 2>/dev/null | tail -1 | sed 's/====//g' | xargs)

        echo "[$ts] step=\"${cur_step:-init}\" free_mem=${free_mem_gb}GB load=${load} py_procs=${py_procs} py_rss=${py_rss_gb}GB step1_files=${s1_done}/68606 step2_csvs=${s2_csvs}/25 step2_chunks=${s2_chunks}" >> "$MONITOR_LOG"
        sleep 60
    done
) &
MONITOR_PID=$!
trap 'kill $MONITOR_PID 2>/dev/null || true' EXIT

echo "Monitor log: $MONITOR_LOG (PID $MONITOR_PID)"

# ------------------------------------------------
# Marker-based idempotence
# ------------------------------------------------
MARKER_DIR="${PIPELINE_DIR}/.markers"
mkdir -p "$MARKER_DIR"

marker_path() { echo "${MARKER_DIR}/$1.done"; }
write_marker() {
    echo "job_id=${SLURM_JOB_ID:-local}" > "$(marker_path "$1")"
    echo "time=$(date -Is)" >> "$(marker_path "$1")"
}
have_marker() { [[ -f "$(marker_path "$1")" ]]; }

# ================================================
# STEP 1: Derive uncharacterized regions per cell
# ================================================
echo
echo "==== STEP 1: Derive uncharacterized regions ===="
if have_marker "step1_uncharacterized"; then
    echo "  Step 1 already complete, skipping."
else
    python 01_derive_uncharacterized.py --max-workers "$STEP1_WORKERS"
    write_marker "step1_uncharacterized"
    echo "  Step 1 complete."
fi

sync; sleep 10

# ================================================
# STEP 2: Aggregate training data + extract sequences
# ================================================
echo
echo "==== STEP 2: Aggregate training data ===="
if have_marker "step2_aggregate"; then
    echo "  Step 2 already complete, skipping."
else
    python 02_aggregate_training_data.py --max-workers "$STEP2_WORKERS"
    write_marker "step2_aggregate"
    echo "  Step 2 complete."
fi

sync; sleep 10

# ================================================
# STEP 2b: Convert chunks to uncompressed mmap .npy
# ================================================
# The compressed .npz chunks decompress to ~4.2 TB of sequence data — far
# more than RAM — so RegulomeDataset must mmap uncompressed per-chunk .npy
# files instead of loading everything. This step does the one-time
# conversion and is idempotent.
echo
echo "==== STEP 2b: Convert chunks to mmap (.npy) ===="
if have_marker "step2b_mmap"; then
    echo "  Step 2b already complete, skipping."
else
    python 02b_convert_to_mmap.py --workers "$STEP2_WORKERS"
    # The script writes its own marker on success; double-check here.
    if [[ -f "$(marker_path step2b_mmap)" ]]; then
        echo "  Step 2b complete."
    else
        echo "  Step 2b did not finish cleanly — check 02b output above."
        exit 1
    fi
fi

sync; sleep 10

# ================================================
# STEP 2e: Build multi-label tensors (is_enhancer, is_silencer)
# ================================================
# Required by --task=multilabel training. Idempotent; skipped if marker
# already present. Cheap (~5 min one-time) — worth running unconditionally.
echo
echo "==== STEP 2e: Build multi-label tensors ===="
if have_marker "step2e_multilabel"; then
    echo "  Step 2e already complete, skipping."
else
    python 02e_build_multilabel.py --workers "$STEP2_WORKERS"
    if [[ -f "$(marker_path step2e_multilabel)" ]]; then
        echo "  Step 2e complete."
    else
        echo "  Step 2e did not finish cleanly — check 02e output above."
        exit 1
    fi
fi

sync; sleep 5

# ================================================
# STEP 2c: Repack sequences to uint8 (optional, env-controlled)
# ================================================
# Shrinks seq_*.npy from float32 one-hot (391 MB/chunk) to uint8 base index
# (25 MB/chunk) — 16x reduction. Total mmap data drops from ~5.9 TB to ~2.0 TB,
# fitting much more comfortably in the 944 GB OS page cache. Idempotent;
# RegulomeDataset auto-detects seq8_*.npy and unpacks to one-hot at read.
#
# Enable with: REPACK_SEQ_UINT8=1 sbatch ...
echo
echo "==== STEP 2c: Repack sequences to uint8 (optional) ===="
if [ "${REPACK_SEQ_UINT8:-0}" != "1" ]; then
    echo "  Skipped (set REPACK_SEQ_UINT8=1 to enable)."
elif have_marker "step2d_seq_uint8"; then
    echo "  Already done (step2d_seq_uint8.done present)."
else
    python 02d_repack_seq_uint8.py --workers "$STEP2_WORKERS"
    if [[ -f "$(marker_path step2d_seq_uint8)" ]]; then
        echo "  Step 2c (uint8 repack) complete."
    else
        echo "  Step 2c did not finish cleanly — check 02d output above."
        exit 1
    fi
fi

sync; sleep 5

# ================================================
# STEP 2f: Bit-pack TF features (optional, env-controlled)
# ================================================
# tf_*.npy is float32 (n, 879) — by far the dominant per-row I/O cost
# (3.5 KB/row vs 0.5 KB for uint8 seq). Bit-packing to (n, 110) uint8 is
# lossless (TF is binary) and shrinks tf bytes by 32×. Total dataset
# bytes drops ~2 TB → ~322 GB — fits cleanly in OS page cache, making
# training essentially memory-speed instead of NFS-bound.
#
# Idempotent. Enable with: REPACK_TF_BITS=1 sbatch ...
echo
echo "==== STEP 2f: Bit-pack TF features (optional) ===="
if [ "${REPACK_TF_BITS:-0}" != "1" ]; then
    echo "  Skipped (set REPACK_TF_BITS=1 to enable)."
elif have_marker "step2f_tf_bits"; then
    echo "  Already done (step2f_tf_bits.done present)."
else
    python 02f_repack_tf_bits.py --workers "$STEP2_WORKERS"
    if [[ -f "$(marker_path step2f_tf_bits)" ]]; then
        echo "  Step 2f (TF bit-pack) complete."
    else
        echo "  Step 2f did not finish cleanly — check 02f output above."
        exit 1
    fi
fi

sync; sleep 5

# ================================================
# STEP 2d: Prewarm OS page cache (optional, env-controlled)
# ================================================
# Sequentially reads mmap files into the OS page cache so the first epoch
# isn't cold. Pure best-effort: only the last ~700 GB read fits in cache.
# Cheap to run (~25 min), no marker — cache is volatile and re-running is
# only useful if cache has gone cold.
#
# Enable with: PREWARM_CACHE=1 sbatch ...
# Optional knobs: PREWARM_TYPES="seq8 tf" (defaults to "seq" or "seq8" based
# on whether the uint8 repack ran), PREWARM_PARALLEL=4
echo
echo "==== STEP 2d: Prewarm OS page cache (optional) ===="
if [ "${PREWARM_CACHE:-0}" != "1" ]; then
    echo "  Skipped (set PREWARM_CACHE=1 to enable)."
else
    # Default to seq8 if the uint8 repack ran; otherwise seq.
    if [[ -z "${PREWARM_TYPES:-}" ]]; then
        if have_marker "step2d_seq_uint8"; then
            PREWARM_TYPES="seq8"
        else
            PREWARM_TYPES="seq"
        fi
    fi
    PREWARM_TYPES="$PREWARM_TYPES" PARALLEL="${PREWARM_PARALLEL:-4}" \
        bash 02c_prewarm_cache.sh
    echo "  Step 2d (cache prewarm) complete."
fi

sync; sleep 5

# ================================================
# STEP 3: Train model (multi-GPU)
# ================================================
echo
echo "==== STEP 3: Train ALL models (CNN + Transformer + Hybrid + XGBoost + Ensemble) ===="
if have_marker "step3_train"; then
    echo "  Step 3 already complete, skipping."
else
    echo "  Training 4 model architectures + ensemble with ${N_GPUS} GPUs..."

    # Scale batch size and LR with GPU count; dataloader workers = CPUs / GPUs.
    # Overrides (env vars):
    #   BATCH_SIZE       — hard override, wins over everything (e.g. BATCH_SIZE=512)
    #   BATCH_SIZE_CAP   — upper bound for the auto-scaled batch size (default 1024).
    #                      Use 512 on A5000 (24 GB) to avoid OOM vs A6000 (48 GB).
    #   LR               — hard override for learning rate
    DL_WORKERS=$(python -c "import os, torch; print(max(2, os.cpu_count() // max(torch.cuda.device_count(), 1) // 2))")
    BATCH_SIZE_CAP="${BATCH_SIZE_CAP:-1024}"
    if [ -n "${BATCH_SIZE:-}" ]; then
        echo "  BATCH_SIZE override: using ${BATCH_SIZE}"
    else
        BATCH_SIZE=$(python -c "print(min(${BATCH_SIZE_CAP}, 128 * ${N_GPUS}))")
    fi
    if [ -z "${LR:-}" ]; then
        LR=$(python -c "import math; print(f'{1e-3 * math.sqrt(${N_GPUS}):.4f}')")
    fi

    echo "  Batch size: ${BATCH_SIZE} (cap=${BATCH_SIZE_CAP}), LR: ${LR}, DataLoader workers/GPU: ${DL_WORKERS}"

    # Task: multilabel = independent sigmoid heads for is_enhancer +
    # is_silencer (3-bucket downstream: enhancer / silencer / dual /
    # neither). multiclass = legacy softmax over LABEL_SCHEME classes.
    TASK="${TASK:-multilabel}"
    # Label scheme: 5class merges TFBS + cis_regulatory into other_regulatory
    # (rows retained; only the classification target is regrouped).
    # Override by setting LABEL_SCHEME=7class. Ignored when TASK=multilabel.
    LABEL_SCHEME="${LABEL_SCHEME:-5class}"
    # Weighted samples per epoch across all ranks (oversamples minorities).
    # Full-dataset epochs are ~529M steps; 20M gives reasonable epoch time
    # while covering minorities many times. Tune via env.
    SAMPLES_PER_EPOCH="${SAMPLES_PER_EPOCH:-20000000}"
    # NCCL watchdog timeout (default is 600 s / 10 min — too short when a
    # rank has a long legitimate stall e.g. eval over a 38M-row set on
    # cold NFS). 2 h default gives plenty of headroom.
    export TORCH_NCCL_WATCHDOG_TIMEOUT_SEC="${TORCH_NCCL_WATCHDOG_TIMEOUT_SEC:-7200}"

    # Auto-retry wrapper. Each retry will resume from the latest last_ckpt.pt
    # for whichever model was in progress (multilabel checkpoints include
    # next_batch_idx so we lose at most ~30-60 min per crash). Transient
    # failures we've seen: NCCL bringup SIGSEGV, pin-memory thread death,
    # NFS hiccups, single-rank watchdog timeout. All resume-safe.
    MAX_TRAIN_RETRIES="${MAX_TRAIN_RETRIES:-8}"
    RETRY_BACKOFF_SEC="${RETRY_BACKOFF_SEC:-60}"
    ATTEMPT=0
    TRAIN_SUCCESS=0
    # Build the command once for reuse + visibility in the log
    if [ "$N_GPUS" -gt 1 ]; then
        TRAIN_CMD=(torchrun --nproc_per_node="$N_GPUS" 04_train.py)
    else
        TRAIN_CMD=(python 04_train.py)
    fi
    # Which DL models to train. Defaults to "cnn,hybrid" — skips the pure
    # Transformer (whose attention over the full 512-bp sequence OOM'd at
    # bs>=2048 on A5000 and was slow at bs=1024). The Hybrid contains a
    # CNN front-end that downsamples 512→128 before attention, so it gets
    # most of the transformer's benefit at a fraction of the compute. The
    # ensemble of CNN + Hybrid is built downstream.
    # Override: MODEL_TYPE=all (or any comma-separated subset)
    MODEL_TYPE="${MODEL_TYPE:-cnn,hybrid}"

    TRAIN_ARGS=(
        --data-dir "${PIPELINE_DIR}/training_data"
        --output-dir "${PIPELINE_DIR}/model_output"
        --model-type "$MODEL_TYPE"
        --task "$TASK"
        --label-scheme "$LABEL_SCHEME"
        --samples-per-epoch "$SAMPLES_PER_EPOCH"
        --epochs 50
        --batch-size "$BATCH_SIZE"
        --lr "$LR"
        --patience 10
        --num-workers "$DL_WORKERS"
    )
    while [ "$ATTEMPT" -lt "$MAX_TRAIN_RETRIES" ]; do
        ATTEMPT=$((ATTEMPT + 1))
        echo
        echo "==== Training attempt ${ATTEMPT}/${MAX_TRAIN_RETRIES} ===="
        # Use `if cmd; then ... else ...` so set -e doesn't trip on nonzero exit.
        if "${TRAIN_CMD[@]}" "${TRAIN_ARGS[@]}"; then
            TRAIN_SUCCESS=1
            echo "  Training attempt ${ATTEMPT} succeeded."
            break
        else
            RC=$?
            # Check whether all DL models are already done — if so, treat as success.
            ALL_DONE=1
            for m in cnn transformer hybrid; do
                [[ -f "${PIPELINE_DIR}/model_output/${m}/done.marker" ]] || ALL_DONE=0
            done
            if [ "$ALL_DONE" -eq 1 ]; then
                echo "  Training exited rc=${RC} but all model done.marker files present — treating as success."
                TRAIN_SUCCESS=1
                break
            fi
            echo "  Training attempt ${ATTEMPT} failed (rc=${RC})."
            if [ "$ATTEMPT" -lt "$MAX_TRAIN_RETRIES" ]; then
                # Quick on-disk sanity sweep so the next attempt starts clean.
                find "${PIPELINE_DIR}/model_output" -maxdepth 3 -name "*.tmp" -mmin +5 -delete 2>/dev/null || true
                echo "  Backing off ${RETRY_BACKOFF_SEC}s before retry..."
                sleep "$RETRY_BACKOFF_SEC"
            fi
        fi
    done

    if [ "$TRAIN_SUCCESS" -ne 1 ]; then
        echo "ERROR: Step 3 failed after ${MAX_TRAIN_RETRIES} attempts."
        exit 1
    fi

    write_marker "step3_train"
    echo "  Step 3 complete."
fi

sync; sleep 10

# ================================================
# STEP 4: Predict on uncharacterized regions
# ================================================
echo
echo "==== STEP 4: Predict uncharacterized regions ===="
if have_marker "step4_predict"; then
    echo "  Step 4 already complete, skipping."
else
    # PREDICT_ENSEMBLE=1 (default) uses the CNN+Hybrid weighted ensemble;
    # set to 0 to use only the single best model (best_model.pt).
    PREDICT_ENSEMBLE="${PREDICT_ENSEMBLE:-1}"
    PRED_ENSEMBLE_FLAG=""
    [ "$PREDICT_ENSEMBLE" = "1" ] && PRED_ENSEMBLE_FLAG="--ensemble"
    python 05_predict_uncharacterized.py \
        --threshold 0.8 \
        --batch-size 4096 \
        $PRED_ENSEMBLE_FLAG
    write_marker "step4_predict"
    echo "  Step 4 complete."
fi

sync; sleep 10

# ================================================
# STEP 5: Visualize results
# ================================================
echo
echo "==== STEP 5: Visualize results ===="
if have_marker "step5_visualize"; then
    echo "  Step 5 already complete, skipping."
else
    python 06_visualize_results.py
    write_marker "step5_visualize"
    echo "  Step 5 complete."
fi

echo
echo "==== Pipeline complete ===="
echo "Results:"
echo "  Model:       ${PIPELINE_DIR}/model_output/"
echo "  Predictions: ${PIPELINE_DIR}/predictions/"
echo "  Figures:     ${PIPELINE_DIR}/figures/"
echo "Date: $(date)"
