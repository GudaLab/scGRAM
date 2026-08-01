#!/bin/bash
#SBATCH --partition=<partition>
#SBATCH --nodelist=<node>
#SBATCH --gres=gpu:8
#SBATCH --time=200:00:00
#SBATCH --mem=200G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=benchmark
#SBATCH --output=benchmark_%j.out
#SBATCH --error=benchmark_%j.err

# Full benchmark suite — runs each model in sequence so they don't fight
# resources. Skips any model whose results.json already exists, so re-
# submitting is idempotent.
#
# Order: classical → Basset (no HF download) → Sei → DNABERT-2 → HyenaDNA
#        → NT → Caduceus. Heaviest models last.
#
# Each model phase: (1) extract frozen embeddings to mmap, (2) train MLP
# head with TF + tabular fusion, (3) evaluate on chr21/22/X.

set -euo pipefail
source /path/to/conda/etc/profile.d/conda.sh
conda activate scgram-env

PIPE=/path/to/scgram
cd "$PIPE"
export TORCH_NCCL_WATCHDOG_TIMEOUT_SEC=7200
# HuggingFace cache on a large shared disk (home may not be shared; models are big)
export HF_HOME=/path/to/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
mkdir -p "$HF_HOME"

LABEL_SET="${LABEL_SET:-full}"
RESULTS=$PIPE/benchmarks/results
mkdir -p "$RESULTS"

# ---- Step 2e guard: ensure lab4_*.npy exists when label_set=full ----
# (run_v2_retrain.sh also generates this; harmless if already present.)
if [[ "$LABEL_SET" == "full" ]]; then
    LAB4_SAMPLE="$PIPE/training_data/mmap/lab4_0000.npy"
    if [[ ! -f "$LAB4_SAMPLE" ]]; then
        echo "==== Step 2e (full) — deriving lab4 for benchmark labels ===="
        N_CPUS_BUILD=$(nproc)
        python 02e_build_multilabel.py --label-set full \
               --workers "$N_CPUS_BUILD"
    else
        echo "lab4 already present, skipping Step 2e."
    fi
fi

run_if_missing() {
    local name=$1 ; shift
    if [[ -f "$RESULTS/$name/results.json" ]]; then
        echo "[$name] already complete — skipping."; return
    fi
    echo
    echo "=============================="
    echo " $name"
    echo "=============================="
    "$@" || { echo "[$name] FAILED — continuing to next model"; return; }
}

# ---- 1. Classical baselines (no HF, just sklearn / xgboost) ----
run_if_missing logistic-tf    python benchmarks/run_classical.py \
                                --baseline logistic-tf --label-set "$LABEL_SET"
run_if_missing logistic-kmer  python benchmarks/run_classical.py \
                                --baseline logistic-kmer --label-set "$LABEL_SET"
run_if_missing xgboost        python benchmarks/run_classical.py \
                                --baseline xgboost --label-set "$LABEL_SET"

# ---- 2. Basset (architecture-from-scratch baseline) ----
run_if_missing basset    python benchmarks/run_deep.py --model basset \
                            --phase all --label-set "$LABEL_SET" \
                            --embed-batch 256 --head-batch 8192 --workers 12

# ---- 3. Sei — SKIPPED: not distributed via HF, would need manual
#          PyTorch port of FunctionLab's .pth release. Basset already
#          fills the scratch-trained CNN slot in the comparison.

# ---- 4. DNABERT-2 (downloads ~500 MB) ----
run_if_missing dnabert2  python benchmarks/run_deep.py --model dnabert2 \
                            --phase all --label-set "$LABEL_SET" \
                            --embed-batch 256 --head-batch 4096 --workers 8

# ---- 5. HyenaDNA — SKIPPED: two attempts confirmed unviable timing.
#          0.01 chunks/s at batch=32 → 0.03 chunks/s at batch=256 (only
#          3× from 8× batch) → 4-5 day ETA per pass. The Hyena layers'
#          sequential dependency limits real batch parallelism. 614
#          embed chunks preserved on disk for possible later resume.
# run_if_missing hyenadna  python benchmarks/run_deep.py --model hyenadna \
#                             --phase all --label-set "$LABEL_SET" \
#                             --embed-batch 256 --head-batch 4096 --workers 8

# ---- 6. Nucleotide Transformer 500M (downloads ~2 GB) ----
run_if_missing nt        python benchmarks/run_deep.py --model nt \
                            --phase all --label-set "$LABEL_SET" \
                            --embed-batch 64 --head-batch 4096 --workers 8

# ---- 7. Caduceus-PH ----
run_if_missing caduceus  python benchmarks/run_deep.py --model caduceus \
                            --phase all --label-set "$LABEL_SET" \
                            --embed-batch 128 --head-batch 4096 --workers 8

# ---- aggregate + plot ----
echo
echo "=============================="
echo " Aggregating + plotting"
echo "=============================="
python benchmarks/aggregate.py
echo "Benchmark suite complete."
