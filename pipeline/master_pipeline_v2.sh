#!/bin/bash
# Master pipeline runner. Chains:
#   (1) wait for v2 training to finish
#   (2) run the benchmark suite (3 classical + 6 deep)
#   (3) reg-vs-non-reg re-bucketing eval
#   (4) final comparison aggregation
# Runs under nohup; safe to detach from any shell.

set -uo pipefail
cd /path/to/scgram
source /path/to/conda/etc/profile.d/conda.sh && conda activate scgram-env
export TORCH_NCCL_WATCHDOG_TIMEOUT_SEC=7200
export OMP_NUM_THREADS=1
export PYTHONMALLOC=malloc
export HF_HOME=/path/to/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
mkdir -p "$HF_HOME"
ulimit -n 131072

LOG=/path/to/scgram/master_pipeline.log

stage() { echo "==== [$(date '+%F %T')] $* ====" | tee -a "$LOG"; }

# -----------------------------------------------------------------------
# 1. Wait for v2 retrain to complete (CNN + Hybrid + Ensemble all done)
# -----------------------------------------------------------------------
stage "Waiting for v2 retrain to finish"
while pgrep -f "04_train.py.*model_output_v2" >/dev/null; do
    sleep 60
done
stage "v2 retrain process exited"

# Inspect what survived
for sub in cnn hybrid ensemble; do
    if [ -f "model_output_v2/$sub/done.marker" ]; then
        echo "  ✓ model_output_v2/$sub/done.marker present" | tee -a "$LOG"
    else
        echo "  ✗ model_output_v2/$sub/done.marker MISSING" | tee -a "$LOG"
    fi
done

# -----------------------------------------------------------------------
# 2. Benchmark suite — runs even if v2 partially failed so we still get
#    a comparison plot
# -----------------------------------------------------------------------
stage "Launching benchmark suite (classical + Basset + Sei + DNABERT-2 + HyenaDNA + NT + Caduceus)"
bash benchmarks/run_benchmarks.sh 2>&1 | tee -a "$LOG"
stage "Benchmark suite finished"

# -----------------------------------------------------------------------
# 3. Reg-vs-non-reg re-bucketing evaluation (reviewer comment #2)
# -----------------------------------------------------------------------
stage "Running reg-vs-non-reg evaluation"
python 12_eval_reg_vs_nonreg.py 2>&1 | tee -a "$LOG"

# -----------------------------------------------------------------------
# 4. Final aggregation — comparison.csv / .json / .md + figure 17
# -----------------------------------------------------------------------
stage "Aggregating final comparison"
python benchmarks/aggregate.py 2>&1 | tee -a "$LOG"

stage "Master pipeline complete"
ls -la benchmarks/results/comparison.* figures/17_benchmark_comparison.png \
       figures/18_reg_vs_nonreg.png 2>/dev/null | tee -a "$LOG"
