#!/bin/bash
# setup_env.sh — idempotent conda env bootstrap for the regulatory-classifier pipeline.
# Run this once on any new node (home dir is not shared, so the env must be
# recreated per-server). Safe to re-run: skips steps that are already satisfied.
#
# Usage:
#   bash setup_env.sh                 # create env + install all deps
#   bash setup_env.sh --verify        # just check the env is usable

set -euo pipefail

ENV_NAME="scgram-env"
PY_VERSION="3.11"
CUDA_WHEEL_INDEX="https://download.pytorch.org/whl/cu121"

source /path/to/conda/etc/profile.d/conda.sh

if [[ "${1:-}" == "--verify" ]]; then
    conda activate "$ENV_NAME"
    python - <<'PY'
import importlib, sys
mods = ["torch", "duckdb", "pysam", "pyarrow", "numpy", "pandas",
        "sklearn", "xgboost", "umap", "matplotlib", "seaborn", "numba"]
missing = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception as e:
        missing.append(f"{m}: {e}")
import torch
print(f"Python {sys.version.split()[0]}")
print(f"PyTorch {torch.__version__} CUDA={torch.cuda.is_available()} devices={torch.cuda.device_count()}")
if missing:
    print("MISSING:")
    for m in missing: print(f"  {m}")
    sys.exit(1)
print("All required modules importable.")
PY
    exit $?
fi

# ---- 1) Create env if absent ----
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "[setup] env '$ENV_NAME' already exists"
else
    echo "[setup] creating conda env '$ENV_NAME' with python=$PY_VERSION"
    conda create -y -n "$ENV_NAME" "python=$PY_VERSION" pip
fi

conda activate "$ENV_NAME"
echo "[setup] active env: $CONDA_DEFAULT_ENV  python=$(python -V)"

# ---- 2) PyTorch (CUDA 12.1 wheels; driver 12.8 is compatible) ----
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "[setup] torch+CUDA already working"
else
    echo "[setup] installing torch from $CUDA_WHEEL_INDEX"
    pip install --index-url "$CUDA_WHEEL_INDEX" torch
fi

# ---- 3) Core scientific + pipeline deps ----
# Pinning only to lower bounds to match requirements.txt. DuckDB, pysam,
# xgboost, umap-learn, pyarrow, numba are the ones missing from requirements.txt
# but needed by 02/03/04/06 scripts.
pip install --upgrade \
    "numpy>=1.24" \
    "pandas>=2.0" \
    "scipy>=1.10" \
    "scikit-learn>=1.3" \
    "matplotlib>=3.7" \
    "seaborn>=0.12" \
    "pysam>=0.22" \
    "pyarrow>=14" \
    "duckdb>=1.0" \
    "xgboost>=2.0" \
    "umap-learn>=0.5" \
    "numba>=0.58" \
    "tqdm"

# ---- 4) Verify ----
echo "[setup] verifying..."
python - <<'PY'
import torch, duckdb, pysam, pyarrow, numpy, pandas, sklearn, xgboost, umap, numba
print(f"torch    {torch.__version__}  CUDA={torch.cuda.is_available()}  GPUs={torch.cuda.device_count()}")
print(f"duckdb   {duckdb.__version__}")
print(f"pysam    {pysam.__version__}")
print(f"pyarrow  {pyarrow.__version__}")
print(f"numpy    {numpy.__version__}")
print(f"pandas   {pandas.__version__}")
print(f"sklearn  {sklearn.__version__}")
print(f"xgboost  {xgboost.__version__}")
print(f"umap     {umap.__version__}")
print(f"numba    {numba.__version__}")
PY

echo "[setup] done. Activate with: conda activate $ENV_NAME"
