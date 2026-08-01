#!/bin/bash
# 02c_prewarm_cache.sh
# Sequentially read all mmap chunk files into the OS page cache. After this,
# RegulomeDataset reads hit RAM until LRU eviction. Cheapest possible "option 3".
# Best-effort: 944 GB RAM minus working-set holds the last ~700 GB of files
# read; with chunk-sorted batches that's a meaningful warm-cache footprint.
#
# Usage:
#   bash 02c_prewarm_cache.sh                        # warm seq (largest data)
#   PREWARM_TYPES="seq tf" bash 02c_prewarm_cache.sh # warm seq + tf
#   PARALLEL=8 bash 02c_prewarm_cache.sh             # parallel reads
#
# Idempotent. Reads cost nothing structural — just pulls pages into cache.

set -euo pipefail

MMAP_DIR="${MMAP_DIR:-/path/to/scgram/training_data/mmap}"
PREWARM_TYPES="${PREWARM_TYPES:-seq}"
PARALLEL="${PARALLEL:-4}"

if [[ ! -d "$MMAP_DIR" ]]; then
    echo "MMAP_DIR not found: $MMAP_DIR" >&2
    exit 1
fi

echo "[prewarm] dir=$MMAP_DIR types='$PREWARM_TYPES' parallel=$PARALLEL"
echo "[prewarm] free RAM before:"
awk '/MemAvailable/ {printf "  MemAvailable: %.1f GB\n", $2/1024/1024}' /proc/meminfo

start=$(date +%s)
total_files=0
for t in $PREWARM_TYPES; do
    files=( "$MMAP_DIR"/${t}_*.npy )
    n=${#files[@]}
    total_files=$((total_files + n))
    echo "[prewarm] warming $n ${t}_*.npy files (parallel=$PARALLEL)"
    # xargs gives us bounded parallelism for `cat ... > /dev/null`
    printf '%s\n' "${files[@]}" \
        | xargs -P "$PARALLEL" -I{} dd if={} of=/dev/null bs=4M status=none
done
elapsed=$(( $(date +%s) - start ))

echo "[prewarm] read $total_files files in ${elapsed}s"
echo "[prewarm] free RAM after:"
awk '/MemAvailable/ {printf "  MemAvailable: %.1f GB\n", $2/1024/1024}' /proc/meminfo
echo "[prewarm] cached:"
awk '/^Cached/ {printf "  Cached: %.1f GB\n", $2/1024/1024}' /proc/meminfo
