#!/usr/bin/env bash
# =============================================================================
# scGRAM — apply config.sh values to all scripts (one-time setup).
#   1. Edit config.sh with your real paths.
#   2. Run:  bash apply_config.sh
# This replaces the /path/to/... , scgram-env , <partition> and <node>
# placeholders in every script (vendored benchmarks/sei_src/ is left untouched).
# =============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=config.sh
source "$HERE/config.sh"

# Refuse to run on unedited defaults.
if [[ "$SCGRAM_ROOT" == "/path/to/scgram" || "$DATA_ROOT" == "/path/to/data" ]]; then
  echo "ERROR: edit config.sh with your real paths before running apply_config.sh" >&2
  exit 1
fi

mapfile -t FILES < <(find "$HERE" -type f \( -name '*.py' -o -name '*.sh' -o -name '*.sbatch' \) \
    -not -path '*/sei_src/*' -not -name 'apply_config.sh' -not -name 'config.sh')

for f in "${FILES[@]}"; do
  sed -i \
    -e "s#/path/to/scgram#${SCGRAM_ROOT}#g" \
    -e "s#/path/to/resources#${RESOURCES_DIR}#g" \
    -e "s#/path/to/conda#${CONDA_BASE}#g" \
    -e "s#/path/to/.cache/huggingface#${HF_HOME}#g" \
    -e "s#/path/to/training_data#${DATA_ROOT}/training_data#g" \
    -e "s#/path/to/model_output#${DATA_ROOT}/model_output#g" \
    -e "s#/path/to/data#${DATA_ROOT}#g" \
    -e "s#scgram-env#${CONDA_ENV}#g" \
    -e "s#<partition>#${SLURM_PARTITION}#g" \
    -e "s#<node>#${COMPUTE_NODE}#g" \
    "$f"
done

echo "Applied config to ${#FILES[@]} files."
# Report any placeholders that remain (should be none except intentional ones).
LEFT=$(grep -rlIE "/path/to/|<partition>|<node>" "$HERE" --exclude-dir=sei_src \
        --exclude=config.sh --exclude=apply_config.sh --exclude=README.md 2>/dev/null | wc -l)
echo "Files still containing a placeholder: ${LEFT}"
