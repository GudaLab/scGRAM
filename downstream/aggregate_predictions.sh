#!/bin/bash
# Fast shell-based per-subtype aggregation.
# Emits: source<TAB>subtype<TAB>n_cells<TAB>total_peaks<TAB>enh<TAB>sil<TAB>both<TAB>neither
set -e
PRED_ROOT=/path/to/scgram/predictions
OUT=${1:-/path/to/scgram/predictions/per_subtype_summary.tsv}

# List all (source, subtype) pairs
pairs_file=$(mktemp)
for source in "$PRED_ROOT"/*/; do
  s=$(basename "$source")
  for st_dir in "$source"*/; do
    [ -d "$st_dir" ] || continue
    st=$(basename "$st_dir")
    echo "$s|$st"
  done
done > "$pairs_file"

N_PAIRS=$(wc -l < "$pairs_file")
echo "aggregating $N_PAIRS (source, subtype) pairs..." >&2

# Process one subtype at a time in parallel — one worker per subtype
# Each worker uses a single awk to scan all files in that subtype dir.
process_subtype() {
  local pair="$1"
  local source="${pair%%|*}"
  local subtype="${pair#*|}"
  local st_dir="$PRED_ROOT/$source/$subtype"

  # count cells
  local n_cells
  n_cells=$(find "$st_dir" -maxdepth 1 -name "*_predictions.csv" | wc -l)
  [ "$n_cells" -eq 0 ] && return 0

  # single awk over all files: emits category counts + total peaks
  awk -F, '
    FNR==1 { next }
    { cnt[$4]++; n++ }
    END {
      printf "%d\t%d\t%d\t%d\t%d\n",
        n,
        cnt["enhancer"]+0,
        cnt["silencer"]+0,
        cnt["enhancer_silencer"]+0,
        cnt["neither"]+0
    }
  ' "$st_dir"/*_predictions.csv | \
  awk -v s="$source" -v st="$subtype" -v nc="$n_cells" -F'\t' '
    { printf "%s\t%s\t%d\t%d\t%d\t%d\t%d\t%d\n", s, st, nc, $1, $2, $3, $4, $5 }
  '
}
export -f process_subtype
export PRED_ROOT

# Emit header
printf "source\tsubtype\tn_cells\ttotal_peaks\tenhancer\tsilencer\tboth\tneither\n" > "$OUT"

# Run in parallel (up to 8 subtypes at a time — I/O bound so more helps but risks overload)
cat "$pairs_file" | xargs -I{} -P 8 bash -c 'process_subtype "$@"' _ {} >> "$OUT"

rm -f "$pairs_file"

echo "wrote $OUT" >&2
wc -l "$OUT" >&2
