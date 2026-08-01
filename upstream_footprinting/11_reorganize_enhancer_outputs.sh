#!/bin/bash
set -euo pipefail

# Root where Step 10 put its outputs, organized by celltype
# (i.e. where you currently have D1CaB, D2Pu, D12NAC, etc.)
OLD_ROOT="/work/avinash/user/CA/CA/FOXP2_sorted_scBAMs_enhancer_csv"

# New root we want: organized by sample → celltype → overlaps/summaries
NEW_ROOT="/work/avinash/user/CA/CA/FOXP2_sorted_scBAMs_enhancer_by_sample"

mkdir -p "$NEW_ROOT"

echo "OLD_ROOT = $OLD_ROOT"
echo "NEW_ROOT = $NEW_ROOT"
echo

# Loop over each celltype directory (D1CaB, D2Pu, D12NAC, etc.)
for ct_dir in "$OLD_ROOT"/*; do
    [[ -d "$ct_dir" ]] || continue
    celltype=$(basename "$ct_dir")

    echo "📂 Processing celltype directory: $celltype"

    # Loop over all CSV files in this celltype dir
    shopt -s nullglob
    for f in "$ct_dir"/*.csv; do
        [[ -f "$f" ]] || continue

        filename=$(basename "$f")

        # Only care about Step-10 style files:
        #   <sample>_<celltype>_<barcode>_BD_merged_overlaps_enhancer*.csv
        case "$filename" in
            *_overlaps_enhancer_summary.csv)
                kind="summaries"
                ;;
            *_overlaps_enhancer.csv)
                # make sure we don't treat the summary again
                if [[ "$filename" == *_overlaps_enhancer_summary.csv ]]; then
                    continue
                fi
                kind="overlaps"
                ;;
            *)
                # Skip anything else
                echo "  ⚠️ Skipping non-matching file: $filename"
                continue
                ;;
        esac

        # Strip the known suffix so we can safely parse the sample part
        root_no_suffix="$filename"
        root_no_suffix="${root_no_suffix%_overlaps_enhancer_summary.csv}"
        root_no_suffix="${root_no_suffix%_overlaps_enhancer.csv}"

        # Now root_no_suffix looks like:
        #   MM_568_D1CaB_AAGATGCCTT..._BD_merged
        # We want sample = "MM_568" (first two underscore-separated fields)
        IFS="_" read -r part1 part2 _ <<< "$root_no_suffix"
        sample="${part1}_${part2}"

        # Build target directory: NEW_ROOT/sample/celltype/overlaps|summaries
        target_dir="${NEW_ROOT}/${sample}/${celltype}/${kind}"
        mkdir -p "$target_dir"

        echo "  → Moving $filename to $target_dir/"
        mv "$f" "$target_dir/"
    done
    shopt -u nullglob

    echo
done

echo "✅ Reorganization complete."
echo "Files are now under: $NEW_ROOT/<sample>/<celltype>/{overlaps,summaries}/"
