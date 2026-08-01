#!/bin/bash
#SBATCH --partition=guest,batch,unmc_gudalab,unmc_cbsb
#SBATCH --time=165:00:00
#SBATCH --mem=200G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=12_diff_enhancers
#SBATCH --output=12_diff_enhancers_%j.out
#SBATCH --error=12_diff_enhancers_%j.err

set -euo pipefail
trap 'echo "❌ ERROR on line $LINENO. Exiting." >&2' ERR

# Fix for set -u + conda.sh expecting PS1
export PS1=${PS1:-"noninteractive"}

# Activate conda env with pandas/numpy/matplotlib
set +u
source /util/opt/anaconda/4.9.2/etc/profile.d/conda.sh
set -u
conda activate snapatac2

echo "Using Python: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"

python << 'EOF'
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ================== CONFIG (tailored to VIP dataset) ==================

# Root where summary CSVs were reorganized by sample:
#   <root>/<sample>/<celltype>/summaries/*_overlaps_enhancer_summary.csv
SUMMARY_ROOT = "/work/avinash/user/CA/CA/VIP_sorted_scBAMs_enhancer_by_sample"

# Output directory for this differential analysis
OUT_DIR = "/work/avinash/user/CA/CA/VIP_differential_enhancers"
os.makedirs(OUT_DIR, exist_ok=True)

print("SUMMARY_ROOT =", SUMMARY_ROOT)
print("OUT_DIR      =", OUT_DIR)

# ================== 1) Discover per-sample summary files ==================

# group = sample name (e.g. MM_568, MM_549, QY_1339)
group_to_files = {}

if not os.path.isdir(SUMMARY_ROOT):
    raise SystemExit(f"❌ SUMMARY_ROOT does not exist: {SUMMARY_ROOT}")

for sample in sorted(os.listdir(SUMMARY_ROOT)):
    sample_dir = os.path.join(SUMMARY_ROOT, sample)
    if not os.path.isdir(sample_dir):
        continue

    # Recursively find summary CSVs for this sample:
    #   <sample>/<celltype>/summaries/*_overlaps_enhancer_summary.csv
    files = []
    for subdir, _, fnames in os.walk(sample_dir):
        for fname in fnames:
            if fname.endswith("_overlaps_enhancer_summary.csv"):
                files.append(os.path.join(subdir, fname))

    if files:
        group_to_files[sample] = sorted(files)

# Report
print("\nDiscovered groups (samples) and file counts:")
for g, flist in group_to_files.items():
    print(f"  {g}: {len(flist)} summary files")

if not group_to_files:
    raise SystemExit("❗️ No summary files found under SUMMARY_ROOT. Nothing to do.")

group_names = sorted(group_to_files.keys())
print("\nGroups (samples) being analyzed:", group_names)

# ================== 2) Count enhancers per sample ==================

def update_counts(file_list):
    """Return dict: (chrom, start, end) -> count across given files."""
    counts = {}
    for f in file_list:
        try:
            df = pd.read_csv(f)
            # assuming at least these columns exist in summary:
            # Chromosome, Start, End
            for _, row in df.iterrows():
                key = (row['Chromosome'], row['Start'], row['End'])
                counts[key] = counts.get(key, 0) + 1
        except Exception as e:
            print("Error reading file:", f, e)
    return counts

# counts_by_group[group][(chr, start, end)] = count
counts_by_group = {}
n_cells_by_group = {}

for g in group_names:
    files = group_to_files[g]
    counts_by_group[g] = update_counts(files)
    n_cells_by_group[g] = len(files)

print("\nNumber of cells (summary files) per group:")
for g in group_names:
    print(f"  {g}: {n_cells_by_group[g]}")

# ================== 3) Build master table ==================

# All unique enhancers across all groups
all_enhancers = set()
for g in group_names:
    all_enhancers.update(counts_by_group[g].keys())

print(f"\nTotal unique enhancers across all groups: {len(all_enhancers)}")

rows = []
for chrom, start, end in all_enhancers:
    row = {
        "Chromosome": chrom,
        "Start": start,
        "End": end,
    }
    # Counts & percentages per group
    for g in group_names:
        count = counts_by_group[g].get((chrom, start, end), 0)
        n_cells = n_cells_by_group[g]
        pct = (count / n_cells * 100) if n_cells else 0.0
        row[f"Count_{g}"] = count
        row[f"Pct_{g}"] = pct
    rows.append(row)

master_df = pd.DataFrame(rows)

# ================== 4) Differential metrics (Max/Min/Diff, log2FC) ==================

pct_cols = [f"Pct_{g}" for g in group_names]

master_df["Max_pct"] = master_df[pct_cols].max(axis=1)
master_df["Min_pct"] = master_df[pct_cols].min(axis=1)
master_df["Diff_pct"] = master_df["Max_pct"] - master_df["Min_pct"]

# Add a small pseudocount to avoid div by zero
master_df["log2FC"] = np.log2((master_df["Max_pct"] + 1) / (master_df["Min_pct"] + 1))

# ================== 5) Exclusivity: enhancer present in exactly one group ==================

def exclusive_group(row):
    present = []
    for g in group_names:
        if row[f"Count_{g}"] > 0:
            present.append(g)
    if len(present) == 1:
        return present[0]
    return ""

master_df["ExclusiveGroup"] = master_df.apply(exclusive_group, axis=1)

# Sort by differential percentage
master_df.sort_values(by="Diff_pct", ascending=False, inplace=True)

# Save master table
master_table_path = os.path.join(OUT_DIR, "differential_enhancers_master_table.csv")
master_df.to_csv(master_table_path, index=False)
print(f"\n✅ Master table saved to:\n  {master_table_path}")

# ================== 6) Heatmap of top 20 differential enhancers ==================

if len(master_df) > 0:
    topN = min(20, len(master_df))
    top = master_df.head(topN).copy()
    heatmap_data = top[pct_cols].to_numpy()

    plt.figure(figsize=(max(8, len(group_names) * 1.2), 0.4 * topN + 4))
    im = plt.imshow(heatmap_data, cmap="YlOrBr", aspect="auto")
    plt.colorbar(im, label="Percentage of Cells")

    # x-axis: group names (samples)
    plt.xticks(np.arange(len(group_names)), group_names, rotation=45, ha="right", fontsize=8)
    plt.gca().xaxis.tick_top()

    # y-axis: enhancers as "chr:start-end"
    enhancer_labels = top["Chromosome"].astype(str) + ":" + top["Start"].astype(str) + "-" + top["End"].astype(str)
    plt.yticks(np.arange(topN), enhancer_labels, fontsize=6)

    # Annotate each cell with its percentage
    for i in range(heatmap_data.shape[0]):
        for j in range(heatmap_data.shape[1]):
            plt.text(j, i, f"{heatmap_data[i, j]:.1f}", ha="center", va="center", fontsize=5)

    plt.title(f"Top {topN} Differential Enhancers (% Cells with Enhancer)")
    plt.xlabel("Sample")
    plt.ylabel("Enhancer (Chromosome:Start-End)")
    plt.tight_layout()

    heatmap_path = os.path.join(OUT_DIR, "differential_enhancers_heatmap.png")
    plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Heatmap saved to:\n  {heatmap_path}")
else:
    print("No enhancers in master_df; skipping heatmap.")

# ================== 7) Barplot: number of exclusive enhancers per sample ==================

exclusive_counts = {}
for g in group_names:
    exclusive_counts[g] = (master_df["ExclusiveGroup"] == g).sum()

plt.figure(figsize=(max(8, len(group_names) * 0.6), 6))
plt.bar(exclusive_counts.keys(), exclusive_counts.values())
plt.ylabel("Number of Exclusive Enhancers")
plt.title("Exclusive Enhancers per Sample")
plt.xticks(rotation=45, ha="right", fontsize=8)
plt.tight_layout()

barplot_path = os.path.join(OUT_DIR, "exclusive_enhancers_barplot.png")
plt.savefig(barplot_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"✅ Bar plot saved to:\n  {barplot_path}")

# ================== 8) Per-sample rankings ==================

for g in group_names:
    pct_col = f"Pct_{g}"
    cols = ["Chromosome", "Start", "End", pct_col, "log2FC"]
    df_sorted = master_df.sort_values(by=pct_col, ascending=False)[cols]
    ranking_path = os.path.join(OUT_DIR, f"{g}_enhancer_rankings.csv")
    df_sorted.to_csv(ranking_path, index=False)
    print(f"✅ Rankings for {g} saved to:\n  {ranking_path}")

print("\n🎉 Differential enhancer analysis completed for all samples.")
EOF