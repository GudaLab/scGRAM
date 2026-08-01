#!/bin/bash
#SBATCH --partition=guest,batch,unmc_gudalab,unmc_cbsb
#SBATCH --time=165:00:00
#SBATCH --mem=200G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=13_diff_unknown_regions_v2
#SBATCH --output=13_diff_unknown_regions_v2_%j.out
#SBATCH --error=13_diff_unknown_regions_v2_%j.err

set -euo pipefail
trap 'echo "❌ ERROR on line $LINENO. Exiting." >&2' ERR

# Fix for set -u + conda.sh expecting PS1
export PS1=${PS1:-"noninteractive"}

# Activate conda env with pandas/numpy/matplotlib/seaborn
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
import seaborn as sns
from collections import defaultdict

# ================== CONFIG ==================

# Root where UNKNOWN TFBS beds live (outputs of previous UNKNOWN script)
#   Dopaminergic_sorted_scBAMs_UNKNOWN_TFBS/
#       D12NAC/
#         MM_429_D12NAC_..._BD_merged_UNKNOWN.bed
#       D1CaB/
#       D1Pu/
#       D2CaB/
#       D2Pu/
UNKNOWN_ROOT = "/work/avinash/user/CA/CA/Dopaminergic_sorted_scBAMs_UNKNOWN_TFBS"

# Output directory
OUT_DIR = "/work/avinash/user/CA/CA/Dopaminergic_differential_unknown_regions_celltypes_v2"
os.makedirs(OUT_DIR, exist_ok=True)

# Per-cell UNKNOWN bed file pattern
BED_PATTERN = "*_BD_merged_UNKNOWN.bed"

# --------- KEY MEMORY CONTROL PARAMS ---------
# Only keep regions that appear in at least this many cells globally.
# Increase this to be stricter / more memory-friendly; decrease to be more permissive.
MIN_CELLS_GLOBAL = 10

# Threshold (percentage of cells) to call "high in all celltypes"
COMMON_HIGH_MIN_PCT = 50.0

print("UNKNOWN_ROOT         =", UNKNOWN_ROOT)
print("OUT_DIR              =", OUT_DIR)
print("MIN_CELLS_GLOBAL     =", MIN_CELLS_GLOBAL)
print("COMMON_HIGH_MIN_PCT  =", COMMON_HIGH_MIN_PCT)

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
})

def region_labels_from_df(df):
    return df["Chromosome"].astype(str) + ":" + df["Start"].astype(str) + "-" + df["End"].astype(str)

# ================== DISCOVER CELLTYPE FOLDERS ==================

if not os.path.isdir(UNKNOWN_ROOT):
    raise SystemExit(f"❌ UNKNOWN_ROOT does not exist: {UNKNOWN_ROOT}")

celltype_dirs = sorted(
    d for d in os.listdir(UNKNOWN_ROOT)
    if os.path.isdir(os.path.join(UNKNOWN_ROOT, d))
)

if not celltype_dirs:
    raise SystemExit(f"❗️ No celltype folders found under {UNKNOWN_ROOT}")

print("\nCelltype folders discovered (UNKNOWN):")
for ct in celltype_dirs:
    print("  -", ct)

# ================== PASS 1: GLOBAL REGION COUNTS ==================
# Goal: how many cells (across all celltypes) contain each region.
# We *only* store a single global count per region here, not per celltype.

global_counts = defaultdict(int)      # (chrom,start,end) -> #cells (all celltypes)
celltype_ncells = defaultdict(int)    # celltype -> #cells

print("\n=== PASS 1: Counting global UNKNOWN-region frequencies ===")

for ct in celltype_dirs:
    ct_dir = os.path.join(UNKNOWN_ROOT, ct)
    pattern = os.path.join(ct_dir, BED_PATTERN)
    bed_files = sorted(glob.glob(pattern))
    if not bed_files:
        print(f"ℹ️  No UNKNOWN beds in {ct_dir}, skipping.")
        continue

    print(f"  Celltype {ct}: {len(bed_files)} UNKNOWN beds")

    for fpath in bed_files:
        # each bed file is one "cell"
        celltype_ncells[ct] += 1

        try:
            df = pd.read_csv(
                fpath,
                sep="\t",
                header=None,
                comment="#",
                usecols=[0, 1, 2],
                engine="c",
            )
        except Exception as e:
            print(f"    ❌ Error reading {fpath}: {e}")
            continue

        if df.empty:
            continue

        df.columns = ["Chromosome", "Start", "End"]
        df["Chromosome"] = df["Chromosome"].astype(str)
        df["Start"] = pd.to_numeric(df["Start"], errors="coerce").astype("Int64")
        df["End"]   = pd.to_numeric(df["End"],   errors="coerce").astype("Int64")
        df = df.dropna(subset=["Chromosome", "Start", "End"])
        if df.empty:
            continue

        df["Start"] = df["Start"].astype(int)
        df["End"]   = df["End"].astype(int)

        # Unique regions per cell to avoid double counting in same cell
        reg_set = set((row["Chromosome"], row["Start"], row["End"]) for _, row in df.iterrows())
        for key in reg_set:
            global_counts[key] += 1

# Basic summary
total_regions_global = len(global_counts)
print(f"\nTotal unique UNKNOWN regions across all celltypes (before filtering): {total_regions_global}")
print("\nCells per celltype:")
for ct in sorted(celltype_ncells.keys()):
    print(f"  {ct}: {celltype_ncells[ct]} cells")

if not global_counts:
    raise SystemExit("❗️ No UNKNOWN regions found in any cell. Nothing to analyze.")

# ================== FILTER REGIONS BY GLOBAL FREQUENCY ==================

filtered_regions = {
    region for region, count in global_counts.items()
    if count >= MIN_CELLS_GLOBAL
}

n_filtered = len(filtered_regions)
print(f"\nRegions with global cell-count ≥ {MIN_CELLS_GLOBAL}: {n_filtered}")

if n_filtered == 0:
    raise SystemExit(
        f"❗️ After applying MIN_CELLS_GLOBAL={MIN_CELLS_GLOBAL}, no regions remain. "
        "Try lowering MIN_CELLS_GLOBAL if you want more regions."
    )

# Optionally save the list of filtered regions for reference
filtered_regions_list = []
for (chrom, start, end) in filtered_regions:
    filtered_regions_list.append({"Chromosome": chrom, "Start": start, "End": end})

filtered_df = pd.DataFrame(filtered_regions_list)
filtered_regions_path = os.path.join(
    OUT_DIR,
    f"unknown_regions_globalcount_ge{MIN_CELLS_GLOBAL}.csv"
)
filtered_df.to_csv(filtered_regions_path, index=False)
print(f"✅ Filtered UNKNOWN region list saved to:\n  {filtered_regions_path}")

# ================== PASS 2: PER-CELLTYPE COUNTS FOR FILTERED REGIONS ==================

print("\n=== PASS 2: Counting per-celltype occurrences for filtered UNKNOWN regions ===")

celltypes_used = []
celltype_counts = defaultdict(lambda: defaultdict(int))  # celltype -> (region -> #cells)

for ct in celltype_dirs:
    ct_dir = os.path.join(UNKNOWN_ROOT, ct)
    pattern = os.path.join(ct_dir, BED_PATTERN)
    bed_files = sorted(glob.glob(pattern))
    if not bed_files:
        continue

    # Keep only celltypes that actually had beds
    if celltype_ncells[ct] == 0:
        continue
    celltypes_used.append(ct)

    print(f"  Celltype {ct}: processing {len(bed_files)} beds")

    for fpath in bed_files:
        try:
            df = pd.read_csv(
                fpath,
                sep="\t",
                header=None,
                comment="#",
                usecols=[0, 1, 2],
                engine="c",
            )
        except Exception as e:
            print(f"    ❌ Error reading {fpath}: {e}")
            continue

        if df.empty:
            continue

        df.columns = ["Chromosome", "Start", "End"]
        df["Chromosome"] = df["Chromosome"].astype(str)
        df["Start"] = pd.to_numeric(df["Start"], errors="coerce").astype("Int64")
        df["End"]   = pd.to_numeric(df["End"],   errors="coerce").astype("Int64")
        df = df.dropna(subset=["Chromosome", "Start", "End"])
        if df.empty:
            continue

        df["Start"] = df["Start"].astype(int)
        df["End"]   = df["End"].astype(int)

        reg_set = set((row["Chromosome"], row["Start"], row["End"]) for _, row in df.iterrows())
        # only keep those that survived global filter
        for key in reg_set:
            if key in filtered_regions:
                celltype_counts[ct][key] += 1

celltypes = sorted(set(celltypes_used))
if not celltypes:
    raise SystemExit("❗️ No celltypes had filtered regions. Nothing to analyze further.")

print("\nCelltypes used (after filtering):")
for ct in celltypes:
    print(f"  {ct}: {celltype_ncells[ct]} cells")

print(f"\nNumber of filtered regions analyzed: {n_filtered}")

# ================== BUILD MASTER TABLE FOR FILTERED REGIONS ==================

rows = []
for (chrom, start, end) in filtered_regions:
    row = {
        "Chromosome": chrom,
        "Start": start,
        "End": end,
    }
    for ct in celltypes:
        count = celltype_counts[ct].get((chrom, start, end), 0)
        n_cells = celltype_ncells[ct]
        pct = (count / n_cells * 100.0) if n_cells else 0.0
        row[f"Count_{ct}"] = count
        row[f"Pct_{ct}"]   = pct
    rows.append(row)

master_df = pd.DataFrame(rows)

# ================== DIFFERENTIAL METRICS ==================

pct_cols = [f"Pct_{ct}" for ct in celltypes]
master_df["Max_pct"]  = master_df[pct_cols].max(axis=1)
master_df["Min_pct"]  = master_df[pct_cols].min(axis=1)
master_df["Diff_pct"] = master_df["Max_pct"] - master_df["Min_pct"]
master_df["log2FC"]   = np.log2((master_df["Max_pct"] + 1.0) / (master_df["Min_pct"] + 1.0))

master_df.sort_values(by="Diff_pct", ascending=False, inplace=True)

master_table_path = os.path.join(
    OUT_DIR,
    f"differential_unknown_regions_master_table_celltypes_ge{MIN_CELLS_GLOBAL}cells.csv"
)
master_df.to_csv(master_table_path, index=False)
print(f"\n✅ Master UNKNOWN-region table saved to:\n  {master_table_path}")

# ================== EXCLUSIVE REGIONS PER CELLTYPE ==================

def exclusive_celltype(row):
    present = []
    for ct in celltypes:
        if row[f"Count_{ct}"] > 0:
            present.append(ct)
    if len(present) == 1:
        return present[0]
    return ""

master_df["ExclusiveCelltype"] = master_df.apply(exclusive_celltype, axis=1)

exclusive_counts = {}
exclusive_dir = os.path.join(OUT_DIR, "exclusive_unknown_per_celltype")
os.makedirs(exclusive_dir, exist_ok=True)

cols_to_save = (
    ["Chromosome", "Start", "End"]
    + [c for c in master_df.columns if c.startswith("Count_") or c.startswith("Pct_")]
    + ["Max_pct", "Min_pct", "Diff_pct", "log2FC", "ExclusiveCelltype"]
)

for ct in celltypes:
    sub = master_df[master_df["ExclusiveCelltype"] == ct].copy()
    exclusive_counts[ct] = len(sub)
    if not sub.empty:
        excl_path = os.path.join(
            exclusive_dir,
            f"exclusive_unknown_{ct}_ge{MIN_CELLS_GLOBAL}cells.csv"
        )
        sub[cols_to_save].to_csv(excl_path, index=False)
        print(f"✅ Exclusive UNKNOWN regions for {ct} saved to:\n  {excl_path}")

plt.figure(figsize=(max(8, len(celltypes) * 0.7), 6))
x_labels = list(exclusive_counts.keys())
y_vals   = [exclusive_counts[ct] for ct in x_labels]
plt.bar(x_labels, y_vals)
plt.ylabel("Number of Exclusive UNKNOWN Regions")
plt.title(f"Exclusive UNKNOWN Regions per Celltype (regions ≥ {MIN_CELLS_GLOBAL} cells globally)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

bar_png = os.path.join(OUT_DIR, f"exclusive_unknown_per_celltype_ge{MIN_CELLS_GLOBAL}cells.png")
bar_pdf = os.path.join(OUT_DIR, f"exclusive_unknown_per_celltype_ge{MIN_CELLS_GLOBAL}cells.pdf")
plt.savefig(bar_png, dpi=300, bbox_inches="tight")
plt.savefig(bar_pdf, dpi=300, bbox_inches="tight")
plt.close()
print(f"✅ Barplot of exclusive UNKNOWN regions per celltype saved to:\n  {bar_png}\n  {bar_pdf}")

# ================== REGIONS HIGH IN ALL CELLTYPES ==================

common_high_df = master_df[master_df["Min_pct"] >= COMMON_HIGH_MIN_PCT].copy()
common_high_path = os.path.join(
    OUT_DIR,
    f"unknown_regions_high_in_all_celltypes_minpct{int(COMMON_HIGH_MIN_PCT)}_ge{MIN_CELLS_GLOBAL}cells.csv"
)
common_high_df.to_csv(common_high_path, index=False)
print(f"\n✅ UNKNOWN regions high in ALL celltypes (Min_pct ≥ {COMMON_HIGH_MIN_PCT}%, "
      f"global ≥ {MIN_CELLS_GLOBAL} cells) saved to:\n  {common_high_path}")
print(f"   Number of such regions: {len(common_high_df)}")

if not common_high_df.empty:
    heat_data = common_high_df[pct_cols].to_numpy()
    labels = region_labels_from_df(common_high_df)
    n_regions = len(common_high_df)

    plt.figure(figsize=(max(10, len(celltypes) * 0.8), 0.4 * n_regions + 4))
    im = plt.imshow(heat_data, cmap="YlGnBu", aspect="auto")
    plt.colorbar(im, label="Percentage of Cells")
    plt.xticks(np.arange(len(celltypes)), celltypes, rotation=45, ha="right")
    plt.gca().xaxis.tick_top()
    plt.yticks(np.arange(n_regions), labels, fontsize=6)
    plt.title(
        f"UNKNOWN Regions High in All Celltypes "
        f"(Min_pct ≥ {COMMON_HIGH_MIN_PCT}%, global ≥ {MIN_CELLS_GLOBAL} cells)"
    )
    plt.xlabel("Celltype")
    plt.ylabel("Region (Chromosome:Start-End)")
    plt.tight_layout()

    heat_common_png = os.path.join(
        OUT_DIR,
        f"unknown_common_high_regions_minpct{int(COMMON_HIGH_MIN_PCT)}_ge{MIN_CELLS_GLOBAL}cells.png"
    )
    heat_common_pdf = os.path.join(
        OUT_DIR,
        f"unknown_common_high_regions_minpct{int(COMMON_HIGH_MIN_PCT)}_ge{MIN_CELLS_GLOBAL}cells.pdf"
    )
    plt.savefig(heat_common_png, dpi=300, bbox_inches="tight")
    plt.savefig(heat_common_pdf, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Heatmap for common-high UNKNOWN regions saved to:\n  {heat_common_png}\n  {heat_common_pdf}")

# ================== HEATMAPS & VOLCANO-STYLE PLOTS FOR TOP N ==================

def plot_topN_heatmap(df, N, celltypes, out_prefix):
    if df.empty:
        return
    topN = min(N, len(df))
    top = df.head(topN).copy()
    pct_cols = [f"Pct_{ct}" for ct in celltypes]
    heatmap_data = top[pct_cols].to_numpy()

    plt.figure(figsize=(max(10, len(celltypes) * 0.8), 0.4 * topN + 4))
    im = plt.imshow(heatmap_data, cmap="YlOrBr", aspect="auto")
    plt.colorbar(im, label="Percentage of Cells")
    plt.xticks(np.arange(len(celltypes)), celltypes, rotation=45, ha="right")
    plt.gca().xaxis.tick_top()

    labels = region_labels_from_df(top)
    plt.yticks(np.arange(topN), labels, fontsize=6)

    for i in range(heatmap_data.shape[0]):
        for j in range(heatmap_data.shape[1]):
            plt.text(j, i, f"{heatmap_data[i, j]:.1f}", ha="center", va="center", fontsize=5)

    plt.title(f"Top {topN} Differential UNKNOWN Regions (% Cells per Celltype)")
    plt.xlabel("Celltype")
    plt.ylabel("Region (Chromosome:Start-End)")
    plt.tight_layout()

    png = f"{out_prefix}_heatmap_top{topN}.png"
    pdf = f"{out_prefix}_heatmap_top{topN}.pdf"
    plt.savefig(png, dpi=300, bbox_inches="tight")
    plt.savefig(pdf, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ UNKNOWN heatmap (top {topN}) saved to:\n  {png}\n  {pdf}")

def plot_topN_volcano(df, N, out_prefix):
    if df.empty:
        return
    topN = min(N, len(df))
    top = df.head(topN).copy()
    x = top["log2FC"].to_numpy()
    y = top["Diff_pct"].to_numpy()
    labels = region_labels_from_df(top)

    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, s=30, alpha=0.8, edgecolor="k", linewidth=0.3)
    plt.axvline(0, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("log2( Max_pct + 1 / Min_pct + 1 )")
    plt.ylabel("Diff_pct (Max_pct - Min_pct)")
    plt.title(f"UNKNOWN Regions Volcano-style Plot (Top {topN} by Diff_pct)")

    for xi, yi, lab in zip(x, y, labels):
        plt.text(xi, yi, lab, fontsize=5, ha="center", va="bottom")

    plt.tight_layout()
    png = f"{out_prefix}_volcano_top{topN}.png"
    pdf = f"{out_prefix}_volcano_top{topN}.pdf"
    plt.savefig(png, dpi=300, bbox_inches="tight")
    plt.savefig(pdf, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ UNKNOWN volcano-style plot (top {topN}) saved to:\n  {png}\n  {pdf}")

base_prefix = os.path.join(
    OUT_DIR,
    f"unknown_regions_differential_celltypes_ge{MIN_CELLS_GLOBAL}cells"
)

for N in [25, 50, 100]:
    if len(master_df) == 0:
        break
    plot_topN_heatmap(master_df, N, celltypes, base_prefix)
    plot_topN_volcano(master_df, N, base_prefix)

# ================== DISTRIBUTION OF Diff_pct ==================

if len(master_df) > 0:
    plt.figure(figsize=(8, 5))
    plt.hist(master_df["Diff_pct"], bins=50, alpha=0.8)
    plt.xlabel("Diff_pct (Max_pct - Min_pct)")
    plt.ylabel("Number of UNKNOWN regions")
    plt.title(
        f"Distribution of Differential UNKNOWN Regions Across Celltypes "
        f"(regions ≥ {MIN_CELLS_GLOBAL} cells globally)"
    )
    plt.tight_layout()

    hist_png = os.path.join(
        OUT_DIR,
        f"unknown_diff_pct_distribution_ge{MIN_CELLS_GLOBAL}cells.png"
    )
    hist_pdf = os.path.join(
        OUT_DIR,
        f"unknown_diff_pct_distribution_ge{MIN_CELLS_GLOBAL}cells.pdf"
    )
    plt.savefig(hist_png, dpi=300, bbox_inches="tight")
    plt.savefig(hist_pdf, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ UNKNOWN Diff_pct distribution saved to:\n  {hist_png}\n  {hist_pdf}")

print("\n🎉 UNKNOWN-region analysis (memory-optimized) complete:")
print(f"   • Only regions with global cell-count ≥ {MIN_CELLS_GLOBAL} were analyzed in detail.")
print("   • Differential percentages + log2FC across celltypes.")
print("   • Exclusive UNKNOWN regions per celltype.")
print(f"   • UNKNOWN regions high in ALL celltypes (Min_pct ≥ {COMMON_HIGH_MIN_PCT}%).")
EOF
