#!/bin/bash
#SBATCH --partition=guest,batch,unmc_gudalab,unmc_cbsb
#SBATCH --time=165:00:00
#SBATCH --mem=200G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=12_diff_regulome_celltypes
#SBATCH --output=12_diff_regulome_celltypes_%j.out
#SBATCH --error=12_diff_regulome_celltypes_%j.err

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
# For VIP:
# SUMMARY_ROOT = "/work/avinash/user/CA/CA/VIP_sorted_scBAMs_REGULOME_csv"
# OUT_DIR      = "/work/avinash/user/CA/CA/VIP_differential_regulome_celltypes"

# For VIP (example – you will adjust these to your actual paths):
#   /work/.../VIP_sorted_scBAMs_REGULOME_csv/VIP_1, VIP_2, VIP_3/...
SUMMARY_ROOT = "/work/avinash/user/CA/CA/VIP_sorted_scBAMs_REGULOME_csv"
OUT_DIR      = "/work/avinash/user/CA/CA/VIP_differential_regulome_celltypes"

# Pattern (same as before)
SUMMARY_PATTERN = "*_merged_overlaps_REGULOME_summary.csv"

# Output directory for REGULOME analysis
os.makedirs(OUT_DIR, exist_ok=True)

# Subdirectory for similarity matrices / plots
SIM_DIR = os.path.join(OUT_DIR, "Similarity")
os.makedirs(SIM_DIR, exist_ok=True)

print("SUMMARY_ROOT =", SUMMARY_ROOT)
print("OUT_DIR      =", OUT_DIR)
print("SIM_DIR      =", SIM_DIR)

# Style
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
})

# ================== helpers ==================

def parse_sample_from_basename(basename):
    """
    Parse *sample* from a basename like:
      MM_429_D12NAC_AAGAT..._BD_merged_overlaps_REGULOME_summary
      MM_549_VIP_1_AACG..._BD_merged_overlaps_REGULOME_summary

    We treat:
      sample = parts[0] + "_" + parts[1]  (e.g. MM_429, QY_1339).
    Celltype is taken from the DIRECTORY name, not the filename.
    """
    parts = basename.split("_")
    if len(parts) < 2:
        return "UNKNOWN"
    return "_".join(parts[0:2])

def region_labels_from_df(df):
    return df["Chromosome"].astype(str) + ":" + df["Start"].astype(str) + "-" + df["End"].astype(str)

def compute_jaccard_matrix(cell_enh_dict):
    """Return (cell_ids, matrix) where matrix[i,j] is 0–100 Jaccard %."""
    cell_ids = sorted(cell_enh_dict.keys())
    n = len(cell_ids)
    if n == 0:
        return [], None

    mat = np.zeros((n, n), dtype=float)
    for i in range(n):
        s1 = cell_enh_dict[cell_ids[i]]
        for j in range(i, n):
            if i == j:
                mat[i, j] = 100.0
            else:
                s2 = cell_enh_dict[cell_ids[j]]
                inter = len(s1 & s2)
                union = len(s1 | s2)
                val = (inter / union * 100.0) if union > 0 else 0.0
                mat[i, j] = mat[j, i] = val
    return cell_ids, mat

def plot_jaccard_heatmap(cell_ids, mat, title, out_prefix):
    n = len(cell_ids)
    if n == 0 or mat is None:
        return
    df = pd.DataFrame(mat, index=cell_ids, columns=cell_ids)

    max_non_diag = df.replace(100.0, np.nan).max().max()
    if np.isnan(max_non_diag):
        max_non_diag = 100.0

    plt.figure(figsize=(min(20, max(8, n * 0.25)), min(20, max(8, n * 0.25))))
    mask = np.eye(n, dtype=bool)
    ax = sns.heatmap(
        df,
        cmap="viridis",
        square=True,
        vmin=0,
        vmax=max_non_diag,
        mask=mask,
        cbar_kws={"label": "Jaccard overlap (%)"},
    )
    for i in range(n):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, color='red', lw=0))
    plt.title(title)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()

    png = f"{out_prefix}.png"
    pdf = f"{out_prefix}.pdf"
    plt.savefig(png, dpi=300, bbox_inches="tight")
    plt.savefig(pdf, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Jaccard heatmap saved to:\n  {png}\n  {pdf}")

    csv_path = f"{out_prefix}.csv"
    df.to_csv(csv_path)
    print(f"✅ Jaccard matrix CSV saved to:\n  {csv_path}")

# ================== 1) Traverse REGULOME summary files ==================

celltype_counts   = defaultdict(lambda: defaultdict(int))   # celltype -> region -> #cells
celltype_ncells   = defaultdict(int)                        # celltype -> #cells
sample_counts     = defaultdict(lambda: defaultdict(int))   # sample   -> region -> #cells
sample_ncells     = defaultdict(int)                        # sample   -> #cells
ct_sample_counts  = defaultdict(lambda: defaultdict(int))   # (sample,celltype) -> region -> #cells
ct_sample_ncells  = defaultdict(int)                        # (sample,celltype) -> #cells

all_regions = set()

# For Jaccard
cell_enh_by_celltype = defaultdict(dict)   # celltype -> {cell_id -> set(regions)}
global_cell_enh      = {}                  # cell_id -> set(regions)

if not os.path.isdir(SUMMARY_ROOT):
    raise SystemExit(f"❌ SUMMARY_ROOT does not exist: {SUMMARY_ROOT}")

# Celltype dirs:
#   VIP: D12NAC, D1CaB, D1Pu, D2CaB, D2Pu
#   VIP: VIP_1, VIP_2, VIP_3  (for example)
for dir_celltype in sorted(os.listdir(SUMMARY_ROOT)):
    ct_dir = os.path.join(SUMMARY_ROOT, dir_celltype)
    if not os.path.isdir(ct_dir):
        continue

    # This directory name IS the celltype label we should use
    celltype = dir_celltype

    # Find all *_merged_overlaps_REGULOME_summary.csv in this celltype
    pattern = os.path.join(ct_dir, SUMMARY_PATTERN)
    summary_files = sorted(glob.glob(pattern))
    if not summary_files:
        continue

    for fpath in summary_files:
        basename = os.path.splitext(os.path.basename(fpath))[0]

        # Only parse sample from filename; celltype is the directory name
        sample = parse_sample_from_basename(basename)

        try:
            df = pd.read_csv(fpath, low_memory=False)
        except Exception as e:
            print(f"Error reading {fpath}: {e}")
            continue

        # Build region set for this cell
        try:
            reg_set = set(
                (row["Chromosome"], row["Start"], row["End"])
                for _, row in df.iterrows()
            )
        except KeyError as e:
            print(f"Missing expected columns in {fpath}: {e}")
            continue

        if not reg_set:
            continue

        cell_id = basename  # unique ID that encodes sample, celltype, barcode, etc.

        cell_enh_by_celltype[celltype][cell_id] = reg_set
        global_cell_enh[cell_id] = reg_set

        celltype_ncells[celltype] += 1
        sample_ncells[sample] += 1
        ct_sample_ncells[(sample, celltype)] += 1

        for key in reg_set:
            all_regions.add(key)
            celltype_counts[celltype][key] += 1
            sample_counts[sample][key] += 1
            ct_sample_counts[(sample, celltype)][key] += 1

celltypes = sorted(celltype_counts.keys())
samples   = sorted(sample_counts.keys())

print("\nCelltypes discovered (REGULOME) and #cells:")
for ct in celltypes:
    print(f"  {ct}: {celltype_ncells[ct]} cells")

print("\nSamples discovered (REGULOME) and #cells:")
for s in samples:
    print(f"  {s}: {sample_ncells[s]} cells")

print(f"\nTotal unique REGULOME regions across all celltypes/samples: {len(all_regions)}")

if not celltypes:
    raise SystemExit("❗️ No celltypes found. Nothing to analyze (REGULOME).")

# ================== 2) Build celltype-level master table ==================

rows = []
for (chrom, start, end) in all_regions:
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

# ================== 3) Differential metrics ==================

pct_cols = [f"Pct_{ct}" for ct in celltypes]
master_df["Max_pct"]  = master_df[pct_cols].max(axis=1)
master_df["Min_pct"]  = master_df[pct_cols].min(axis=1)
master_df["Diff_pct"] = master_df["Max_pct"] - master_df["Min_pct"]
master_df["log2FC"]   = np.log2((master_df["Max_pct"] + 1.0) / (master_df["Min_pct"] + 1.0))

# ================== 4) Exclusive regions per celltype ==================

def exclusive_celltype(row):
    present = []
    for ct in celltypes:
        if row[f"Count_{ct}"] > 0:
            present.append(ct)
    if len(present) == 1:
        return present[0]
    return ""

master_df["ExclusiveCelltype"] = master_df.apply(exclusive_celltype, axis=1)
master_df.sort_values(by="Diff_pct", ascending=False, inplace=True)

master_table_path = os.path.join(OUT_DIR, "differential_regulome_master_table_celltypes.csv")
master_df.to_csv(master_table_path, index=False)
print(f"\n✅ Master REGULOME celltype-level table saved to:\n  {master_table_path}")

# ================== 5) Heatmaps & volcano plots ==================

for N in [25, 50, 100]:
    if len(master_df) == 0:
        break
    topN = min(N, len(master_df))
    top = master_df.head(topN).copy()
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

    plt.title(f"Top {topN} Differential REGULOME Regions (% Cells per Celltype)")
    plt.xlabel("Celltype")
    plt.ylabel("Region (Chromosome:Start-End)")
    plt.tight_layout()

    heat_png = os.path.join(OUT_DIR, f"regulome_heatmap_top{topN}_celltypes.png")
    heat_pdf = os.path.join(OUT_DIR, f"regulome_heatmap_top{topN}_celltypes.pdf")
    plt.savefig(heat_png, dpi=300, bbox_inches="tight")
    plt.savefig(heat_pdf, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ REGULOME heatmap (top {topN}) saved to:\n  {heat_png}\n  {heat_pdf}")

for N in [25, 50, 100]:
    if len(master_df) == 0:
        break
    topN = min(N, len(master_df))
    top = master_df.head(topN).copy()
    x = top["log2FC"].to_numpy()
    y = top["Diff_pct"].to_numpy()
    labels = region_labels_from_df(top)

    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, s=30, alpha=0.8, edgecolor="k", linewidth=0.3)
    plt.axvline(0, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("log2( Max_pct + 1 / Min_pct + 1 )")
    plt.ylabel("Diff_pct (Max_pct - Min_pct)")
    plt.title(f"REGULOME Volcano Plot (Top {topN} Differential Regions)")

    for xi, yi, lab in zip(x, y, labels):
        plt.text(xi, yi, lab, fontsize=5, ha="center", va="bottom")

    plt.tight_layout()
    volc_png = os.path.join(OUT_DIR, f"regulome_volcano_top{topN}_celltypes.png")
    volc_pdf = os.path.join(OUT_DIR, f"regulome_volcano_top{topN}_celltypes.pdf")
    plt.savefig(volc_png, dpi=300, bbox_inches="tight")
    plt.savefig(volc_pdf, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ REGULOME volcano (top {topN}) saved to:\n  {volc_png}\n  {volc_pdf}")

if len(master_df) > 0:
    plt.figure(figsize=(8, 5))
    plt.hist(master_df["Diff_pct"], bins=50, alpha=0.8)
    plt.xlabel("Diff_pct (Max_pct - Min_pct)")
    plt.ylabel("Number of REGULOME regions")
    plt.title("Distribution of Differential REGULOME Percentages Across Celltypes")
    plt.tight_layout()

    hist_png = os.path.join(OUT_DIR, "regulome_diff_pct_distribution.png")
    hist_pdf = os.path.join(OUT_DIR, "regulome_diff_pct_distribution.pdf")
    plt.savefig(hist_png, dpi=300, bbox_inches="tight")
    plt.savefig(hist_pdf, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ REGULOME Diff_pct distribution saved to:\n  {hist_png}\n  {hist_pdf}")

# ================== 6) Exclusive regions per celltype ==================

exclusive_counts = {}
exclusive_dir = os.path.join(OUT_DIR, "exclusive_regulome_per_celltype")
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
        excl_path = os.path.join(exclusive_dir, f"exclusive_regulome_{ct}.csv")
        sub[cols_to_save].to_csv(excl_path, index=False)
        print(f"✅ Exclusive REGULOME regions for {ct} saved to:\n  {excl_path}")

plt.figure(figsize=(max(8, len(celltypes) * 0.7), 6))
x_labels = list(exclusive_counts.keys())
y_vals   = [exclusive_counts[ct] for ct in x_labels]
plt.bar(x_labels, y_vals)
plt.ylabel("Number of Exclusive REGULOME Regions")
plt.title("Exclusive REGULOME Regions per Celltype")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

bar_png = os.path.join(OUT_DIR, "exclusive_regulome_per_celltype.png")
bar_pdf = os.path.join(OUT_DIR, "exclusive_regulome_per_celltype.pdf")
plt.savefig(bar_png, dpi=300, bbox_inches="tight")
plt.savefig(bar_pdf, dpi=300, bbox_inches="tight")
plt.close()
print(f"✅ Barplot of exclusive REGULOME features per celltype saved to:\n  {bar_png}\n  {bar_pdf}")

# ================== 7) Per-(sample,celltype) rankings ==================

subtype_dir = os.path.join(OUT_DIR, "per_sample_celltype_regulome_rankings")
os.makedirs(subtype_dir, exist_ok=True)

for (sample, ct), reg_dict in ct_sample_counts.items():
    n_cells = ct_sample_ncells[(sample, ct)]
    if n_cells == 0:
        continue

    rows_sub = []
    for (chrom, start, end), count in reg_dict.items():
        pct = (count / n_cells * 100.0) if n_cells else 0.0
        rows_sub.append({
            "Chromosome": chrom,
            "Start": start,
            "End": end,
            "Count_in_subtype": count,
            "Pct_in_subtype": pct,
            "Sample": sample,
            "Celltype": ct,
        })

    if not rows_sub:
        continue

    df_sub = pd.DataFrame(rows_sub)
    df_sub.sort_values(by="Pct_in_subtype", ascending=False, inplace=True)

    sub_out_dir = os.path.join(subtype_dir, sample, ct)
    os.makedirs(sub_out_dir, exist_ok=True)
    out_path = os.path.join(sub_out_dir, f"regulome_rankings_{sample}_{ct}.csv")
    df_sub.to_csv(out_path, index=False)
    print(f"✅ Per-subtype REGULOME rankings saved to:\n  {out_path}")

# ================== 8) Jaccard similarity (per celltype + global) ==================

for ct in celltypes:
    ct_cells = cell_enh_by_celltype[ct]
    n_cells_ct = len(ct_cells)
    if n_cells_ct < 2:
        print(f"ℹ️  REGULOME: celltype {ct} has <2 cells; skipping Jaccard.")
        continue

    print(f"\n🔍 Computing REGULOME Jaccard matrix for celltype {ct} ({n_cells_ct} cells)...")
    cell_ids_ct, mat_ct = compute_jaccard_matrix(ct_cells)
    out_prefix_ct = os.path.join(SIM_DIR, f"Jaccard_REGULOME_{ct}_celltype")
    plot_jaccard_heatmap(cell_ids_ct, mat_ct,
                         title=f"Pairwise REGULOME Jaccard (%) - {ct}",
                         out_prefix=out_prefix_ct)

n_global_cells = len(global_cell_enh)
print(f"\nGlobal REGULOME Jaccard: {n_global_cells} cells with REGULOME data.")

if n_global_cells >= 2 and n_global_cells <= 250:
    print("🔍 Computing global REGULOME Jaccard matrix...")
    cell_ids_glob, mat_glob = compute_jaccard_matrix(global_cell_enh)
    out_prefix_glob = os.path.join(SIM_DIR, "Jaccard_REGULOME_GLOBAL")
    plot_jaccard_heatmap(cell_ids_glob, mat_glob,
                         title="Global Pairwise REGULOME Jaccard (%) - All Cells",
                         out_prefix=out_prefix_glob)
elif n_global_cells > 250:
    print("⚠️ More than 250 cells globally – skipping global REGULOME Jaccard to avoid huge matrices.")
else:
    print("ℹ️ <2 cells globally – skipping global REGULOME Jaccard.")

print("\n🎉 REGULOME analysis complete: differential regions, exclusives, rankings, and similarity.")
EOF