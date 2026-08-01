#!/bin/bash
#SBATCH --partition=guest,batch,unmc_gudalab,unmc_cbsb
#SBATCH --time=165:00:00
#SBATCH --mem=200G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=12_diff_PROMOTER_celltypes
#SBATCH --output=12_diff_PROMOTER_celltypes_%j.out
#SBATCH --error=12_diff_PROMOTER_celltypes_%j.err

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
SUMMARY_ROOT = "/work/avinash/user/CA/CA/Dopaminergic_sorted_scBAMs_PROMOTER_csv"
OUT_DIR = "/work/avinash/user/CA/CA/Dopaminergic_differential_PROMOTER_celltypes"
os.makedirs(OUT_DIR, exist_ok=True)

SIM_DIR = os.path.join(OUT_DIR, "Similarity")
os.makedirs(SIM_DIR, exist_ok=True)

print("SUMMARY_ROOT =", SUMMARY_ROOT)
print("OUT_DIR      =", OUT_DIR)
print("SIM_DIR      =", SIM_DIR)

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
})

# ================== Helpers ==================

def parse_sample_from_filename(base_name: str) -> str:
    """
    Your files look like:
      MM_429_D12NAC_<barcode>_BD__merged_overlaps_PROMOTER_summary.csv
    After removing extension, splitting by '_' gives:
      ['MM','429','D12NAC', ...]
    Sample is 'MM_429'
    """
    parts = base_name.split("_")
    if len(parts) >= 2 and parts[0] == "MM":
        return f"{parts[0]}_{parts[1]}"
    # fallback
    return parts[0] if parts else "UNKNOWN"

def discover_files_for_celltype_dir(ct_dir: str):
    """
    Supports BOTH layouts:
    1) New (your current):
       ct_dir/*.csv
    2) Old:
       ct_dir/<sample>/<ct>/summaries/*.csv
    """
    patterns = [
        os.path.join(ct_dir, "*_merged_overlaps_PROMOTER_summary.csv"),
        os.path.join(ct_dir, "*_overlaps_PROMOTER_summary.csv"),
    ]
    files = []
    for p in patterns:
        files.extend(glob.glob(p))

    # Backward compatible: if nothing found, try old nested structure under SUMMARY_ROOT
    # (This is optional but safe)
    if not files:
        patterns_old = [
            os.path.join(ct_dir, "*", "*", "summaries", "*_overlaps_PROMOTER_summary.csv"),
            os.path.join(ct_dir, "*", "*", "summaries", "*_merged_overlaps_PROMOTER_summary.csv"),
        ]
        for p in patterns_old:
            files.extend(glob.glob(p))

    return sorted(set(files))

# ================== 1) Traverse summaries ==================

celltype_counts = defaultdict(lambda: defaultdict(int))   # celltype -> enh_key -> #cells
celltype_ncells = defaultdict(int)                        # celltype -> #cells

sample_counts = defaultdict(lambda: defaultdict(int))     # sample   -> enh_key -> #cells
sample_ncells = defaultdict(int)                          # sample   -> #cells

ct_sample_counts = defaultdict(lambda: defaultdict(int))  # (sample, celltype) -> enh_key -> #cells
ct_sample_ncells = defaultdict(int)                       # (sample, celltype) -> #cells

all_PROMOTERs = set()

cell_enh_by_celltype = defaultdict(dict)   # ct -> {cell_id: set((chr, start, end))}
global_cell_enh = {}                       # cell_id -> set(...)

if not os.path.isdir(SUMMARY_ROOT):
    raise SystemExit(f"❌ SUMMARY_ROOT does not exist: {SUMMARY_ROOT}")

# In your real structure, the first level under SUMMARY_ROOT are celltypes
top_level = sorted([d for d in os.listdir(SUMMARY_ROOT) if os.path.isdir(os.path.join(SUMMARY_ROOT, d))])

print("\nTop-level directories under SUMMARY_ROOT (interpreted as celltypes):")
for d in top_level:
    print("  -", d)

for celltype in top_level:
    ct_dir = os.path.join(SUMMARY_ROOT, celltype)
    file_list = discover_files_for_celltype_dir(ct_dir)

    if not file_list:
        print(f"⚠️ No summary CSVs found for celltype dir: {ct_dir}")
        continue

    print(f"\nFound {len(file_list)} summary files for celltype {celltype}")

    for f in file_list:
        base_name = os.path.splitext(os.path.basename(f))[0]
        cell_id = base_name  # keep unique ID from filename
        sample = parse_sample_from_filename(base_name)

        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"Error reading {f}: {e}")
            continue

        # Expect these columns in summary CSV
        needed = {"Chromosome", "Start", "End"}
        if not needed.issubset(df.columns):
            print(f"Missing expected columns in {f}. Have={list(df.columns)}; need={sorted(needed)}")
            continue

        enh_set = set(
            (row["Chromosome"], int(row["Start"]), int(row["End"]))
            for _, row in df.iterrows()
        )

        if not enh_set:
            continue

        cell_enh_by_celltype[celltype][cell_id] = enh_set
        global_cell_enh[cell_id] = enh_set

        celltype_ncells[celltype] += 1
        sample_ncells[sample] += 1
        ct_sample_ncells[(sample, celltype)] += 1

        for key in enh_set:
            all_PROMOTERs.add(key)
            celltype_counts[celltype][key] += 1
            sample_counts[sample][key] += 1
            ct_sample_counts[(sample, celltype)][key] += 1

celltypes = sorted(celltype_counts.keys())
samples  = sorted(sample_counts.keys())

print("\nCelltypes discovered and #cells:")
for ct in celltypes:
    print(f"  {ct}: {celltype_ncells[ct]} cells")

print("\nSamples discovered and #cells:")
for s in samples:
    print(f"  {s}: {sample_ncells[s]} cells")

print(f"\nTotal unique PROMOTERs across all celltypes/samples: {len(all_PROMOTERs)}")

if not celltypes:
    raise SystemExit("❗️ No celltypes found. Nothing to analyze. (No matching *_summary.csv files were read.)")

# ================== 2) Build celltype-level master table ==================

rows = []
for (chrom, start, end) in all_PROMOTERs:
    row = {"Chromosome": chrom, "Start": start, "End": end}
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

# ================== 4) Exclusives per celltype ==================

def exclusive_celltype(row):
    present = [ct for ct in celltypes if row[f"Count_{ct}"] > 0]
    return present[0] if len(present) == 1 else ""

master_df["ExclusiveCelltype"] = master_df.apply(exclusive_celltype, axis=1)
master_df.sort_values(by="Diff_pct", ascending=False, inplace=True)

master_table_path = os.path.join(OUT_DIR, "differential_PROMOTERs_master_table_celltypes.csv")
master_df.to_csv(master_table_path, index=False)
print(f"\n✅ Master celltype-level table saved to:\n  {master_table_path}")

# ================== 5) Visualizations ==================

def PROMOTER_labels_from_df(df):
    return df["Chromosome"].astype(str) + ":" + df["Start"].astype(str) + "-" + df["End"].astype(str)

# Heatmaps
for N in [25, 50, 100]:
    if len(master_df) == 0:
        print("No PROMOTERs; skipping heatmaps.")
        break
    topN = min(N, len(master_df))
    top = master_df.head(topN).copy()
    heatmap_data = top[pct_cols].to_numpy()

    plt.figure(figsize=(max(10, len(celltypes) * 0.8), 0.4 * topN + 4))
    im = plt.imshow(heatmap_data, cmap="YlOrBr", aspect="auto")
    plt.colorbar(im, label="Percentage of Cells")

    plt.xticks(np.arange(len(celltypes)), celltypes, rotation=45, ha="right")
    plt.gca().xaxis.tick_top()

    labels = PROMOTER_labels_from_df(top)
    plt.yticks(np.arange(topN), labels, fontsize=6)

    for i in range(heatmap_data.shape[0]):
        for j in range(heatmap_data.shape[1]):
            plt.text(j, i, f"{heatmap_data[i, j]:.1f}", ha="center", va="center", fontsize=5)

    plt.title(f"Top {topN} Differential PROMOTERs (% Cells per Celltype)")
    plt.xlabel("Celltype")
    plt.ylabel("PROMOTER (Chromosome:Start-End)")
    plt.tight_layout()

    heatmap_png = os.path.join(OUT_DIR, f"heatmap_top{topN}_celltypes.png")
    heatmap_pdf = os.path.join(OUT_DIR, f"heatmap_top{topN}_celltypes.pdf")
    plt.savefig(heatmap_png, dpi=300, bbox_inches="tight")
    plt.savefig(heatmap_pdf, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Heatmap (top {topN}) saved to:\n  {heatmap_png}\n  {heatmap_pdf}")

# Volcano
for N in [25, 50, 100]:
    if len(master_df) == 0:
        print("No PROMOTERs; skipping volcano plots.")
        break
    topN = min(N, len(master_df))
    top = master_df.head(topN).copy()

    x = top["log2FC"].to_numpy()
    y = top["Diff_pct"].to_numpy()
    labels = PROMOTER_labels_from_df(top)

    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, s=30, alpha=0.8, edgecolor="k", linewidth=0.3)
    plt.axvline(0, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("log2( (Max_pct+1) / (Min_pct+1) )")
    plt.ylabel("Diff_pct (Max_pct - Min_pct)")
    plt.title(f"Volcano Plot (Top {topN} Differential PROMOTERs)")

    for xi, yi, lab in zip(x, y, labels):
        plt.text(xi, yi, lab, fontsize=5, ha="center", va="bottom")

    plt.tight_layout()
    volcano_png = os.path.join(OUT_DIR, f"volcano_top{topN}_celltypes.png")
    volcano_pdf = os.path.join(OUT_DIR, f"volcano_top{topN}_celltypes.pdf")
    plt.savefig(volcano_png, dpi=300, bbox_inches="tight")
    plt.savefig(volcano_pdf, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Volcano (top {topN}) saved to:\n  {volcano_png}\n  {volcano_pdf}")

# Diff_pct distribution
if len(master_df) > 0:
    plt.figure(figsize=(8, 5))
    plt.hist(master_df["Diff_pct"], bins=50, alpha=0.8)
    plt.xlabel("Diff_pct (Max_pct - Min_pct)")
    plt.ylabel("Number of PROMOTERs")
    plt.title("Distribution of Differential Percentages Across Celltypes")
    plt.tight_layout()

    hist_png = os.path.join(OUT_DIR, "diff_pct_distribution.png")
    hist_pdf = os.path.join(OUT_DIR, "diff_pct_distribution.pdf")
    plt.savefig(hist_png, dpi=300, bbox_inches="tight")
    plt.savefig(hist_pdf, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Diff_pct distribution saved to:\n  {hist_png}\n  {hist_pdf}")

# ================== 6) Exclusive PROMOTERs per celltype (CSV + barplot) ==================

exclusive_counts = {}
exclusive_dir = os.path.join(OUT_DIR, "exclusive_per_celltype")
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
        excl_path = os.path.join(exclusive_dir, f"exclusive_PROMOTERs_{ct}.csv")
        sub[cols_to_save].to_csv(excl_path, index=False)
        print(f"✅ Exclusive PROMOTERs for {ct} saved to:\n  {excl_path}")

plt.figure(figsize=(max(8, len(celltypes) * 0.7), 6))
x_labels = list(exclusive_counts.keys())
y_vals   = [exclusive_counts[ct] for ct in x_labels]

plt.bar(x_labels, y_vals)
plt.ylabel("Number of Exclusive PROMOTERs")
plt.title("Exclusive PROMOTERs per Celltype")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

bar_excl_png = os.path.join(OUT_DIR, "exclusive_PROMOTERs_per_celltype.png")
bar_excl_pdf = os.path.join(OUT_DIR, "exclusive_PROMOTERs_per_celltype.pdf")
plt.savefig(bar_excl_png, dpi=300, bbox_inches="tight")
plt.savefig(bar_excl_pdf, dpi=300, bbox_inches="tight")
plt.close()
print(f"✅ Barplot of exclusive PROMOTERs per celltype saved to:\n  {bar_excl_png}\n  {bar_excl_pdf}")

# ================== 7) Per-(sample,celltype) rankings ==================

subtype_dir = os.path.join(OUT_DIR, "per_sample_celltype_rankings")
os.makedirs(subtype_dir, exist_ok=True)

for (sample, ct), enh_dict in ct_sample_counts.items():
    n_cells = ct_sample_ncells[(sample, ct)]
    if n_cells == 0:
        continue

    rows_sub = []
    for (chrom, start, end), count in enh_dict.items():
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

    df_sub = pd.DataFrame(rows_sub).sort_values(by="Pct_in_subtype", ascending=False)
    sub_out_dir = os.path.join(subtype_dir, sample, ct)
    os.makedirs(sub_out_dir, exist_ok=True)
    out_path = os.path.join(sub_out_dir, f"rankings_{sample}_{ct}.csv")
    df_sub.to_csv(out_path, index=False)
    print(f"✅ Per-subtype rankings saved to:\n  {out_path}")

# ================== 8) Jaccard similarity ==================

def compute_jaccard_matrix(cell_enh_dict):
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
                mat[i, j] = mat[j, i] = (inter / union * 100.0) if union > 0 else 0.0
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
        df, cmap="viridis", square=True, vmin=0, vmax=max_non_diag,
        mask=mask, cbar_kws={"label": "Jaccard overlap (%)"},
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

for ct in celltypes:
    ct_cells = cell_enh_by_celltype[ct]
    n_cells_ct = len(ct_cells)
    if n_cells_ct < 2:
        print(f"ℹ️  Celltype {ct} has <2 cells with PROMOTER data; skipping Jaccard.")
        continue
    print(f"\n🔍 Computing Jaccard matrix for celltype {ct} ({n_cells_ct} cells)...")
    cell_ids_ct, mat_ct = compute_jaccard_matrix(ct_cells)
    out_prefix_ct = os.path.join(SIM_DIR, f"Jaccard_{ct}_celltype")
    plot_jaccard_heatmap(cell_ids_ct, mat_ct,
                         title=f"Pairwise PROMOTER Jaccard (%) - {ct}",
                         out_prefix=out_prefix_ct)

n_global_cells = len(global_cell_enh)
print(f"\nGlobal Jaccard: {n_global_cells} cells with PROMOTER data.")

if 2 <= n_global_cells <= 250:
    print("🔍 Computing global Jaccard matrix...")
    cell_ids_glob, mat_glob = compute_jaccard_matrix(global_cell_enh)
    out_prefix_glob = os.path.join(SIM_DIR, "Jaccard_GLOBAL")
    plot_jaccard_heatmap(cell_ids_glob, mat_glob,
                         title="Global Pairwise PROMOTER Jaccard (%) - All Cells",
                         out_prefix=out_prefix_glob)
elif n_global_cells > 250:
    print("⚠️ More than 250 cells globally – skipping global Jaccard to avoid huge matrices.")
else:
    print("ℹ️ <2 cells globally – skipping global Jaccard.")

print("\n🎉 Step 12 complete: differential PROMOTERs + celltype exclusives + per-subtype rankings + Jaccard similarity.")
EOF
