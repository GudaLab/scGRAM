#!/usr/bin/env python3
"""
Cross-group regulatory divergence analysis — BRAIN CELL TYPES ONLY.

Groups: Dopaminergic, MSN, FOXP2, PVALB, VIP, CNGA

Reads each group's regulatory_only_master_table.csv and pairwise CSVs,
then produces 9 cross-group analysis panels (C1–C9):
  C1  Spearman divergence clustermap with dendrogram
  C2  PCA of regulatory fingerprints + scree
  C3  t-SNE (if >5 subtypes)
  C4  Region-type composition shift (stacked bar, dendrogram order)
  C5  Top-region cross-group heatmap
  C6  Cross-group concordance (shared significant regions)
  C7  Regulatory complexity index (Shannon diversity bubble chart)
  C8  Conserved vs divergent regions
  C9  Intra- vs inter-group divergence

Usage:
  nohup $HOME/.conda/envs/snapatac2_env/bin/python \
    /path/to/data/14_cross_group_brain.py \
    >> /path/to/data/14_cross_group_brain.out 2>&1 &
"""

import os, sys, glob, time, warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr, entropy
from sklearn.decomposition import PCA
try:
    from sklearn.manifold import TSNE
    HAS_TSNE = True
except ImportError:
    HAS_TSNE = False

warnings.filterwarnings("ignore", category=FutureWarning)

# ================================================================
#  CONFIG
# ================================================================

BASE_DIR  = "/path/to/data"
CROSS_DIR = os.path.join(BASE_DIR, "cross_group_brain")
CROSS_CKPT = os.path.join(CROSS_DIR, ".ckpt_done")

BRAIN_GROUPS = [
    # --- Glutamatergic / excitatory (new) ---
    "ITL23", "ITL34", "ITL4", "ITL45", "L6B", "CHO", "AMY",
    # --- GABAergic / inhibitory ---
    "Dopaminergic", "MSN", "FOXP2", "PVALB", "VIP", "CNGA",
    "BFEXA", "CNMIX", "BNGA", "PV_ChCs",
    # --- Non-neuronal (glia / astrocyte) ---
    "ASCT", "MGC_2", "MGC_1",
    # --- newest single-subtype GABAergic additions (2026-07) ---
    "MDGA", "ICGA_1", "ICGA_2", "SEPGA", "SIGA", "CTXMIX",
]

TOP_PER_GROUP = 500  # top variable regions per group for cross-group matrix

SIMPLE_TYPE_COLORS = {
    "enhancer":         "#3498db",
    "silencer":         "#2ecc71",
    "promoter":         "#f39c12",
    "gene_body":        "#9b59b6",
    "uncharacterized":  "#e74c3c",
    "insulator":        "#1abc9c",
    "other_regulatory": "#95a5a6",
}
REG_CATS = ["enhancer","silencer","promoter","uncharacterized","insulator","other_regulatory"]

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.size":10,"axes.labelsize":12,"axes.titlesize":14,
    "xtick.labelsize":9,"ytick.labelsize":9,"legend.fontsize":9,
    "figure.dpi":150,"savefig.dpi":300,"savefig.bbox":"tight",
    "pdf.fonttype":42,"ps.fonttype":42,
})

# ================================================================
#  HELPERS
# ================================================================

def rlabel(c,s,e): return f"{c}:{s}-{e}"

def _simplify_type(raw_type):
    rt = raw_type.lower()
    if "uncharacterized" in rt: return "uncharacterized"
    if "insulator" in rt or "enhancer_blocking" in rt: return "insulator"
    if "enhancer" in rt: return "enhancer"
    if "silencer" in rt: return "silencer"
    if "promoter" in rt: return "promoter"
    if "gene_body" in rt: return "gene_body"
    return "other_regulatory"

def _save(fig, stem):
    for ext in ("png","pdf"):
        try:
            fig.savefig(f"{stem}.{ext}", dpi=300, bbox_inches="tight")
        except ValueError as e:
            if "too large" in str(e):
                try: fig.savefig(f"{stem}.{ext}", dpi=100, bbox_inches="tight")
                except: pass
            else: raise
    plt.close(fig)

def _smart_legend(ax, title=None, fontsize=7, title_fontsize=8,
                  max_per_col=6, outside=True, **kwargs):
    handles, labels = ax.get_legend_handles_labels()
    n = len(handles)
    if n == 0: return
    ncol = max(1, (n + max_per_col - 1) // max_per_col)
    if outside:
        ax.legend(handles, labels, title=title, fontsize=fontsize,
                  title_fontsize=title_fontsize, ncol=ncol,
                  loc="upper center", bbox_to_anchor=(0.5, -0.08),
                  framealpha=0.9, edgecolor="#bdc3c7",
                  columnspacing=1.0, handletextpad=0.4, borderpad=0.4, **kwargs)
    else:
        ax.legend(handles, labels, title=title, fontsize=fontsize,
                  title_fontsize=title_fontsize, ncol=ncol,
                  loc="best", framealpha=0.9, edgecolor="#bdc3c7",
                  columnspacing=1.0, handletextpad=0.4, borderpad=0.4, **kwargs)

def _label_group_centroids(ax, coords, point_groups, group_order, palette):
    """One bold, non-overlapping label per group at its centroid.
    Simple axis-aware repulsion avoids label collisions (no adjustText dep).
    Per-subtype identity is preserved in the figure's companion CSV."""
    cents, names = [], []
    for gn in group_order:
        idx = [i for i, g in enumerate(point_groups) if g == gn]
        if not idx:
            continue
        names.append(gn)
        cents.append([float(np.mean(coords[idx, 0])), float(np.mean(coords[idx, 1]))])
    if not cents:
        return
    pts = np.array(cents, dtype=float)
    xr = (coords[:, 0].max() - coords[:, 0].min()) or 1.0
    yr = (coords[:, 1].max() - coords[:, 1].min()) or 1.0
    mindx, mindy = 0.085 * xr, 0.055 * yr
    for _ in range(400):
        moved = False
        for a in range(len(pts)):
            for b in range(a + 1, len(pts)):
                dx, dy = pts[b, 0] - pts[a, 0], pts[b, 1] - pts[a, 1]
                if abs(dx) < mindx and abs(dy) < mindy:
                    push = (mindx - abs(dx)) / 2 + 1e-9
                    sgn = 1.0 if dx >= 0 else -1.0
                    pts[a, 0] -= sgn * push
                    pts[b, 0] += sgn * push
                    moved = True
        if not moved:
            break
    for n, (x, y) in zip(names, pts):
        ax.annotate(n, (x, y), fontsize=8, fontweight="bold", ha="center", va="center",
                    color="#1a1a1a", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white",
                              ec=palette.get(n, "#888"), lw=1.0, alpha=0.9))

def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

# ================================================================
#  MAIN
# ================================================================

if os.path.exists(CROSS_CKPT):
    log(f"Cross-group brain already done: {CROSS_CKPT}")
    sys.exit(0)

os.makedirs(CROSS_DIR, exist_ok=True)
t0 = time.time()
log(f"Cross-group BRAIN analysis: {BRAIN_GROUPS}")

# ============================================================
# A. DATA COLLECTION
# ============================================================
log("A: loading regulatory master tables")

subtype_profiles  = {}
subtype_group     = {}
subtype_ncells    = {}
subtype_type_comp = {}
all_cross_regions = set()
group_sig_regions = {}

available_groups = []
for pfx in BRAIN_GROUPS:
    out_dir = os.path.join(BASE_DIR, f"{pfx}_differential_unified_celltypes")
    reg_master = os.path.join(out_dir, "regulatory_only", "regulatory_only_master_table.csv")
    if not os.path.exists(reg_master):
        log(f"  SKIP {pfx}: no regulatory master table at {reg_master}")
        continue

    available_groups.append(pfx)
    df = pd.read_csv(reg_master, low_memory=False, dtype={"Chromosome":str})
    pct_c = [c for c in df.columns if c.startswith("Pct_")]
    cts = [c.replace("Pct_","") for c in pct_c]

    top_var = df.nlargest(TOP_PER_GROUP, "Diff_pct")
    df["_stype"] = df["Region_Type"].apply(_simplify_type)

    # Collect significant regions from pairwise CSVs
    sig_regs = set()
    pair_dir = os.path.join(out_dir, "regulatory_only", "pairwise")
    if os.path.isdir(pair_dir):
        for pf in glob.glob(os.path.join(pair_dir, "*.csv")):
            try:
                pw = pd.read_csv(pf, usecols=["Chromosome","Start","End","FDR"],
                                 low_memory=False, dtype={"Chromosome":str})
                for _,r in pw[pw["FDR"]<0.05].iterrows():
                    sig_regs.add((r["Chromosome"],int(r["Start"]),int(r["End"])))
            except: pass
    group_sig_regions[pfx] = sig_regs

    for ct in cts:
        label = f"{pfx}__{ct}"
        profile = {}
        for _,row in top_var.iterrows():
            reg = (row["Chromosome"], int(row["Start"]), int(row["End"]))
            profile[reg] = row[f"Pct_{ct}"]
            all_cross_regions.add(reg)
        subtype_profiles[label] = profile
        subtype_group[label] = pfx

        for _,row in df.iterrows():
            cnt_val = row[f"Count_{ct}"]
            pct_val = row[f"Pct_{ct}"]
            if cnt_val > 0 and pct_val > 0:
                subtype_ncells[label] = int(round(cnt_val / (pct_val/100)))
                break
        else:
            subtype_ncells[label] = 0

        type_comp = {}
        for st_cat in REG_CATS:
            active = (df["_stype"]==st_cat) & (df[f"Count_{ct}"]>0)
            type_comp[st_cat] = int(active.sum())
        subtype_type_comp[label] = type_comp

    log(f"  {pfx}: {len(cts)} subtypes, {len(sig_regs):,} sig regions")

subtypes_all = sorted(subtype_profiles.keys())
regions_all  = sorted(all_cross_regions)
n_st = len(subtypes_all)
n_reg = len(regions_all)
log(f"A DONE: {n_st} subtypes, {n_reg:,} regions from {len(available_groups)} groups")

if n_st < 2:
    log("Need >= 2 subtypes for cross-group analysis. Exiting.")
    sys.exit(1)

# ============================================================
# B. PRESENCE MATRIX + DISTANCE
# ============================================================
log("B: building presence matrix and Spearman distance")

mat = np.zeros((n_st, n_reg), dtype=np.float32)
for i, st in enumerate(subtypes_all):
    prof = subtype_profiles[st]
    for j, reg in enumerate(regions_all):
        mat[i,j] = prof.get(reg, 0.0)

dist_mat = np.zeros((n_st, n_st))
for i in range(n_st):
    for j in range(i+1, n_st):
        rho, _ = spearmanr(mat[i], mat[j])
        if np.isnan(rho): rho = 0.0
        d = 1.0 - rho
        dist_mat[i,j] = dist_mat[j,i] = d

dist_df = pd.DataFrame(dist_mat, index=subtypes_all, columns=subtypes_all)
dist_df.to_csv(os.path.join(CROSS_DIR, "spearman_divergence_matrix.csv"))

condensed = squareform(dist_mat)
condensed = np.clip(condensed, 0, None)
Z = linkage(condensed, method="average")
ordered_idx = leaves_list(Z)
ordered_labels = [subtypes_all[k] for k in ordered_idx]

unique_groups = sorted(set(subtype_group.values()))
gpal = dict(zip(unique_groups, sns.color_palette("husl", len(unique_groups)).as_hex()))
row_colors = pd.Series([gpal[subtype_group[st]] for st in subtypes_all],
                       index=subtypes_all, name="Group")
log("B DONE")

# ============================================================
# C1. CLUSTERMAP
# ============================================================
log("C1: clustermap")
fs = (max(16, n_st*0.28), max(14, n_st*0.25))
g = sns.clustermap(
    dist_df, row_linkage=Z, col_linkage=Z,
    cmap="RdYlBu_r", vmin=0, vmax=np.percentile(dist_mat[dist_mat>0],95),
    figsize=fs, row_colors=row_colors, col_colors=row_colors,
    linewidths=0, xticklabels=True, yticklabels=True,
    cbar_kws={"label":"Spearman Divergence (1 - \u03c1)", "shrink":0.4},
    dendrogram_ratio=(0.12, 0.12),
)
g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(),
                              fontsize=max(3,min(7,400//n_st)), rotation=90)
g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(),
                              fontsize=max(3,min(7,400//n_st)), rotation=0)
g.fig.suptitle("Cross-Group Regulatory Divergence — Brain Cell Types\n"
               f"Top {TOP_PER_GROUP} variable regulatory regions per group, {n_st} subtypes",
               fontsize=14, fontweight="bold", y=1.02)
handles_g = [mpatches.Patch(color=gpal[gn], label=gn) for gn in unique_groups]
ncol_g = max(1, (len(handles_g)+5)//6)
g.fig.legend(handles=handles_g, title="Sample Group", fontsize=7, title_fontsize=8,
             ncol=ncol_g, loc="lower center", bbox_to_anchor=(0.5,-0.03),
             framealpha=0.9, columnspacing=1.2)
for ext in ("png","pdf"):
    g.savefig(os.path.join(CROSS_DIR, f"C1_divergence_clustermap.{ext}"),
              dpi=300, bbox_inches="tight")
plt.close()
log("C1 saved")

# ============================================================
# C2. PCA
# ============================================================
log("C2: PCA")
pca = PCA(n_components=min(10, n_st-1, n_reg))
pca_coords = pca.fit_transform(mat)

fig, ax = plt.subplots(figsize=(10, 8))
for gn in unique_groups:
    idx = [i for i,st in enumerate(subtypes_all) if subtype_group[st]==gn]
    ax.scatter(pca_coords[idx,0], pca_coords[idx,1],
               c=gpal[gn], s=60, alpha=0.8, edgecolors="k", linewidths=0.4,
               label=gn, zorder=2)
# Group-centroid labels (non-overlapping); per-subtype coords in companion CSV
_label_group_centroids(ax, pca_coords,
                       [subtype_group[st] for st in subtypes_all], unique_groups, gpal)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=12)
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=12)
ax.set_title("PCA of Regulatory Landscape — Brain Cell Types\n"
             f"({n_reg:,} regions, {n_st} subtypes across {len(unique_groups)} groups)",
             fontsize=13, fontweight="bold")
_smart_legend(ax, title="Sample Group", fontsize=7, title_fontsize=8)
ax.grid(True, alpha=0.15); fig.subplots_adjust(bottom=0.16)
_save(fig, os.path.join(CROSS_DIR, "C2_PCA_regulatory_fingerprints"))

# Scree
fig, ax = plt.subplots(figsize=(7, 4))
nc = len(pca.explained_variance_ratio_)
ax.bar(range(1,nc+1), pca.explained_variance_ratio_*100, color="#3498db", edgecolor="white")
ax.plot(range(1,nc+1), np.cumsum(pca.explained_variance_ratio_)*100, "o-", color="#e74c3c", markersize=5)
ax.set_xlabel("Principal Component", fontsize=12)
ax.set_ylabel("Variance Explained (%)", fontsize=12)
ax.set_title("PCA Scree Plot — Brain Cell Types", fontsize=13, fontweight="bold")
ax.legend(["Cumulative", "Per-component"], fontsize=9, framealpha=0.85)
plt.tight_layout()
_save(fig, os.path.join(CROSS_DIR, "C2_PCA_scree"))
log("C2 saved")

# ============================================================
# C3. t-SNE
# ============================================================
if HAS_TSNE and n_st > 5:
    log("C3: t-SNE")
    perp = min(30, max(2, n_st//3))
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42, metric="precomputed", init="random")
    tsne_coords = tsne.fit_transform(dist_mat)

    fig, ax = plt.subplots(figsize=(10, 8))
    for gn in unique_groups:
        idx = [i for i,st in enumerate(subtypes_all) if subtype_group[st]==gn]
        ax.scatter(tsne_coords[idx,0], tsne_coords[idx,1],
                   c=gpal[gn], s=60, alpha=0.8, edgecolors="k", linewidths=0.4,
                   label=gn, zorder=2)
    _label_group_centroids(ax, tsne_coords,
                           [subtype_group[st] for st in subtypes_all], unique_groups, gpal)
    ax.set_xlabel("t-SNE 1", fontsize=12); ax.set_ylabel("t-SNE 2", fontsize=12)
    ax.set_title(f"t-SNE of Regulatory Divergence — Brain (perplexity={perp})",
                 fontsize=13, fontweight="bold")
    _smart_legend(ax, title="Sample Group", fontsize=7, title_fontsize=8)
    ax.grid(True, alpha=0.15); fig.subplots_adjust(bottom=0.16)
    _save(fig, os.path.join(CROSS_DIR, "C3_tSNE_regulatory_divergence"))
    log("C3 saved")

# ============================================================
# C4. Region-type composition shift
# ============================================================
log("C4: region-type composition shift")
fig, ax = plt.subplots(figsize=(max(14, n_st*0.3), 7))
bottoms = np.zeros(n_st)
x_pos = np.arange(n_st)
for st_cat in REG_CATS:
    vals = np.array([subtype_type_comp[ordered_labels[k]].get(st_cat,0)
                     for k in range(n_st)], dtype=float)
    totals = np.array([sum(subtype_type_comp[ordered_labels[k]].values())
                       for k in range(n_st)], dtype=float)
    pcts = np.where(totals>0, vals/totals*100, 0.0)
    ax.bar(x_pos, pcts, bottom=bottoms, label=st_cat,
           color=SIMPLE_TYPE_COLORS.get(st_cat,"#95a5a6"), edgecolor="white", linewidth=0.2)
    bottoms += pcts
ax.set_xticks(x_pos)
ax.set_xticklabels([ordered_labels[k].replace("__","\n") for k in range(n_st)],
                   fontsize=max(3,min(6,300//n_st)), rotation=90)
ax.set_ylabel("Region Type Proportion (%)", fontsize=12)
ax.set_title("Regulatory Composition Shift — Brain Cell Types\n"
             "(ordered by divergence dendrogram)", fontsize=13, fontweight="bold")
_smart_legend(ax, title="Region Type", fontsize=8, title_fontsize=9)
fig.subplots_adjust(bottom=0.25)
_save(fig, os.path.join(CROSS_DIR, "C4_region_type_composition_shift"))
log("C4 saved")

# ============================================================
# C5. Top-region cross-group heatmap
# ============================================================
log("C5: top region cross-group heatmap")
top_cross_regs = []; top_cross_labels = []
for pfx in sorted(available_groups):
    reg_master = os.path.join(BASE_DIR, f"{pfx}_differential_unified_celltypes",
                              "regulatory_only","regulatory_only_master_table.csv")
    if not os.path.exists(reg_master): continue
    df = pd.read_csv(reg_master, low_memory=False, dtype={"Chromosome":str})
    top10 = df.nlargest(10, "Diff_pct")
    for _,row in top10.iterrows():
        reg = (row["Chromosome"],int(row["Start"]),int(row["End"]))
        if reg not in top_cross_regs:
            top_cross_regs.append(reg)
            st = _simplify_type(row["Region_Type"])
            top_cross_labels.append(f"{rlabel(*reg)} [{st}] ({pfx})")

if len(top_cross_regs) > 0:
    n_tr = len(top_cross_regs)
    rank_mat = np.zeros((n_tr, n_st))
    for i, reg in enumerate(top_cross_regs):
        for j, st in enumerate(subtypes_all):
            rank_mat[i,j] = subtype_profiles[st].get(reg, 0.0)
    rank_mat_ord = rank_mat[:, ordered_idx]

    fig_h = max(10, n_tr*0.35); fig_w = max(16, n_st*0.28)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(rank_mat_ord, cmap="YlOrRd", aspect="auto", vmin=0)
    cb = plt.colorbar(im, ax=ax, shrink=0.35, pad=0.01)
    cb.set_label("Presence (%)", fontsize=10)
    ax.set_xticks(np.arange(n_st))
    ax.set_xticklabels(ordered_labels, rotation=90, fontsize=max(3,min(6,300//n_st)))
    ax.xaxis.tick_top()
    ax.set_yticks(np.arange(n_tr))
    ax.set_yticklabels(top_cross_labels, fontsize=6)
    ax.set_title("Top 10 Regulatory Regions per Group — Brain Cell Types\n"
                 "(columns ordered by divergence dendrogram)", fontsize=13, fontweight="bold", pad=20)
    plt.tight_layout()
    _save(fig, os.path.join(CROSS_DIR, "C5_top_regions_cross_group_heatmap"))
    log("C5 saved")

# ============================================================
# C5b. ENHANCED landscape heatmap with dendrogram + significance
# ============================================================
log("C5b: enhanced landscape heatmap with dendrogram")

# --- collect per-region metadata ---
from scipy.cluster.hierarchy import dendrogram as _dendro

top_source_groups = []   # which group contributed each region
top_types = []           # simplified region type
top_diff_pct = []        # max - min across all subtypes
top_is_sig = []          # significant in any group?
top_min_fdr = []         # min FDR across all pairwise CSVs for this region

for i, reg in enumerate(top_cross_regs):
    # source group from label (last parenthetical)
    src = top_cross_labels[i].rsplit("(", 1)[-1].rstrip(")")
    top_source_groups.append(src)

    # region type from label
    rt = top_cross_labels[i].split("[")[1].split("]")[0] if "[" in top_cross_labels[i] else "other"
    top_types.append(rt)

    # Diff_pct from presence matrix
    pcts = rank_mat[i, :]
    top_diff_pct.append(pcts.max() - pcts.min())

    # significance check
    is_sig = any(reg in group_sig_regions.get(g, set()) for g in available_groups)
    top_is_sig.append(is_sig)

    # min FDR: targeted lookup from source group pairwise CSVs
    min_fdr = 1.0
    pair_dir = os.path.join(BASE_DIR, f"{src}_differential_unified_celltypes",
                            "regulatory_only", "pairwise")
    if os.path.isdir(pair_dir):
        c, s, e = reg
        for pf in glob.glob(os.path.join(pair_dir, "*.csv")):
            try:
                pw = pd.read_csv(pf, usecols=["Chromosome","Start","End","FDR"],
                                 low_memory=False, dtype={"Chromosome":str})
                match = pw[(pw["Chromosome"]==c) & (pw["Start"]==s) & (pw["End"]==e)]
                if not match.empty:
                    min_fdr = min(min_fdr, match.iloc[0]["FDR"])
            except:
                pass
    top_min_fdr.append(min_fdr)

# Sort by Diff_pct descending (divergence ranking)
sort_idx = np.argsort(top_diff_pct)[::-1]
top_cross_regs_sorted = [top_cross_regs[i] for i in sort_idx]
top_cross_labels_sorted = [top_cross_labels[i] for i in sort_idx]
top_types_sorted = [top_types[i] for i in sort_idx]
top_source_sorted = [top_source_groups[i] for i in sort_idx]
top_diff_sorted = [top_diff_pct[i] for i in sort_idx]
top_fdr_sorted = [top_min_fdr[i] for i in sort_idx]
top_sig_sorted = [top_is_sig[i] for i in sort_idx]
rank_mat_sorted = rank_mat[sort_idx, :]

n_tr_s = len(top_cross_regs_sorted)

# Build formatted row labels with significance
row_labels_enhanced = []
for i in range(n_tr_s):
    c, s, e = top_cross_regs_sorted[i]
    fdr = top_fdr_sorted[i]
    stars = "***" if fdr < 0.001 else "**" if fdr < 0.01 else "*" if fdr < 0.05 else ""
    row_labels_enhanced.append(
        f"{c}:{s}-{e}  [{top_types_sorted[i]}]  "
        f"Diff={top_diff_sorted[i]:.1f}%  {stars}"
    )

# Build DataFrame for clustermap (un-ordered columns — clustermap will reorder via Z)
heat_df = pd.DataFrame(rank_mat_sorted, index=row_labels_enhanced, columns=subtypes_all)

# Column colors: one color per subtype within each group
# Use a different shade per subtype in the same group
group_subtypes = {}
for st in subtypes_all:
    g = subtype_group[st]
    group_subtypes.setdefault(g, []).append(st)

# Generate per-subtype colors: shades of the group hue
import colorsys
def _shade(hex_color, factor):
    """Lighten/darken a hex color. factor>1 = lighter, <1 = darker."""
    r = int(hex_color[1:3], 16)/255
    g_c = int(hex_color[3:5], 16)/255
    b = int(hex_color[5:7], 16)/255
    h, l, s = colorsys.rgb_to_hls(r, g_c, b)
    l = max(0.15, min(0.85, l * factor))
    r2, g2, b2 = colorsys.hls_to_rgb(h, l, s)
    return f"#{int(r2*255):02x}{int(g2*255):02x}{int(b2*255):02x}"

subtype_colors = {}
for g, sts in group_subtypes.items():
    base = gpal[g]
    n_sts = len(sts)
    if n_sts == 1:
        subtype_colors[sts[0]] = base
    else:
        for k, st in enumerate(sorted(sts)):
            factor = 0.7 + 0.6 * k / (n_sts - 1)  # range 0.7 to 1.3
            subtype_colors[st] = _shade(base, factor)

# Prepare color annotations for clustermap
col_colors_subtype = pd.Series(
    [subtype_colors[st] for st in subtypes_all],
    index=subtypes_all, name="Subtype"
)
col_colors_group = pd.Series(
    [gpal[subtype_group[st]] for st in subtypes_all],
    index=subtypes_all, name="Group"
)

row_colors_type = pd.Series(
    [SIMPLE_TYPE_COLORS.get(t, "#95a5a6") for t in top_types_sorted],
    index=row_labels_enhanced, name="Region Type"
)
row_colors_src = pd.Series(
    [gpal.get(s, "#95a5a6") for s in top_source_sorted],
    index=row_labels_enhanced, name="Source Group"
)

# --- Generate clustermap (compressed, with dendrogram + % labels) ---

# Compressed dimensions: half the block size
fig_w = max(14, n_st * 0.50)    # narrow columns
fig_h = max(12, n_tr_s * 0.18 + 5)  # short rows

g5 = sns.clustermap(
    heat_df,
    col_linkage=Z,
    row_cluster=False,
    col_colors=[col_colors_group, col_colors_subtype],
    row_colors=[row_colors_type, row_colors_src],
    cmap="YlOrRd",
    vmin=0,
    figsize=(fig_w, fig_h),
    xticklabels=True,
    yticklabels=True,
    linewidths=0.3,
    linecolor="white",
    dendrogram_ratio=(0.22, 0),   # large top dendrogram (22% of height)
    cbar_pos=(0.02, 0.78, 0.015, 0.15),
    cbar_kws={"label": "Presence (%)", "orientation": "vertical"},
)

# --- Overlay % text on every cell ---
# Get the reordered data from clustermap (columns reordered by dendrogram)
heat_ax = g5.ax_heatmap
reordered_data = g5.data2d.values   # this is the data as displayed

max_col_idx = np.argmax(reordered_data, axis=1)   # column of max value per row

for i in range(reordered_data.shape[0]):
    for j in range(reordered_data.shape[1]):
        val = reordered_data[i, j]
        if val < 0.5:
            continue   # skip near-zero cells to avoid clutter

        is_max = (j == max_col_idx[i])
        fdr_val = top_fdr_sorted[i]
        stars = ""
        if is_max:
            if fdr_val < 0.001:   stars = "***"
            elif fdr_val < 0.01:  stars = "**"
            elif fdr_val < 0.05:  stars = "*"

        txt = f"{val:.0f}"
        if stars:
            txt = f"{val:.0f}{stars}"

        # Dark text on light cells, white on dark
        text_color = "white" if val > 40 else "black"
        fw = "bold" if is_max else "normal"
        fs = 5.5 if is_max else 4.5

        heat_ax.text(j + 0.5, i + 0.5, txt, ha="center", va="center",
                     fontsize=fs, fontweight=fw, color=text_color)

# --- Format column tick labels: short subtype names, bold ---
short_xticks = []
for lbl in heat_ax.get_xticklabels():
    txt = lbl.get_text()
    short = txt.split("__")[1] if "__" in txt else txt
    short_xticks.append(short)
heat_ax.set_xticklabels(short_xticks, fontsize=9, fontweight="bold",
                         rotation=55, ha="right")

# --- Format row labels: bold the significant ones, increase font ---
new_ylabels = []
for lbl in heat_ax.get_yticklabels():
    new_ylabels.append(lbl.get_text())
heat_ax.set_yticklabels(new_ylabels, fontsize=7)

# Bold the y-tick labels that have significance stars
for lbl in heat_ax.get_yticklabels():
    txt = lbl.get_text()
    if "***" in txt:
        lbl.set_fontweight("bold")
        lbl.set_color("#b71c1c")
    elif "**" in txt:
        lbl.set_fontweight("bold")
        lbl.set_color("#c62828")
    elif "*" in txt and "***" not in txt and "**" not in txt:
        lbl.set_fontweight("bold")
        lbl.set_color("#d32f2f")

# --- Style the dendrogram ---
dend_ax = g5.ax_col_dendrogram
dend_ax.set_title("Subtype Relationship Dendrogram (Spearman Distance)",
                   fontsize=10, fontweight="bold", pad=4)
# Make dendrogram lines thicker and darker
for line in dend_ax.collections:
    line.set_linewidth(1.5)
for line in dend_ax.lines:
    line.set_linewidth(1.5)
    line.set_color("#2c3e50")

# --- Title ---
g5.fig.suptitle(
    "Cross-Group Regulatory Divergence: Top Enriched Regions\n"
    "Ranked by Differential Presence Across Brain Cell Subtypes\n"
    f"{n_tr_s} regions | {n_st} subtypes | {len(available_groups)} groups | "
    f"Fisher's exact + BH FDR",
    fontsize=14, fontweight="bold", y=1.06
)

# --- Significance threshold box ---
g5.fig.text(0.98, 1.04,
    "Significance:  * FDR < 0.05   ** FDR < 0.01   *** FDR < 0.001",
    fontsize=9, fontstyle="italic", fontweight="bold", ha="right",
    transform=g5.fig.transFigure,
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff3e0", edgecolor="#e65100", alpha=0.9))

# --- Legend: group colors + region type colors, multi-column ---
legend_handles = []
legend_handles.append(mpatches.Patch(color="none", label="$\\bf{Sample\\ Groups:}$"))
for g_name in sorted(gpal.keys()):
    legend_handles.append(mpatches.Patch(color=gpal[g_name], label=g_name))
legend_handles.append(mpatches.Patch(color="none", label=""))
legend_handles.append(mpatches.Patch(color="none", label="$\\bf{Region\\ Types:}$"))
used_types = sorted(set(top_types_sorted))
for t in used_types:
    legend_handles.append(mpatches.Patch(color=SIMPLE_TYPE_COLORS.get(t, "#95a5a6"), label=t))

n_items = len(legend_handles)
ncol_legend = max(4, (n_items + 2) // 3)
g5.fig.legend(
    handles=legend_handles, fontsize=9, title_fontsize=10,
    ncol=ncol_legend, loc="lower center", bbox_to_anchor=(0.52, -0.07),
    framealpha=0.9, edgecolor="#bdc3c7", columnspacing=3.0,
    handletextpad=0.5, borderpad=0.5
)

# --- Save ---
for ext in ("png", "pdf"):
    try:
        g5.savefig(os.path.join(CROSS_DIR, f"C5b_enhanced_cross_group_heatmap.{ext}"),
                   dpi=300, bbox_inches="tight")
    except ValueError:
        g5.savefig(os.path.join(CROSS_DIR, f"C5b_enhanced_cross_group_heatmap.{ext}"),
                   dpi=150, bbox_inches="tight")
plt.close()
log("C5b enhanced heatmap saved")

# ============================================================
# C6. Cross-group concordance
# ============================================================
log("C6: cross-group concordance")
groups_with_sig = sorted(group_sig_regions.keys())
ng = len(groups_with_sig)
conc_mat = np.zeros((ng, ng))
for i in range(ng):
    for j in range(ng):
        si = group_sig_regions[groups_with_sig[i]]
        sj = group_sig_regions[groups_with_sig[j]]
        conc_mat[i,j] = len(si) if i==j else len(si & sj)

conc_df = pd.DataFrame(conc_mat.astype(int), index=groups_with_sig, columns=groups_with_sig)
conc_df.to_csv(os.path.join(CROSS_DIR, "C6_significant_region_concordance.csv"))

fig, ax = plt.subplots(figsize=(max(8, ng*0.7), max(7, ng*0.6)))
mask_diag = np.eye(ng, dtype=bool)
sns.heatmap(conc_df, annot=True, fmt="d", cmap="Blues", ax=ax,
            mask=mask_diag, linewidths=0.5, linecolor="white",
            cbar_kws={"label":"Shared Significant Regions"})
for i in range(ng):
    ax.text(i+0.5, i+0.5, f"{int(conc_mat[i,i]):,}", ha="center", va="center",
            fontsize=7, fontweight="bold", color="#c0392b")
ax.set_title("Cross-Group Concordance — Brain Cell Types\n"
             "(diagonal = total sig per group, off-diagonal = shared)",
             fontsize=12, fontweight="bold")
plt.xticks(rotation=45, ha="right", fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
_save(fig, os.path.join(CROSS_DIR, "C6_concordance_heatmap"))
log("C6 saved")

# ============================================================
# C7. Regulatory complexity index
# ============================================================
log("C7: regulatory complexity")
complexity_rows = []
for st in ordered_labels:
    prof = subtype_profiles[st]
    tc = subtype_type_comp.get(st, {})
    total_active = sum(tc.values())
    type_counts = np.array([tc.get(c,0) for c in REG_CATS], dtype=float)
    type_probs = type_counts / type_counts.sum() if type_counts.sum()>0 else type_counts
    shannon = entropy(type_probs, base=2)
    pct_vals = np.array(list(prof.values()))
    complexity_rows.append({
        "Subtype": st, "Group": subtype_group[st],
        "nCells": subtype_ncells.get(st, 0),
        "Total_active_regions": total_active,
        "Shannon_type_diversity": round(shannon, 4),
        "Mean_presence_pct": round(pct_vals.mean(), 2) if len(pct_vals)>0 else 0,
        "Median_presence_pct": round(np.median(pct_vals), 2) if len(pct_vals)>0 else 0,
        "Regions_gt10pct": int((pct_vals>10).sum()) if len(pct_vals)>0 else 0,
    })
complexity_df = pd.DataFrame(complexity_rows)
complexity_df.to_csv(os.path.join(CROSS_DIR, "C7_regulatory_complexity_index.csv"), index=False)

fig, ax = plt.subplots(figsize=(12, 8))
for gn in unique_groups:
    sub = complexity_df[complexity_df["Group"]==gn]
    sizes = np.clip(sub["nCells"].values/10, 10, 300)
    ax.scatter(sub["Total_active_regions"], sub["Shannon_type_diversity"],
               s=sizes, c=gpal[gn], alpha=0.7, edgecolors="k", linewidths=0.4,
               label=gn, zorder=2)
    for _,r in sub.iterrows():
        short = r["Subtype"].split("__")[1] if "__" in r["Subtype"] else r["Subtype"]
        ax.annotate(short, (r["Total_active_regions"], r["Shannon_type_diversity"]),
                    fontsize=5, alpha=0.7, xytext=(3,3), textcoords="offset points")
ax.set_xlabel("Total Active Regulatory Regions", fontsize=12)
ax.set_ylabel("Shannon Diversity (bits)", fontsize=12)
ax.set_title("Regulatory Complexity — Brain Cell Types\n(bubble size = cell count)",
             fontsize=13, fontweight="bold")
_smart_legend(ax, title="Sample Group", fontsize=7, title_fontsize=8)
ax.grid(True, alpha=0.15); fig.subplots_adjust(bottom=0.16)
_save(fig, os.path.join(CROSS_DIR, "C7_regulatory_complexity_bubble"))
log("C7 saved")

# ============================================================
# C8. Conserved vs divergent
# ============================================================
log("C8: conserved vs divergent")
region_breadth = {}
for j, reg in enumerate(regions_all):
    region_breadth[reg] = int((mat[:,j] > 5.0).sum())

n80 = int(n_st * 0.8); n20 = int(n_st * 0.2)
universal = [r for r,b in region_breadth.items() if b >= n80]
shared    = [r for r,b in region_breadth.items() if n20 <= b < n80]
specific  = [r for r,b in region_breadth.items() if b < n20]

conserv = {"Universal (>80% subtypes)": len(universal),
           "Shared (20-80%)": len(shared),
           "Subtype-specific (<20%)": len(specific),
           "Total": n_reg}
pd.DataFrame([conserv]).to_csv(os.path.join(CROSS_DIR, "C8_conserved_vs_divergent_summary.csv"), index=False)

fig, ax = plt.subplots(figsize=(7, 5))
cats = list(conserv.keys())[:-1]
vals = [conserv[c] for c in cats]
colors = ["#27ae60", "#f39c12", "#e74c3c"]
bars = ax.bar(cats, vals, color=colors, edgecolor="white", linewidth=0.5)
for bar, v in zip(bars, vals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(),
            f"{v:,}\n({v/n_reg*100:.1f}%)", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_ylabel("Number of Regulatory Regions", fontsize=12)
ax.set_title("Conserved vs Divergent — Brain Cell Types\n"
             f"(across {n_st} subtypes, >5% presence threshold)", fontsize=13, fontweight="bold")
plt.tight_layout()
_save(fig, os.path.join(CROSS_DIR, "C8_conserved_vs_divergent_bar"))
log("C8 saved")

# ============================================================
# C9. Pairwise divergence summary
# ============================================================
log("C9: pairwise divergence summary")
summary_rows = []
for i, st_a in enumerate(subtypes_all):
    for j, st_b in enumerate(subtypes_all):
        if i >= j: continue
        summary_rows.append({
            "Subtype_A": st_a, "Group_A": subtype_group[st_a],
            "Subtype_B": st_b, "Group_B": subtype_group[st_b],
            "Spearman_divergence": round(dist_mat[i,j], 6),
            "Same_group": subtype_group[st_a] == subtype_group[st_b],
        })
pd.DataFrame(summary_rows).sort_values("Spearman_divergence").to_csv(
    os.path.join(CROSS_DIR, "C9_pairwise_divergence_all_subtypes.csv"), index=False)

div_df = pd.DataFrame(summary_rows)
fig, ax = plt.subplots(figsize=(8, 6))
for same, grp in div_df.groupby("Same_group"):
    lbl = "Within group" if same else "Between groups"
    col = "#3498db" if same else "#e74c3c"
    ax.hist(grp["Spearman_divergence"], bins=40, alpha=0.5, color=col,
            label=f"{lbl} (n={len(grp):,})")
ax.set_xlabel("Spearman Divergence (1 - \u03c1)", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_title("Intra- vs Inter-Group Divergence — Brain Cell Types",
             fontsize=13, fontweight="bold")
_smart_legend(ax, fontsize=10, outside=False)
plt.tight_layout()
_save(fig, os.path.join(CROSS_DIR, "C9_intra_vs_inter_divergence"))
log("C9 saved")

# ============================================================
# C10. Per-group TOP-ACTIVITY heatmaps
#   For each subtype, find the most active regions (highest % cells)
#   and show how they compare across all other subtypes in the group.
# ============================================================
log("C10: per-group top-activity heatmaps")

TOP_PER_SUBTYPE = 10
ACTIVITY_DIR = os.path.join(CROSS_DIR, "top_activity_per_group")
os.makedirs(ACTIVITY_DIR, exist_ok=True)

for pfx in sorted(available_groups):
    reg_master = os.path.join(BASE_DIR, f"{pfx}_differential_unified_celltypes",
                              "regulatory_only", "regulatory_only_master_table.csv")
    if not os.path.exists(reg_master):
        continue

    df_g = pd.read_csv(reg_master, low_memory=False, dtype={"Chromosome":str})
    pct_cols_g = [c for c in df_g.columns if c.startswith("Pct_")]
    cts_g = [c.replace("Pct_","") for c in pct_cols_g]
    n_cts_g = len(cts_g)

    if n_cts_g < 2:
        # Single-subtype group: generate enrichment bar chart instead of heatmap
        only_ct = cts_g[0] if cts_g else pfx
        pct_col = f"Pct_{only_ct}"
        if pct_col not in df_g.columns:
            log(f"  SKIP {pfx}: no Pct column found")
            continue
        TOP_N_ENRICH = 30
        top_en = df_g.nlargest(TOP_N_ENRICH, pct_col).copy()
        if top_en.empty:
            log(f"  SKIP {pfx}: no rows to plot")
            continue

        top_en["_stype"] = top_en["Region_Type"].apply(_simplify_type)
        top_en["_label"] = top_en.apply(
            lambda r: f"{r['Chromosome']}:{r['Start']}-{r['End']} [{_simplify_type(r['Region_Type'])}]",
            axis=1
        )

        fig_h_s = max(8, len(top_en) * 0.35 + 2)
        fig, ax = plt.subplots(figsize=(13, fig_h_s))
        y_pos = np.arange(len(top_en))[::-1]   # descending from top
        colors_bar = [SIMPLE_TYPE_COLORS.get(t, "#95a5a6") for t in top_en["_stype"]]
        bars = ax.barh(y_pos, top_en[pct_col].values, color=colors_bar,
                       edgecolor="black", linewidth=0.4, height=0.75)

        # Annotate bars with exact % values
        for i, (bar, val) in enumerate(zip(bars, top_en[pct_col].values)):
            ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                    f"{val:.1f}%", va="center", fontsize=9, fontweight="bold")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_en["_label"].tolist(), fontsize=8)
        ax.set_xlabel("Presence (% of cells)", fontsize=12, fontweight="bold")
        ax.set_title(
            f"Top {len(top_en)} Most Enriched Regulatory Regions — {pfx}\n"
            f"(Single-subtype group: {only_ct} | n cells from master table)",
            fontsize=13, fontweight="bold"
        )
        ax.grid(True, alpha=0.2, axis="x")
        ax.invert_yaxis()

        # Legend: region types present
        used_types_s = sorted(top_en["_stype"].unique())
        handles_s = [mpatches.Patch(color=SIMPLE_TYPE_COLORS.get(t,"#95a5a6"), label=t) for t in used_types_s]
        ax.legend(handles=handles_s, title="Region Type", fontsize=9, title_fontsize=10,
                  loc="lower right", framealpha=0.9, edgecolor="#bdc3c7")

        plt.tight_layout()
        for ext in ("png","pdf"):
            try:
                fig.savefig(os.path.join(ACTIVITY_DIR, f"C10_top_activity_{pfx}.{ext}"),
                            dpi=300, bbox_inches="tight")
            except ValueError:
                fig.savefig(os.path.join(ACTIVITY_DIR, f"C10_top_activity_{pfx}.{ext}"),
                            dpi=150, bbox_inches="tight")
        plt.close(fig)
        log(f"  C10 {pfx}: single-subtype enrichment bar — top {len(top_en)} regions")
        continue

    # Load pairwise FDR for significance stars
    fdr_lookup = {}
    pair_dir = os.path.join(BASE_DIR, f"{pfx}_differential_unified_celltypes",
                            "regulatory_only", "pairwise")
    if os.path.isdir(pair_dir):
        for pf in glob.glob(os.path.join(pair_dir, "*.csv")):
            try:
                pw = pd.read_csv(pf, usecols=["Chromosome","Start","End","FDR"],
                                 low_memory=False, dtype={"Chromosome":str})
                for _, r in pw.iterrows():
                    key = (r["Chromosome"], int(r["Start"]), int(r["End"]))
                    if key not in fdr_lookup or r["FDR"] < fdr_lookup[key]:
                        fdr_lookup[key] = r["FDR"]
            except:
                pass

    # For each subtype, collect top N by presence %
    all_items = []
    seen = set()

    for ct in cts_g:
        top = df_g.nlargest(TOP_PER_SUBTYPE, f"Pct_{ct}")
        for _, row in top.iterrows():
            reg = (row["Chromosome"], int(row["Start"]), int(row["End"]))
            if reg not in seen:
                seen.add(reg)
                rtype = _simplify_type(row["Region_Type"])
                pcts = {c: row[f"Pct_{c}"] for c in cts_g}
                fdr = fdr_lookup.get(reg, 1.0)
                stars = "***" if fdr < 0.001 else "**" if fdr < 0.01 else "*" if fdr < 0.05 else ""
                all_items.append({
                    "reg": reg, "anchor": ct, "anchor_pct": row[f"Pct_{ct}"],
                    "type": rtype, "pcts": pcts, "fdr": fdr, "stars": stars,
                })

    # Sort: by anchor subtype order, then by anchor_pct descending
    ct_order = {ct: i for i, ct in enumerate(cts_g)}
    all_items.sort(key=lambda x: (ct_order[x["anchor"]], -x["anchor_pct"]))

    n_regs_g = len(all_items)

    # Build matrix + labels
    mat_act = np.zeros((n_regs_g, n_cts_g))
    rlbls = []
    r_anchors = []
    r_types = []
    r_stars = []

    for i, item in enumerate(all_items):
        c, s, e = item["reg"]
        rlbls.append(f"{c}:{s}-{e}  [{item['type']}]  {item['anchor_pct']:.0f}%{item['stars']}")
        r_anchors.append(item["anchor"])
        r_types.append(item["type"])
        r_stars.append(item["stars"])
        for j, ct in enumerate(cts_g):
            mat_act[i, j] = item["pcts"].get(ct, 0.0)

    # Within-group subtype distance for dendrogram
    pct_full = df_g[pct_cols_g].values.T   # (n_cts, n_all_regions)
    g_dist = np.zeros((n_cts_g, n_cts_g))
    for ii in range(n_cts_g):
        for jj in range(ii+1, n_cts_g):
            rho_val, _ = spearmanr(pct_full[ii], pct_full[jj])
            if np.isnan(rho_val): rho_val = 0.0
            g_dist[ii,jj] = g_dist[jj,ii] = 1.0 - rho_val
    Z_g = linkage(np.clip(squareform(g_dist), 0, None), method="average")

    # Subtype color palette for this group
    ct_pal = dict(zip(cts_g, sns.color_palette("husl", n_cts_g).as_hex()))

    # Build clustermap data
    heat_act = pd.DataFrame(mat_act, index=rlbls, columns=cts_g)

    col_c = pd.Series([ct_pal[ct] for ct in cts_g], index=cts_g, name="Subtype")
    row_c_anch = pd.Series([ct_pal[a] for a in r_anchors], index=rlbls, name="Top in")
    row_c_type = pd.Series([SIMPLE_TYPE_COLORS.get(t,"#95a5a6") for t in r_types],
                           index=rlbls, name="Type")

    fig_w = max(12, n_cts_g * 0.9)
    fig_h = max(10, n_regs_g * 0.20 + 5)

    g10 = sns.clustermap(
        heat_act, col_linkage=Z_g, row_cluster=False,
        col_colors=col_c,
        row_colors=[row_c_anch, row_c_type],
        cmap="YlOrRd", vmin=0,
        figsize=(fig_w, fig_h),
        xticklabels=True, yticklabels=True,
        linewidths=0.3, linecolor="white",
        dendrogram_ratio=(0.20, 0),
        cbar_pos=(0.02, 0.78, 0.015, 0.15),
        cbar_kws={"label": "Presence (%)", "orientation": "vertical"},
    )

    # Overlay % text + stars on max-presence cell
    hax = g10.ax_heatmap
    rd = g10.data2d.values
    max_c = np.argmax(rd, axis=1)

    for i in range(rd.shape[0]):
        for j in range(rd.shape[1]):
            val = rd[i, j]
            if val < 0.5:
                continue
            is_max = (j == max_c[i])
            txt = f"{val:.0f}"
            if is_max and r_stars[i]:
                txt += r_stars[i]
            tc = "white" if val > 45 else "black"
            fw = "bold" if is_max else "normal"
            fs = 6 if is_max else 5
            hax.text(j+0.5, i+0.5, txt, ha="center", va="center",
                     fontsize=fs, fontweight=fw, color=tc)

    # Section separators between anchor subtypes
    prev_a = r_anchors[0]
    for i in range(1, n_regs_g):
        if r_anchors[i] != prev_a:
            hax.axhline(y=i, color="#2c3e50", linewidth=2.0)
            # Section label on the left
            mid = (i + [k for k in range(i, n_regs_g) if r_anchors[k] != r_anchors[i]][0]
                   if any(r_anchors[k] != r_anchors[i] for k in range(i, n_regs_g))
                   else i + (n_regs_g - i) / 2) / 2  # rough midpoint
            prev_a = r_anchors[i]

    # Column labels bold
    hax.set_xticklabels([l.get_text() for l in hax.get_xticklabels()],
                        fontsize=11, fontweight="bold", rotation=45, ha="right")
    hax.set_yticklabels(hax.get_yticklabels(), fontsize=7)

    # Bold significant row labels
    for lbl in hax.get_yticklabels():
        txt = lbl.get_text()
        if "***" in txt:
            lbl.set_fontweight("bold"); lbl.set_color("#b71c1c")
        elif "**" in txt:
            lbl.set_fontweight("bold"); lbl.set_color("#c62828")
        elif "*" in txt and "**" not in txt:
            lbl.set_fontweight("bold"); lbl.set_color("#d32f2f")

    # Dendrogram styling
    dax = g10.ax_col_dendrogram
    dax.set_title(f"Subtype Similarity ({pfx})", fontsize=11, fontweight="bold", pad=4)
    for line in dax.collections:
        line.set_linewidth(1.5)
    for line in dax.lines:
        line.set_linewidth(1.5)
        line.set_color("#2c3e50")

    # Title
    g10.fig.suptitle(
        f"Most Active Regulatory Regions per Subtype — {pfx}\n"
        f"Top {TOP_PER_SUBTYPE} by Presence (% cells) | "
        f"Sections grouped by anchor subtype\n"
        f"{n_regs_g} regions | {n_cts_g} subtypes | "
        f"* FDR<0.05  ** FDR<0.01  *** FDR<0.001",
        fontsize=14, fontweight="bold", y=1.06
    )

    # Significance box
    g10.fig.text(0.98, 1.04,
        "Row labels: coordinate [type] anchor_activity% + FDR stars\n"
        "Left bars: anchor subtype (color) | region type",
        fontsize=8, ha="right", fontstyle="italic",
        transform=g10.fig.transFigure,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#e8f5e9", edgecolor="#2e7d32", alpha=0.9))

    # Legend
    handles_10 = []
    handles_10.append(mpatches.Patch(color="none", label="$\\bf{Subtypes:}$"))
    for ct in cts_g:
        handles_10.append(mpatches.Patch(color=ct_pal[ct], label=ct))
    handles_10.append(mpatches.Patch(color="none", label=""))
    handles_10.append(mpatches.Patch(color="none", label="$\\bf{Region\\ Types:}$"))
    used_t = sorted(set(r_types))
    for t in used_t:
        handles_10.append(mpatches.Patch(color=SIMPLE_TYPE_COLORS.get(t,"#95a5a6"), label=t))

    ncl = max(3, (len(handles_10)+2)//3)
    g10.fig.legend(handles=handles_10, fontsize=9, ncol=ncl,
                   loc="lower center", bbox_to_anchor=(0.52, -0.07),
                   framealpha=0.9, edgecolor="#bdc3c7", columnspacing=2.5)

    for ext in ("png","pdf"):
        try:
            g10.savefig(os.path.join(ACTIVITY_DIR, f"C10_top_activity_{pfx}.{ext}"),
                        dpi=300, bbox_inches="tight")
        except ValueError:
            g10.savefig(os.path.join(ACTIVITY_DIR, f"C10_top_activity_{pfx}.{ext}"),
                        dpi=150, bbox_inches="tight")
    plt.close()
    log(f"  C10 {pfx}: {n_regs_g} regions across {n_cts_g} subtypes")

log("C10 complete")

# ============================================================
# C11. Cross-group top-activity heatmap
#   Top regions by MAX presence (highest activity in any subtype)
#   compared across all subtypes of all groups.
# ============================================================
log("C11: cross-group top-activity heatmap")

top_act_regs = []
top_act_labels = []
top_act_types = []
top_act_maxpct = []

for pfx_c11 in sorted(available_groups):
    rm = os.path.join(BASE_DIR, f"{pfx_c11}_differential_unified_celltypes",
                      "regulatory_only", "regulatory_only_master_table.csv")
    if not os.path.exists(rm): continue
    df_c11 = pd.read_csv(rm, low_memory=False, dtype={"Chromosome":str})
    top = df_c11.nlargest(10, "Max_pct")
    for _, row in top.iterrows():
        reg = (row["Chromosome"], int(row["Start"]), int(row["End"]))
        if reg not in top_act_regs:
            top_act_regs.append(reg)
            st = _simplify_type(row["Region_Type"])
            top_act_types.append(st)
            top_act_maxpct.append(row["Max_pct"])
            top_act_labels.append(
                f"{rlabel(*reg)} [{st}] Max={row['Max_pct']:.0f}% ({pfx_c11})")

if len(top_act_regs) > 0:
    # Sort by Max_pct descending
    sort_act = np.argsort(top_act_maxpct)[::-1]
    top_act_regs = [top_act_regs[i] for i in sort_act]
    top_act_labels = [top_act_labels[i] for i in sort_act]
    top_act_types = [top_act_types[i] for i in sort_act]
    top_act_maxpct = [top_act_maxpct[i] for i in sort_act]

    n_ta = len(top_act_regs)
    act_mat = np.zeros((n_ta, n_st))
    for i, reg in enumerate(top_act_regs):
        for j, st_name in enumerate(subtypes_all):
            act_mat[i,j] = subtype_profiles[st_name].get(reg, 0.0)

    act_df = pd.DataFrame(act_mat, index=top_act_labels, columns=subtypes_all)

    row_c_t11 = pd.Series([SIMPLE_TYPE_COLORS.get(t,"#95a5a6") for t in top_act_types],
                           index=top_act_labels, name="Type")

    fig_w11 = max(14, n_st * 0.50)
    fig_h11 = max(10, n_ta * 0.22 + 5)

    g11 = sns.clustermap(
        act_df, col_linkage=Z, row_cluster=False,
        col_colors=[col_colors_group, col_colors_subtype],
        row_colors=row_c_t11,
        cmap="YlOrRd", vmin=0,
        figsize=(fig_w11, fig_h11),
        xticklabels=True, yticklabels=True,
        linewidths=0.3, linecolor="white",
        dendrogram_ratio=(0.22, 0),
        cbar_pos=(0.02, 0.78, 0.015, 0.15),
        cbar_kws={"label": "Presence (%)", "orientation": "vertical"},
    )

    # Overlay % text
    hax11 = g11.ax_heatmap
    rd11 = g11.data2d.values
    max_c11 = np.argmax(rd11, axis=1)
    for i in range(rd11.shape[0]):
        for j in range(rd11.shape[1]):
            val = rd11[i,j]
            if val < 0.5: continue
            is_max = (j == max_c11[i])
            txt = f"{val:.0f}"
            tc = "white" if val > 45 else "black"
            fw = "bold" if is_max else "normal"
            fs = 5.5 if is_max else 4.5
            hax11.text(j+0.5, i+0.5, txt, ha="center", va="center",
                       fontsize=fs, fontweight=fw, color=tc)

    # Column labels
    short_x11 = [l.get_text().split("__")[1] if "__" in l.get_text() else l.get_text()
                 for l in hax11.get_xticklabels()]
    hax11.set_xticklabels(short_x11, fontsize=9, fontweight="bold", rotation=55, ha="right")
    hax11.set_yticklabels(hax11.get_yticklabels(), fontsize=7)

    # Dendrogram styling
    dax11 = g11.ax_col_dendrogram
    dax11.set_title("Subtype Relationship Dendrogram", fontsize=10, fontweight="bold", pad=4)
    for line in dax11.collections: line.set_linewidth(1.5)
    for line in dax11.lines: line.set_linewidth(1.5); line.set_color("#2c3e50")

    g11.fig.suptitle(
        "Cross-Group Top-Activity Regulatory Regions\n"
        "Ranked by Maximum Presence (% cells) Across Brain Cell Subtypes\n"
        f"{n_ta} regions | {n_st} subtypes | {len(available_groups)} groups",
        fontsize=14, fontweight="bold", y=1.06
    )

    # Legend
    h11 = []
    h11.append(mpatches.Patch(color="none", label="$\\bf{Groups:}$"))
    for gn in sorted(gpal.keys()):
        h11.append(mpatches.Patch(color=gpal[gn], label=gn))
    h11.append(mpatches.Patch(color="none", label=""))
    h11.append(mpatches.Patch(color="none", label="$\\bf{Types:}$"))
    for t in sorted(set(top_act_types)):
        h11.append(mpatches.Patch(color=SIMPLE_TYPE_COLORS.get(t,"#95a5a6"), label=t))
    ncl11 = max(4, (len(h11)+2)//3)
    g11.fig.legend(handles=h11, fontsize=9, ncol=ncl11,
                   loc="lower center", bbox_to_anchor=(0.52, -0.07),
                   framealpha=0.9, edgecolor="#bdc3c7", columnspacing=3.0)

    for ext in ("png","pdf"):
        try:
            g11.savefig(os.path.join(CROSS_DIR, f"C11_cross_group_top_activity.{ext}"),
                        dpi=300, bbox_inches="tight")
        except ValueError:
            g11.savefig(os.path.join(CROSS_DIR, f"C11_cross_group_top_activity.{ext}"),
                        dpi=150, bbox_inches="tight")
    plt.close()
    log(f"C11: {n_ta} regions across {n_st} subtypes")

log("C11 complete")

# ============================================================
# COMPANION DATA TABLES — one CSV behind every figure
# ============================================================
log("Writing companion data tables for all figures")
TAB_DIR = os.path.join(CROSS_DIR, "figure_tables")
os.makedirs(TAB_DIR, exist_ok=True)
_index_rows = []

def _dump(df, fname, figure, desc, index=True):
    try:
        path = os.path.join(TAB_DIR, fname)
        df.to_csv(path, index=index)
        _index_rows.append({"Figure": figure, "Image": figure + ".png",
                            "Companion_Table": os.path.join("figure_tables", fname),
                            "Description": desc})
        log(f"  table: {fname} ({len(df)} rows)")
    except Exception as e:
        log(f"  WARN table {fname}: {e}")

def _short(st):
    return st.split("__")[1] if "__" in st else st

# C1 — divergence matrix (already saved at top level; register + copy here)
try:
    _dump(dist_df, "C1_divergence_matrix.csv", "C1_divergence_clustermap",
          "Pairwise Spearman divergence (1-rho) between all subtypes")
except Exception as e: log(f"  WARN C1 table: {e}")

# C2 — PCA coordinates + scree
try:
    pca_tab = pd.DataFrame({
        "Subtype": [_short(s) for s in subtypes_all],
        "Group":   [subtype_group[s] for s in subtypes_all],
        "n_cells": [subtype_ncells.get(s, 0) for s in subtypes_all],
    })
    for k in range(pca_coords.shape[1]):
        pca_tab[f"PC{k+1}"] = pca_coords[:, k]
    _dump(pca_tab, "C2_PCA_coordinates.csv", "C2_PCA_regulatory_fingerprints",
          "Per-subtype PCA coordinates of the regulatory landscape", index=False)
    scree = pd.DataFrame({
        "PC": [f"PC{i+1}" for i in range(len(pca.explained_variance_ratio_))],
        "Variance_explained_pct": pca.explained_variance_ratio_ * 100,
        "Cumulative_pct": np.cumsum(pca.explained_variance_ratio_) * 100,
    })
    _dump(scree, "C2_PCA_scree.csv", "C2_PCA_scree",
          "Variance explained per principal component", index=False)
except Exception as e: log(f"  WARN C2 table: {e}")

# C3 — t-SNE coordinates
try:
    if "tsne_coords" in globals():
        tsne_tab = pd.DataFrame({
            "Subtype": [_short(s) for s in subtypes_all],
            "Group":   [subtype_group[s] for s in subtypes_all],
            "tSNE_1": tsne_coords[:, 0], "tSNE_2": tsne_coords[:, 1],
        })
        _dump(tsne_tab, "C3_tSNE_coordinates.csv", "C3_tSNE_regulatory_divergence",
              "Per-subtype t-SNE coordinates", index=False)
except Exception as e: log(f"  WARN C3 table: {e}")

# C4 — region-type composition (%)
try:
    comp_rows = []
    for k in range(n_st):
        lbl = ordered_labels[k]
        tot = sum(subtype_type_comp[lbl].values()) or 1
        row = {"Subtype": _short(lbl), "Group": subtype_group[lbl]}
        for cat in REG_CATS:
            row[f"{cat}_pct"] = subtype_type_comp[lbl].get(cat, 0) / tot * 100
            row[f"{cat}_n"]   = subtype_type_comp[lbl].get(cat, 0)
        comp_rows.append(row)
    _dump(pd.DataFrame(comp_rows), "C4_region_type_composition.csv",
          "C4_region_type_composition_shift",
          "Per-subtype regulatory region-type composition (% and counts)", index=False)
except Exception as e: log(f"  WARN C4 table: {e}")

# C5 — top-10-per-group region presence matrix
try:
    if "rank_mat" in globals() and len(top_cross_regs) > 0:
        c5 = pd.DataFrame(rank_mat, index=top_cross_labels,
                          columns=[_short(s) for s in subtypes_all])
        _dump(c5, "C5_top_regions_presence_matrix.csv",
              "C5_top_regions_cross_group_heatmap",
              "Presence (%) of each top-10-per-group region across all subtypes")
except Exception as e: log(f"  WARN C5 table: {e}")

# C5b — enhanced heatmap matrix + per-region stats
try:
    if "rank_mat_sorted" in globals():
        c5b = pd.DataFrame(rank_mat_sorted, columns=[_short(s) for s in subtypes_all])
        c5b.insert(0, "Region", [f"{c}:{s}-{e}" for (c, s, e) in top_cross_regs_sorted])
        c5b.insert(1, "Region_type", top_types_sorted)
        c5b.insert(2, "Source_group", top_source_sorted)
        c5b.insert(3, "Diff_pct", np.round(top_diff_sorted, 3))
        c5b.insert(4, "min_FDR", top_fdr_sorted)
        _dump(c5b, "C5b_enhanced_heatmap_matrix.csv",
              "C5b_enhanced_cross_group_heatmap",
              "Top divergent regions: presence matrix + diff% + min FDR + source group",
              index=False)
except Exception as e: log(f"  WARN C5b table: {e}")

# C10 — per-group top-activity tables
try:
    ACT_TAB = os.path.join(CROSS_DIR, "top_activity_per_group")
    os.makedirs(ACT_TAB, exist_ok=True)
    for pfx_t in sorted(available_groups):
        rm = os.path.join(BASE_DIR, f"{pfx_t}_differential_unified_celltypes",
                          "regulatory_only", "regulatory_only_master_table.csv")
        if not os.path.exists(rm):
            continue
        dft = pd.read_csv(rm, low_memory=False, dtype={"Chromosome": str})
        keep = [c for c in dft.columns
                if c in ("Chromosome", "Start", "End", "Region_Type", "Max_pct",
                         "Min_pct", "Diff_pct", "Max_Celltype")
                or c.startswith("Pct_")]
        top = dft.nlargest(30, "Max_pct")[keep]
        top.to_csv(os.path.join(ACT_TAB, f"C10_top_activity_{pfx_t}_table.csv"), index=False)
    _index_rows.append({"Figure": "C10_top_activity_<group>",
                        "Image": "top_activity_per_group/C10_top_activity_<group>.png",
                        "Companion_Table": "top_activity_per_group/C10_top_activity_<group>_table.csv",
                        "Description": "Per-group top-30 most active regulatory regions (by max presence)"})
    log("  tables: C10 per-group top-activity")
except Exception as e: log(f"  WARN C10 tables: {e}")

# C11 — cross-group top-activity matrix
try:
    if "act_mat" in globals() and len(top_act_regs) > 0:
        c11 = pd.DataFrame(act_mat, columns=[_short(s) for s in subtypes_all])
        c11.insert(0, "Region", [f"{c}:{s}-{e}" for (c, s, e) in top_act_regs])
        c11.insert(1, "Region_type", top_act_types)
        c11.insert(2, "Max_pct", np.round(top_act_maxpct, 3))
        _dump(c11, "C11_cross_group_top_activity_matrix.csv",
              "C11_cross_group_top_activity",
              "Highest-activity regions (by max presence) across all subtypes", index=False)
except Exception as e: log(f"  WARN C11 table: {e}")

# Register already-saved top-level CSVs (C6/C7/C8/C9)
for fig, tab, desc in [
    ("C6_concordance_heatmap", "C6_significant_region_concordance.csv",
     "Jaccard concordance of significant regions between groups"),
    ("C7_regulatory_complexity_bubble", "C7_regulatory_complexity_index.csv",
     "Per-subtype active-region count + Shannon type diversity"),
    ("C8_conserved_vs_divergent_bar", "C8_conserved_vs_divergent_summary.csv",
     "Counts of conserved vs group-divergent regions"),
    ("C9_intra_vs_inter_divergence", "C9_pairwise_divergence_all_subtypes.csv",
     "All pairwise subtype divergences, labelled intra- vs inter-group"),
]:
    if os.path.exists(os.path.join(CROSS_DIR, tab)):
        _index_rows.append({"Figure": fig, "Image": fig + ".png",
                            "Companion_Table": tab, "Description": desc})

idx_df = pd.DataFrame(_index_rows, columns=["Figure", "Image", "Companion_Table", "Description"])
idx_df.to_csv(os.path.join(CROSS_DIR, "FIGURES_INDEX.csv"), index=False)
log(f"FIGURES_INDEX.csv written ({len(idx_df)} figures)")

# Mark complete
with open(CROSS_CKPT, "w") as f:
    f.write(time.strftime("%Y-%m-%d %H:%M:%S"))

elapsed = time.time() - t0
log(f"COMPLETE in {elapsed/60:.1f} min — outputs in {CROSS_DIR}")
