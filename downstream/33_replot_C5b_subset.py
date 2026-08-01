#!/usr/bin/env python3
"""
33_replot_C5b_subset.py
=======================
Replot the C5b "enhanced cross-group heatmap" (cross_group_brain) for ONLY a
user-specified subset of regions/coordinates, in the identical style
(subtype dendrogram + group/subtype colour strips, region-type + source-group
row strips, YlOrRd presence %, value+FDR-star annotation).

Reuses the exact rendering logic of 21_replot_heatmaps.py.

Usage:
  python 33_replot_C5b_subset.py --coords "chr5:88659268-88682400,chr14:28754799-28786001" --out my_regions
  python 33_replot_C5b_subset.py --coords-file coords.txt --out my_regions   # one coord per line
  python 33_replot_C5b_subset.py --contains "chr5:513"                       # substring match (e.g. a locus)
Matching is flexible: full "chr:start-end", or a substring/prefix of the Region string.
Env: `zeros` (matplotlib, seaborn, scipy).
"""
import os, sys, argparse, colorsys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

BASE = "/path/to/data"
CGB = f"{BASE}/cross_group_brain"
FT = f"{CGB}/figure_tables"
C5B = f"{FT}/C5b_enhanced_heatmap_matrix.csv"
META = ["Region", "Region_type", "Source_group", "Diff_pct", "min_FDR"]
SIMPLE_TYPE_COLORS = {"enhancer": "#3498db", "silencer": "#2ecc71", "promoter": "#f39c12",
                      "gene_body": "#9b59b6", "uncharacterized": "#e74c3c", "insulator": "#1abc9c",
                      "other_regulatory": "#95a5a6"}


def _shade(hexc, factor):
    r, g, b = int(hexc[1:3], 16) / 255, int(hexc[3:5], 16) / 255, int(hexc[5:7], 16) / 255
    h, l, s = colorsys.rgb_to_hls(r, g, b); l = max(0.15, min(0.85, l * factor))
    r2, g2, b2 = colorsys.hls_to_rgb(h, l, s)
    return f"#{int(r2*255):02x}{int(g2*255):02x}{int(b2*255):02x}"


def stars(fdr):
    return "***" if fdr < 0.001 else "**" if fdr < 0.01 else "*" if fdr < 0.05 else ""


# ---- shared subtype order / dendrogram / colours (same as 21_replot_heatmaps.py) ----
dist = pd.read_csv(f"{CGB}/spearman_divergence_matrix.csv", index_col=0)
subtypes_all = list(dist.index)
groups = [s.split("__")[0] for s in subtypes_all]
uniq_groups = sorted(set(groups))
gpal = dict(zip(uniq_groups, sns.color_palette("husl", len(uniq_groups)).as_hex()))
Z = linkage(squareform(np.clip(dist.values, 0, None), checks=False), method="average")
subtype_group = dict(zip(subtypes_all, groups))
group_subtypes = {}
for st in subtypes_all:
    group_subtypes.setdefault(subtype_group[st], []).append(st)
subtype_colors = {}
for g, sts in group_subtypes.items():
    if len(sts) == 1:
        subtype_colors[sts[0]] = gpal[g]
    else:
        for k, st in enumerate(sorted(sts)):
            subtype_colors[st] = _shade(gpal[g], 0.7 + 0.6 * k / (len(sts) - 1))
col_colors_group = pd.Series([gpal[subtype_group[s]] for s in subtypes_all], index=subtypes_all, name="Group")
col_colors_subtype = pd.Series([subtype_colors[s] for s in subtypes_all], index=subtypes_all, name="Subtype")
n_st = len(subtypes_all)


def render(df, title, stem):
    sub_cols = [c for c in df.columns if c not in META]
    rlabels = [f"{r['Region']}  [{r['Region_type']}]  Diff={r['Diff_pct']:.1f}%  {stars(r['min_FDR'])}"
               for _, r in df.iterrows()]
    # CSV subtype columns are SHORT names in subtypes_all order -> assign positionally
    # (matches 21_replot_heatmaps.py; matching by name fails since dendrogram uses "Group__subtype")
    assert len(sub_cols) == n_st, f"expected {n_st} subtype cols, got {len(sub_cols)}"
    heat = pd.DataFrame(df[sub_cols].values, index=rlabels, columns=subtypes_all)
    n_tr = len(df)
    row_colors = [pd.Series([SIMPLE_TYPE_COLORS.get(t, "#95a5a6") for t in df["Region_type"]], index=rlabels, name="Region type"),
                  pd.Series([gpal.get(s, "#95a5a6") for s in df["Source_group"]], index=rlabels, name="Source group")]
    g = sns.clustermap(heat, col_linkage=Z, row_cluster=False,
                       col_colors=[col_colors_group, col_colors_subtype], row_colors=row_colors,
                       cmap="YlOrRd", vmin=0,
                       figsize=(max(15, n_st * 0.52), max(7, n_tr * 0.45 + 5)),
                       xticklabels=True, yticklabels=True, linewidths=0.3, linecolor="white",
                       dendrogram_ratio=(0.16, 0.02), cbar_pos=(1.04, 0.35, 0.015, 0.18),
                       cbar_kws={"label": "Presence (%)"})
    hax = g.ax_heatmap
    rd = g.data2d.values; maxc = np.argmax(rd, axis=1); fdr_vals = df["min_FDR"].values
    for i in range(rd.shape[0]):
        for j in range(rd.shape[1]):
            v = rd[i, j]
            if v < 0.5:
                continue
            ismax = (j == maxc[i])
            txt = f"{v:.0f}" + (stars(fdr_vals[i]) if ismax else "")
            hax.text(j + 0.5, i + 0.5, txt, ha="center", va="center",
                     fontsize=5.5 if ismax else 4.5, fontweight="bold" if ismax else "normal",
                     color="white" if v > 42 else "black")
    hax.set_xticklabels([l.get_text().split("__")[1] if "__" in l.get_text() else l.get_text()
                         for l in hax.get_xticklabels()], fontsize=9, fontweight="bold", rotation=55, ha="right")
    hax.set_yticklabels(hax.get_yticklabels(), fontsize=7)
    for lbl in hax.get_yticklabels():
        t = lbl.get_text()
        if "***" in t: lbl.set_color("#b71c1c"); lbl.set_fontweight("bold")
        elif "**" in t: lbl.set_color("#c62828"); lbl.set_fontweight("bold")
        elif "*" in t: lbl.set_color("#d32f2f"); lbl.set_fontweight("bold")
    g.ax_col_dendrogram.set_title("Subtype Relationship Dendrogram (Spearman distance)", fontsize=10, fontweight="bold", pad=4)
    for ln in g.ax_col_dendrogram.lines:
        ln.set_linewidth(1.5); ln.set_color("#2c3e50")
    # push suptitle above the dendrogram title (more headroom needed for few-row figures)
    g.fig.suptitle(title, fontsize=13.5, fontweight="bold", y=1.02 + 1.2 / max(7, n_tr * 0.45 + 5))
    used = sorted(set(df["Region_type"]))
    handles = [mpatches.Patch(color="none", label=r"$\bf{Groups:}$")] + \
              [mpatches.Patch(color=gpal[g_], label=g_) for g_ in uniq_groups] + \
              [mpatches.Patch(color="none", label=""), mpatches.Patch(color="none", label=r"$\bf{Region\ types:}$")] + \
              [mpatches.Patch(color=SIMPLE_TYPE_COLORS.get(t, "#95a5a6"), label=t) for t in used]
    g.fig.legend(handles=handles, fontsize=8.5, loc="center left", bbox_to_anchor=(1.035, 0.32),
                 framealpha=0.95, edgecolor="#bdc3c7", borderpad=0.6, labelspacing=0.35)
    for ext in ("png", "pdf"):
        g.fig.savefig(f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close("all")


def norm(s):
    return s.strip().replace(",", "").replace(" ", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coords", default=None, help="comma-separated coords")
    ap.add_argument("--coords-file", default=None, help="file, one coord per line")
    ap.add_argument("--contains", default=None, help="substring match on Region")
    ap.add_argument("--rowsort", default="diff", choices=["diff", "source_group", "input"],
                    help="row order: diff (Diff_pct desc), source_group (then Diff_pct), "
                         "or input (exact order of the given coords/file)")
    ap.add_argument("--out", default=f"{CGB}/C5b_subset")
    args = ap.parse_args()
    df = pd.read_csv(C5B)
    df["_key"] = df["Region"].map(norm)
    wanted = []
    if args.coords:
        wanted = [norm(c) for c in args.coords.split(",") if c.strip()]
    elif args.coords_file:
        wanted = [norm(l) for l in open(args.coords_file) if l.strip() and not l.startswith("#")]
    if args.contains:
        sub = df[df["Region"].str.contains(args.contains, regex=False)].copy()
    else:
        # match exact key OR substring (so "chr5:513" or "chr5:51378199" both work)
        mask = df["_key"].isin(wanted)
        for w in wanted:
            mask = mask | df["_key"].str.contains(w, regex=False)
        sub = df[mask].copy()
    sub = sub.drop_duplicates("Region").drop(columns="_key")
    if sub.empty:
        print("NO MATCHING REGIONS. Requested:", wanted or args.contains)
        print("Available example Regions:", list(df["Region"].head(5)))
        sys.exit(1)
    if args.rowsort == "source_group":
        sub = sub.sort_values(["Source_group", "Diff_pct"], ascending=[True, False]).reset_index(drop=True)
    elif args.rowsort == "input":
        # exact order of the requested coords (each wanted -> its matched Region)
        sub["_key2"] = sub["Region"].map(norm)
        ordered_regions = []
        for w in wanted:
            for _, rr in sub.iterrows():
                if rr["Region"] in ordered_regions:
                    continue
                if rr["_key2"] == w or w in rr["_key2"]:
                    ordered_regions.append(rr["Region"]); break
        # append any matched-but-unlisted rows at the end
        ordered_regions += [r for r in sub["Region"] if r not in ordered_regions]
        sub = sub.set_index("Region").loc[ordered_regions].reset_index().drop(columns="_key2")
    else:
        sub = sub.sort_values("Diff_pct", ascending=False).reset_index(drop=True)
    print(f"matched {len(sub)} region(s), row order = {args.rowsort}")
    sub.to_csv(f"{args.out}_matrix.csv", index=False)

    # ---- companion table: per-region summary for the figure legend ----
    sub_cols = [c for c in sub.columns if c not in META]
    comp = []
    for _, r in sub.iterrows():
        vals = r[sub_cols].astype(float)
        top = vals.sort_values(ascending=False)
        comp.append({
            "Region": r["Region"], "Region_type": r["Region_type"], "Source_group": r["Source_group"],
            "Diff_pct": round(float(r["Diff_pct"]), 1), "min_FDR": r["min_FDR"], "FDR_stars": stars(r["min_FDR"]),
            "max_subtype": top.index[0], "max_presence_pct": round(float(top.iloc[0]), 1),
            "2nd_subtype": top.index[1], "2nd_presence_pct": round(float(top.iloc[1]), 1),
            "n_subtypes_present_ge5pct": int((vals >= 5).sum()),
            "n_subtypes_present_ge25pct": int((vals >= 25).sum()),
            "mean_pct_across_49": round(float(vals.mean()), 1),
        })
    comp_df = pd.DataFrame(comp)
    comp_df.to_csv(f"{args.out}_companion_table.csv", index=False)

    render(sub, f"C5b subset — {len(sub)} selected region(s)"
                + (" (rows grouped by source celltype)" if args.rowsort == "source_group" else "")
                + "\nPresence (%) across 49 CSPMEI subtypes | * FDR<0.05 ** <0.01 *** <0.001",
           args.out)
    print(f"wrote {args.out}.png / .pdf, {args.out}_matrix.csv, {args.out}_companion_table.csv")
    print("\n=== companion table ===")
    print(comp_df[["Region", "Source_group", "Diff_pct", "FDR_stars", "max_subtype",
                   "max_presence_pct", "n_subtypes_present_ge25pct"]].to_string(index=False))


if __name__ == "__main__":
    main()
