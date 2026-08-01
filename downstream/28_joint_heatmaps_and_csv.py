#!/usr/bin/env python3
"""
28_joint_heatmaps_and_csv.py
============================
For every joint per-group differential (both versions), emit:
  - heatmap_top25_significance.(png|pdf)   annotated with per-cell % values
  - heatmap_top50_significance.(png|pdf)   annotated with per-cell % values
  - <group>_top25_differential.csv         full table for the top-25 regions
  - <group>_top50_differential.csv         full table for the top-50 regions
  - <group>_all_differential_ranked.csv.gz complete table, all regions ranked by Diff_pct

Robust rendering: figure height scales with row count; single-subtype groups handled;
NaN-safe; contrast-aware annotation text. Re-runs overwrite cleanly (fixes any
truncated file from an interrupted earlier run).

Env: `zeros`.  Reads Joint_Differential/<version>/<group>_joint_master.csv.gz
"""
from __future__ import annotations
import os, glob, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DIFF = "/path/to/data/Joint_Differential"
plt.rcParams.update({"savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42, "font.size": 9})

FRONT = ["region_id", "Chromosome", "Start", "End", "basket", "region_class",
         "Max_pct", "Min_pct", "Diff_pct", "log2FC", "Max_Celltype",
         "mean_p_enhancer", "mean_p_silencer", "mean_n_tfs"]


def heatmap(top, pcts, group, n, out):
    if not (len(top) and pcts):
        return
    H = top[pcts].values.astype(float)
    vmax = max(1.0, np.nanmax(H))
    thr = vmax * 0.55
    nrow, ncol = H.shape
    fig_h = max(6, 0.34 * nrow + 2.5)
    fig_w = max(7, 0.95 * ncol + 4)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
    im = ax.imshow(H, aspect="auto", cmap="magma", vmin=0, vmax=vmax)
    ax.set_xticks(range(ncol)); ax.set_xticklabels([c[4:] for c in pcts], rotation=45, ha="right", fontsize=8)
    lbls = [f"{str(rc).split('|')[0][0]}:{c}:{s}-{e}" for rc, c, s, e in
            zip(top["region_class"], top["Chromosome"], top["Start"], top["End"])]
    ax.set_yticks(range(nrow)); ax.set_yticklabels(lbls, fontsize=6 if nrow <= 25 else 4.5)
    fs = 6 if nrow <= 25 else 4.5
    for i in range(nrow):
        for j in range(ncol):
            v = H[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=fs,
                    color=("white" if v < thr else "black"))
    fig.colorbar(im, ax=ax, label="per-cell %", shrink=0.5)
    ax.set_title(f"{group}: top-{n} differential regions — values = per-cell %\n(row prefix k=known, n=novel)")
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}/heatmap_top{n}_significance.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_version(version, only=None):
    for mf in sorted(glob.glob(f"{DIFF}/{version}/*_joint_master.csv.gz")):
        group = os.path.basename(mf).replace("_joint_master.csv.gz", "")
        if only and group not in only:
            continue
        out = f"{DIFF}/{version}/{group}_plots"
        os.makedirs(out, exist_ok=True)
        m = pd.read_csv(mf, low_memory=False, dtype={"Chromosome": str})
        subs = [c[4:] for c in m.columns if c.startswith("Pct_")]
        pcts = [f"Pct_{s}" for s in subs]
        ranked = m.sort_values("Diff_pct", ascending=False).reset_index(drop=True)
        cols = [c for c in FRONT if c in ranked.columns] + [c for c in ranked.columns if c.startswith(("Count_", "Pct_"))]
        # CSV exports
        ranked.head(25)[cols].to_csv(f"{out}/{group}_top25_differential.csv", index=False)
        ranked.head(50)[cols].to_csv(f"{out}/{group}_top50_differential.csv", index=False)
        ranked[cols].to_csv(f"{out}/{group}_all_differential_ranked.csv.gz", index=False, compression="gzip")
        # heatmaps top-25 and top-50
        heatmap(ranked.head(25), pcts, group, 25, out)
        heatmap(ranked.head(50), pcts, group, 50, out)
        print(f"[{version}] {group}: heatmaps top25+top50 + CSVs (top25/top50/all={len(ranked):,})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="both", choices=["genebody", "regulatory", "both"])
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    only = set(args.only.split(",")) if args.only else None
    for v in (["genebody", "regulatory"] if args.version == "both" else [args.version]):
        run_version(v, only)
    print("JOINT HEATMAPS + CSV COMPLETE", flush=True)


if __name__ == "__main__":
    main()
