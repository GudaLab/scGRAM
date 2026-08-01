#!/usr/bin/env python3
"""
41_foxp2_percell_novel.py — Panel 7: per-cell novel-peak burden.

For each cell, of its predicted regulatory peaks (category in enhancer/silencer/
enhancer_silencer) that fall inside the group's tested region universe, what
fraction land in NOVEL consensus loci vs KNOWN regulatory regions. Known/novel
1 kb-bin sets are built from the group's joint master (basket=known/novel), so
this reproduces the pipeline's binning at single-cell resolution.

Compares FOXP2_1-4 against representative subtypes of other groups.
Output: FOXP2_novel_evidence/P7_percell_novel_burden.{png,pdf} + _percell.csv
Env: zeros (pandas/numpy/matplotlib).
"""
import os, csv, glob
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/path/to/data"
JD = f"{BASE}/Joint_Differential/regulatory"
PRED = f"{BASE}/unbound_characetrize/predictions"
OUT = f"{BASE}/FOXP2_novel_evidence"
BIN = 1000
THREADS = 16
REG_CATS = {"enhancer", "silencer", "enhancer_silencer"}

# (group, subtype) to profile — FOXP2 all 4 + representative others
TARGETS = [
    ("FOXP2", "FOXP2_1"), ("FOXP2", "FOXP2_2"), ("FOXP2", "FOXP2_3"), ("FOXP2", "FOXP2_4"),
    ("MSN", "MSN_1"), ("Dopaminergic", "D1Pu"), ("CNGA", "CNGA_1"),
    ("ITL23", "ITL23_1"), ("PVALB", "PVALB_1"), ("VIP", "VIP_1"), ("ASCT", "ASCT_1"),
]


def group_bins(group):
    """Return (known_bins, novel_only_bins) as DISJOINT 1kb-bin sets with KNOWN
    precedence (a bin in both is counted known). Per-cell peaks are classified only
    if they fall in this tested universe -- so novel = peak at a novel-only locus,
    not merely 'anything not known'. Removes the earlier circular double-counting."""
    df = pd.read_csv(f"{JD}/{group}_joint_master.csv.gz",
                     usecols=["Chromosome", "Start", "End", "basket"],
                     low_memory=False, dtype={"Chromosome": str})
    known, novel = set(), set()
    for chrom, s, e, b in df.itertuples(index=False):
        tgt = novel if b == "novel" else known
        for k in range(int(s) // BIN, int(e) // BIN + 1):
            tgt.add(f"{chrom}:{k}")
    novel_only = novel - known          # known precedence
    return known, novel_only


def scan_cell(args):
    """Classify each regulatory peak: known if in known set, else novel if in the
    novel-only set; peaks outside the tested universe are ignored."""
    fp, known, novel_only = args
    nk = nn = 0
    try:
        with open(fp, newline="") as f:
            r = csv.reader(f); next(r, None)
            for row in r:
                if len(row) < 4 or row[3] not in REG_CATS:
                    continue
                try:
                    mid = (int(row[1]) + int(row[2])) // 2
                except (ValueError, IndexError):
                    continue
                key = f"{row[0]}:{mid // BIN}"
                if key in known:
                    nk += 1
                elif key in novel_only:
                    nn += 1
    except Exception:
        pass
    tot = nk + nn
    return (nn / tot) if tot else np.nan, tot


def main():
    rows = []
    bincache = {}
    for group, st in TARGETS:
        if group not in bincache:
            bincache[group] = group_bins(group)
            print(f"[bins] {group}: known={len(bincache[group][0]):,} novel={len(bincache[group][1]):,}")
        known, novel = bincache[group]
        cells = sorted(glob.glob(f"{PRED}/{group}/{st}/*_predictions.csv"))
        if not cells:
            print(f"  WARN no cells {group}/{st}"); continue
        with ThreadPoolExecutor(max_workers=THREADS) as pool:
            res = list(pool.map(scan_cell, [(c, known, novel) for c in cells]))
        for frac, tot in res:
            if not np.isnan(frac) and tot >= 5:   # need >=5 classified peaks
                rows.append({"group": group, "subtype": st, "novel_frac": frac * 100,
                             "n_reg_peaks": tot})
        vals = [r["novel_frac"] for r in rows if r["subtype"] == st]
        print(f"  {st}: {len(vals)} cells, median novel {np.median(vals):.1f}%")

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/P7_percell.csv", index=False)

    order = [st for _, st in TARGETS if st in set(df.subtype)]
    data = [df[df.subtype == st]["novel_frac"].values for st in order]
    isfox = [st.startswith("FOXP2") for st in order]
    fig, ax = plt.subplots(figsize=(12, 6))
    parts = ax.violinplot(data, showmedians=True, showextrema=False, widths=0.85)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor("#c0392b" if isfox[i] else "#95a5a6")
        pc.set_alpha(0.75)
    parts["cmedians"].set_color("black")
    meds = [np.median(d) for d in data]
    for i, m in enumerate(meds):
        ax.text(i + 1, m + 2, f"{m:.0f}%", ha="center", fontsize=8, fontweight="bold")
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels(order, rotation=45, ha="right", fontsize=9)
    for i, lbl in enumerate(ax.get_xticklabels()):
        if isfox[i]:
            lbl.set_color("#c0392b"); lbl.set_fontweight("bold")
    ax.set_ylabel("% of cell's regulatory peaks in NOVEL loci")
    fox_med = np.median(np.concatenate([data[i] for i in range(len(order)) if isfox[i]]))
    other_med = np.median(np.concatenate([data[i] for i in range(len(order)) if not isfox[i]]))
    ax.set_title("(7) Per-cell novel-peak burden — single-cell evidence\n"
                 f"FOXP2 cells median {fox_med:.0f}% novel vs {other_med:.0f}% for other subtypes",
                 fontsize=12, fontweight="bold")
    ax.axhline(other_med, ls="--", color="#7f8c8d", lw=1)
    ax.spines[["top", "right"]].set_visible(False)
    for e in ("png", "pdf"):
        fig.savefig(f"{OUT}/P7_percell_novel_burden.{e}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote P7 -> {OUT}  (FOXP2 {fox_med:.0f}% vs others {other_med:.0f}%)")


if __name__ == "__main__":
    main()
