#!/usr/bin/env python3
"""
27_joint_differential_plots.py
==============================
Generate the per-group differential PLOT suite for the JOINT (known+novel) results
in Joint_Differential/{genebody,regulatory}/, matching the plots the known
per-group differential produced (volcano, MA, top-25 significance heatmap,
sig-regions-per-pair, FDR distribution, Diff_pct distribution by type,
region-type composition) — with known/novel basket coloring added.

Reads : Joint_Differential/<version>/<group>_joint_master.csv.gz
        Joint_Differential/<version>/<group>_pairwise/<A>_vs_<B>.csv.gz
Writes: Joint_Differential/<version>/<group>_plots/*.png|.pdf
Env   : `zeros`.
"""
from __future__ import annotations
import os, glob, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/path/to/data"
DIFF = f"{BASE}/Joint_Differential"
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
                     "pdf.fonttype": 42, "font.size": 9})
KNOWN_C, NOVEL_C = "#3498db", "#e74c3c"          # basket colors
UP_C, DN_C, NS_C = "#e74c3c", "#3498db", "#cccccc"


def save(fig, stem):
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def basket_col(df):
    return df["basket"] if "basket" in df.columns else pd.Series(
        np.where(df.get("region_class", "").astype(str).str.startswith("known"), "known", "novel"),
        index=df.index)


def pair_name(fp):
    return os.path.basename(fp).replace(".csv.gz", "").replace(".csv", "")


def plot_volcano(pdf, a, b, out):
    pdf = pdf.copy()
    pdf["neg_log10_FDR"] = -np.log10(pdf["FDR"].clip(lower=1e-300))
    pdf["bk"] = basket_col(pdf)
    sig = pdf["sig_FDR005"].fillna(False).astype(bool) if "sig_FDR005" in pdf else (pdf["FDR"] < 0.05)
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    ns = pdf[~sig]
    ax.scatter(ns["log2FC"], ns["neg_log10_FDR"], c=NS_C, s=6, alpha=0.25,
               edgecolors="none", rasterized=True, zorder=1, label=f"n.s. ({len(ns):,})")
    for bk, col in [("known", KNOWN_C), ("novel", NOVEL_C)]:
        m = sig & (pdf["bk"] == bk)
        ax.scatter(pdf.loc[m, "log2FC"], pdf.loc[m, "neg_log10_FDR"], c=col, s=12,
                   alpha=0.5, edgecolors="none", rasterized=True, zorder=2,
                   label=f"{bk} sig ({int(m.sum()):,})")
    for y in (-np.log10(0.05), -np.log10(0.01), -np.log10(0.001)):
        ax.axhline(y, color="#7f8c8d", ls="--", lw=0.6, alpha=0.5)
    ax.axvline(1, color="#7f8c8d", ls="--", lw=0.6, alpha=0.5)
    ax.axvline(-1, color="#7f8c8d", ls="--", lw=0.6, alpha=0.5)
    ax.set_xlabel(f"log2FC  (>0 up in {a}, <0 up in {b})")
    ax.set_ylabel("-log10(FDR)")
    ax.set_title(f"Volcano — {a} vs {b} (known+novel)")
    ax.legend(frameon=False, fontsize=7, loc="upper center", ncol=2)
    save(fig, f"{out}/volcano_{a}_vs_{b}")


def plot_ma(pdf, a, b, out):
    pdf = pdf.copy()
    pca, pcb = f"Pct_{a}", f"Pct_{b}"
    if pca not in pdf or pcb not in pdf:
        return
    A = (pdf[pca] + pdf[pcb]) / 2.0
    pdf["bk"] = basket_col(pdf)
    sig = pdf["sig_FDR005"].fillna(False).astype(bool) if "sig_FDR005" in pdf else (pdf["FDR"] < 0.05)
    fig, ax = plt.subplots(figsize=(7, 5.5), constrained_layout=True)
    ax.scatter(A[~sig], pdf.loc[~sig, "log2FC"], c=NS_C, s=6, alpha=0.25,
               edgecolors="none", rasterized=True, zorder=1)
    for bk, col in [("known", KNOWN_C), ("novel", NOVEL_C)]:
        m = sig & (pdf["bk"] == bk)
        ax.scatter(A[m], pdf.loc[m, "log2FC"], c=col, s=12, alpha=0.5, edgecolors="none",
                   rasterized=True, zorder=2, label=f"{bk} sig ({int(m.sum()):,})")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("A = mean per-cell % (accessibility)")
    ax.set_ylabel(f"M = log2FC ({a} vs {b})")
    ax.set_title(f"MA — {a} vs {b} (known+novel)")
    ax.legend(frameon=False, fontsize=7)
    save(fig, f"{out}/MA_{a}_vs_{b}")


def plot_group(master, subs, group, out, pair_summary):
    # top-25 significance heatmap (by Diff_pct), Pct per subtype, basket-annotated row labels
    top = master.nlargest(25, "Diff_pct")
    pcts = [f"Pct_{s}" for s in subs if f"Pct_{s}" in master.columns]
    if len(top) and pcts:
        H = top[pcts].values.astype(float)
        vmax = max(1.0, np.nanmax(H))
        fig, ax = plt.subplots(figsize=(max(7, 0.85 * len(pcts) + 4), 10), constrained_layout=True)
        im = ax.imshow(H, aspect="auto", cmap="magma", vmin=0, vmax=vmax)
        ax.set_xticks(range(len(pcts))); ax.set_xticklabels([c[4:] for c in pcts], rotation=45, ha="right", fontsize=8)
        lbls = [f"{rc.split('|')[0][0]}:{c}:{s}-{e}" for rc, c, s, e in
                zip(top["region_class"], top["Chromosome"], top["Start"], top["End"])]
        ax.set_yticks(range(len(top))); ax.set_yticklabels(lbls, fontsize=5)
        # annotate each block with its % value (contrast-aware text color)
        thr = vmax * 0.55
        for i in range(H.shape[0]):
            for j in range(H.shape[1]):
                v = H[i, j]
                ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                        fontsize=5.5, color=("white" if v < thr else "black"))
        fig.colorbar(im, ax=ax, label="per-cell %", shrink=0.5)
        ax.set_title(f"{group}: top-25 differential regions — values = per-cell %\n(row prefix k=known, n=novel)")
        save(fig, f"{out}/heatmap_top25_significance")
    # region-type composition (known|* vs novel|*)
    comp = master["region_class"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    cols = [KNOWN_C if str(i).startswith("known") else NOVEL_C for i in comp.index]
    ax.bar(range(len(comp)), comp.values, color=cols)
    ax.set_xticks(range(len(comp))); ax.set_xticklabels(comp.index, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("regions"); ax.set_title(f"{group}: region-class composition (blue=known, red=novel)")
    for i, v in enumerate(comp.values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=6)
    save(fig, f"{out}/region_type_composition")
    # Diff_pct distribution by basket
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    for bk, col in [("known", KNOWN_C), ("novel", NOVEL_C)]:
        v = master.loc[basket_col(master) == bk, "Diff_pct"].dropna()
        if len(v):
            ax.hist(v, bins=60, alpha=0.55, color=col, label=f"{bk} (n={len(v):,})")
    ax.set_xlabel("Diff_pct (max-min per-cell %)"); ax.set_ylabel("regions")
    ax.set_title(f"{group}: differential magnitude by basket"); ax.legend(frameon=False)
    save(fig, f"{out}/diff_pct_distribution_by_type")
    # sig regions per pair (stacked known/novel) + FDR distribution
    if pair_summary is not None and len(pair_summary):
        ps = pair_summary
        fig, ax = plt.subplots(figsize=(max(7, 0.5 * len(ps) + 2), 5), constrained_layout=True)
        x = range(len(ps))
        kn = ps.get("sig_FDR0.05_known", pd.Series([0] * len(ps))).values
        nv = ps.get("sig_FDR0.05_novel", pd.Series([0] * len(ps))).values
        ax.bar(x, kn, color=KNOWN_C, label="known")
        ax.bar(x, nv, bottom=kn, color=NOVEL_C, label="novel")
        ax.set_xticks(list(x)); ax.set_xticklabels(ps["Comparison"], rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("sig regions @FDR<0.05"); ax.set_title(f"{group}: significant regions per pair (known+novel)")
        ax.legend(frameon=False)
        save(fig, f"{out}/sig_regions_per_pair")


def run_version(version, only=None):
    vdir = f"{DIFF}/{version}"
    masters = sorted(glob.glob(f"{vdir}/*_joint_master.csv.gz"))
    for mf in masters:
        group = os.path.basename(mf).replace("_joint_master.csv.gz", "")
        if only and group not in only:
            continue
        out = f"{vdir}/{group}_plots"
        os.makedirs(out, exist_ok=True)
        master = pd.read_csv(mf, low_memory=False, dtype={"Chromosome": str})
        subs = [c[4:] for c in master.columns if c.startswith("Pct_")]
        psf = f"{vdir}/{group}_pairwise_summary.csv"
        psum = pd.read_csv(psf) if os.path.exists(psf) else None
        if not os.path.exists(f"{out}/region_type_composition.png"):
            plot_group(master, subs, group, out, psum)
        for pf in sorted(glob.glob(f"{vdir}/{group}_pairwise/*.csv.gz")):
            nm = pair_name(pf)
            if "_vs_" not in nm:
                continue
            a, b = nm.split("_vs_", 1)
            if os.path.exists(f"{out}/volcano_{a}_vs_{b}.png") and os.path.exists(f"{out}/MA_{a}_vs_{b}.png"):
                continue
            pdf = pd.read_csv(pf, low_memory=False, dtype={"Chromosome": str})
            plot_volcano(pdf, a, b, out)
            plot_ma(pdf, a, b, out)
        n = len(glob.glob(f"{out}/*.png"))
        print(f"[{version}] {group}: {n} plots -> {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="both", choices=["genebody", "regulatory", "both"])
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    only = set(args.only.split(",")) if args.only else None
    for v in (["genebody", "regulatory"] if args.version == "both" else [args.version]):
        run_version(v, only)
    print("JOINT DIFFERENTIAL PLOTS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
