#!/usr/bin/env python3
"""
40_foxp2_evidence.py — evidence panels that FOXP2's regulatory fingerprint is
~58-66% NOVEL (ML-predicted) vs known regulatory elements.

Panel 1  Ranked novel-fraction-of-fingerprint across all groups (FOXP2 outlier)
Panel 2  Composition of FOXP2's top-500 fingerprint by region_class (donut)
Panel 4  Cumulative novel-fraction vs differential rank (novel carries top signal)
Panel 6  FOXP2_1-4 divergence: known-only vs novel-only (novel resolves substructure)

Outputs: FOXP2_novel_evidence/{P1_outlier,P2_composition,P4_rank_curve,
         P6_known_vs_novel_divergence}.{png,pdf} + a combined figure + source CSVs.
Env: zeros (pandas, numpy, scipy, matplotlib, seaborn).
"""
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

BASE = "/path/to/data"
JD = f"{BASE}/Joint_Differential/regulatory"
JC = f"{BASE}/Joint_CrossGroup/regulatory/figure_tables"
OUT = f"{BASE}/FOXP2_novel_evidence"
os.makedirs(OUT, exist_ok=True)
TOP = 500

NOVEL_COL = {"novel|enhancer": "#c0392b", "novel|dual": "#e67e22"}
KNOWN_COL = {"known|enhancer": "#2980b9", "known|silencer": "#16a085",
             "known|promoter": "#8e44ad", "known|other_regulatory": "#95a5a6",
             "known|gene_body": "#7f8c8d"}
ALLCOL = {**NOVEL_COL, **KNOWN_COL}


def savefig(fig, stem):
    for e in ("png", "pdf"):
        fig.savefig(f"{OUT}/{stem}.{e}", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------- Panel 1: ranked novel-fraction across groups ----------
def panel1(ax):
    kn = pd.read_csv(f"{JC}/KN_novel_fraction_of_fingerprint.csv")
    g = (kn.groupby("group")["novel_frac_of_top"].first() * 100).sort_values(ascending=True)
    g.to_csv(f"{OUT}/P1_novel_fraction_by_group.csv")
    colors = ["#c0392b" if k == "FOXP2" else "#b0bec5" for k in g.index]
    ax.hlines(y=range(len(g)), xmin=0, xmax=g.values, color=colors, linewidth=2, zorder=1)
    ax.scatter(g.values, range(len(g)), color=colors, s=45, zorder=2)
    med = np.median(g.values)
    ax.axvline(med, ls="--", color="#7f8c8d", lw=1)
    ax.text(med + 1, 0.3, f"median {med:.0f}%", fontsize=8, color="#7f8c8d")
    ax.set_yticks(range(len(g))); ax.set_yticklabels(g.index, fontsize=8)
    for lbl in ax.get_yticklabels():
        if lbl.get_text() == "FOXP2":
            lbl.set_fontweight("bold"); lbl.set_color("#c0392b")
    fx = g["FOXP2"]
    ax.annotate(f"FOXP2  {fx:.1f}%", (fx, list(g.index).index("FOXP2")),
                xytext=(-8, 0), textcoords="offset points", ha="right", va="center",
                fontsize=9, fontweight="bold", color="#c0392b")
    ax.set_xlabel("% of top-variable fingerprint that is NOVEL (ML-predicted)")
    ax.set_title("(1) FOXP2 is the sole novel-dominated group\n"
                 f"next-highest {g.drop('FOXP2').max():.0f}% · median {med:.0f}%",
                 fontsize=11, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)


# ---------- Panel 2: FOXP2 fingerprint composition ----------
def panel2(ax):
    df = pd.read_csv(f"{JD}/FOXP2_joint_master.csv.gz", low_memory=False, dtype={"Chromosome": str})
    top = df.nlargest(TOP, "Diff_pct")
    vc = top["region_class"].value_counts()
    vc.to_csv(f"{OUT}/P2_fingerprint_composition.csv")
    order = ["novel|enhancer", "novel|dual", "known|enhancer", "known|silencer",
             "known|other_regulatory", "known|promoter", "known|gene_body"]
    labels = [c for c in order if c in vc.index]
    sizes = [vc[c] for c in labels]
    cols = [ALLCOL.get(c, "#ccc") for c in labels]
    nov = sum(vc[c] for c in vc.index if c.startswith("novel"))
    w, _, at = ax.pie(sizes, colors=cols, startangle=90, counterclock=False,
                      wedgeprops=dict(width=0.42, edgecolor="white"),
                      autopct=lambda p: f"{p:.0f}%" if p >= 6 else "", pctdistance=0.79,
                      textprops=dict(fontsize=8, fontweight="bold", color="white"))
    ax.text(0, 0.12, f"{nov/len(top)*100:.1f}%", ha="center", fontsize=20, fontweight="bold", color="#c0392b")
    ax.text(0, -0.14, "NOVEL", ha="center", fontsize=11, color="#c0392b")
    ax.legend(w, [f"{l}  ({vc[l]})" for l in labels], loc="center left",
              bbox_to_anchor=(1.0, 0.5), fontsize=8, frameon=False)
    ax.set_title(f"(2) FOXP2 top-{TOP} fingerprint composition\n"
                 "novel|enhancer + novel|dual dominate", fontsize=11, fontweight="bold")


# ---------- Panel 4: cumulative novel-fraction vs rank ----------
def panel4(ax):
    df = pd.read_csv(f"{JD}/FOXP2_joint_master.csv.gz", low_memory=False, dtype={"Chromosome": str})
    ts = df.sort_values("Diff_pct", ascending=False).reset_index(drop=True)
    isnov = (ts["basket"] == "novel").values.astype(float)
    ranks = np.arange(1, len(ts) + 1)
    cum = np.cumsum(isnov) / ranks * 100
    N = 2000
    ax.plot(ranks[:N], cum[:N], color="#c0392b", lw=2)
    pd.DataFrame({"rank": ranks[:N], "cumulative_novel_pct": cum[:N].round(2)}).to_csv(
        f"{OUT}/P4_cumulative_novel_by_rank.csv", index=False)
    ax.axhline(50, ls=":", color="#95a5a6", lw=1)
    for k in (25, 50, 100, 250, 500):
        ax.scatter([k], [cum[k - 1]], color="#2c3e50", s=25, zorder=3)
        ax.annotate(f"top-{k}: {cum[k-1]:.0f}%", (k, cum[k - 1]),
                    xytext=(4, 6), textcoords="offset points", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("region rank by differential presence (Diff%, log)")
    ax.set_ylabel("cumulative % NOVEL")
    ax.set_ylim(0, 80)
    ax.set_title("(4) Novel regions carry the strongest FOXP2 signal\n"
                 "novel% peaks among the most-differential regions", fontsize=11, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)


# ---------- Panel 6: known-only vs novel-only FOXP2 divergence ----------
def _div(df_sub, subs):
    M = df_sub[[f"Pct_{s}" for s in subs]].values  # regions x subtypes
    rho, _ = spearmanr(M)
    if np.isscalar(rho):
        rho = np.array([[1.0, rho], [rho, 1.0]])
    d = 1.0 - rho
    np.fill_diagonal(d, 0.0)
    return d


def panel6(ax_k, ax_n):
    df = pd.read_csv(f"{JD}/FOXP2_joint_master.csv.gz", low_memory=False, dtype={"Chromosome": str})
    subs = sorted([c[4:] for c in df.columns if c.startswith("Pct_")])
    known = df[df.basket == "known"]; novel = df[df.basket == "novel"]
    dk = _div(known, subs); dn = _div(novel, subs)
    pd.DataFrame(dk, index=subs, columns=subs).to_csv(f"{OUT}/P6_divergence_known.csv")
    pd.DataFrame(dn, index=subs, columns=subs).to_csv(f"{OUT}/P6_divergence_novel.csv")
    vmax = max(dk.max(), dn.max(), 0.05)
    for ax, d, ttl, n in [(ax_k, dk, "KNOWN regions only", len(known)),
                          (ax_n, dn, "NOVEL regions only", len(novel))]:
        im = ax.imshow(d, cmap="magma_r", vmin=0, vmax=vmax)
        ax.set_xticks(range(len(subs))); ax.set_xticklabels(subs, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(subs))); ax.set_yticklabels(subs, fontsize=8)
        for i in range(len(subs)):
            for j in range(len(subs)):
                ax.text(j, i, f"{d[i,j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if d[i, j] > vmax * 0.5 else "black")
        ax.set_title(f"{ttl}\n({n:,} regions) mean off-diag={d[np.triu_indices(len(subs),1)].mean():.3f}",
                     fontsize=9, fontweight="bold")
    return dk, dn, ax_n, vmax


def main():
    # combined figure
    fig = plt.figure(figsize=(17, 13))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], width_ratios=[1.1, 1, 1],
                          hspace=0.35, wspace=0.4)
    panel1(fig.add_subplot(gs[0, 0]))
    panel2(fig.add_subplot(gs[0, 1]))
    panel4(fig.add_subplot(gs[0, 2]))
    ak = fig.add_subplot(gs[1, 0]); an = fig.add_subplot(gs[1, 1])
    dk, dn, an, vmax = panel6(ak, an)
    cax = fig.add_subplot(gs[1, 2])
    cax.axis("off")
    im = an.images[0]
    cb = fig.colorbar(im, ax=cax, fraction=0.5, pad=0.05)
    cb.set_label("1 - Spearman divergence")
    cax.text(0.5, 0.5,
             "(6) On KNOWN regions the four FOXP2\nsubtypes are ~identical (div~0);\n"
             "on NOVEL regions they diverge.\nFOXP2 substructure lives in novel elements.",
             ha="center", va="center", fontsize=9, wrap=True, transform=cax.transAxes)
    fig.suptitle("FOXP2 regulatory fingerprint is ~58-66% NOVEL (ML-predicted) — evidence panels",
                 fontsize=15, fontweight="bold", y=0.995)
    savefig(fig, "FOXP2_novel_evidence_combined")

    # individual panels
    for name, fn in [("P1_outlier", panel1), ("P2_composition", panel2), ("P4_rank_curve", panel4)]:
        f, a = plt.subplots(figsize=(7.5, 6) if name != "P2_composition" else (8, 6))
        fn(a); savefig(f, name)
    f, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5.6))
    panel6(a1, a2); f.colorbar(a2.images[0], ax=a2, label="1 - Spearman", fraction=0.046)
    f.suptitle("(6) FOXP2_1-4 divergence: known vs novel regions", fontsize=12, fontweight="bold")
    savefig(f, "P6_known_vs_novel_divergence")
    print(f"wrote panels + CSVs -> {OUT}")


if __name__ == "__main__":
    main()
