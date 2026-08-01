#!/usr/bin/env python3
"""
42_foxp2_final_figure.py — assemble the publication figure for "FOXP2's regulatory
fingerprint is ~58-66% novel", reading the panel source CSVs already produced by
40_foxp2_evidence.py / 41_foxp2_percell_novel.py (no re-computation).

MAIN  (Figure): a outlier | b composition | c rank curve | d known-vs-novel divergence
SUPP  (Figure S): per-cell novel burden (with circularity caveat in caption)

Outputs: FOXP2_novel_evidence/FIG_foxp2_novel_main.{png,pdf}
         FOXP2_novel_evidence/FIGS_foxp2_percell_supp.{png,pdf}
Env: zeros.
"""
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = "/path/to/data/FOXP2_novel_evidence"
ALLCOL = {"novel|enhancer": "#c0392b", "novel|dual": "#e67e22",
          "known|enhancer": "#2980b9", "known|silencer": "#16a085",
          "known|promoter": "#8e44ad", "known|other_regulatory": "#95a5a6",
          "known|gene_body": "#7f8c8d"}
RED, GREY = "#c0392b", "#95a5a6"


def _letter(ax, s):
    ax.text(-0.14, 1.06, s, transform=ax.transAxes, fontsize=16,
            fontweight="bold", va="bottom", ha="left")


def panel_a(ax):
    s = pd.read_csv(f"{D}/P1_novel_fraction_by_group.csv", index_col=0).iloc[:, 0]
    s = s.sort_values()
    colors = [RED if k == "FOXP2" else GREY for k in s.index]
    ax.hlines(range(len(s)), 0, s.values, color=colors, lw=2, zorder=1)
    ax.scatter(s.values, range(len(s)), color=colors, s=42, zorder=2)
    med = float(np.median(s.values))
    ax.axvline(med, ls="--", color="#7f8c8d", lw=1)
    ax.text(med + 1.5, 0.4, f"median {med:.0f}%", fontsize=8, color="#7f8c8d")
    ax.set_yticks(range(len(s))); ax.set_yticklabels(s.index, fontsize=8)
    for lbl in ax.get_yticklabels():
        if lbl.get_text() == "FOXP2":
            lbl.set_fontweight("bold"); lbl.set_color(RED)
    fx = s["FOXP2"]
    ax.annotate(f"FOXP2 {fx:.1f}%", (fx, list(s.index).index("FOXP2")),
                xytext=(-8, 0), textcoords="offset points", ha="right", va="center",
                fontsize=9, fontweight="bold", color=RED)
    ax.set_xlabel("% of top-variable fingerprint that is NOVEL", fontsize=9)
    ax.set_title("FOXP2 is the sole novel-dominated group", fontsize=10, fontweight="bold")
    ax.set_xlim(-2, 66); ax.spines[["top", "right"]].set_visible(False)


def panel_b(ax):
    vc = pd.read_csv(f"{D}/P2_fingerprint_composition.csv", index_col=0).iloc[:, 0]
    order = ["novel|enhancer", "novel|dual", "known|enhancer", "known|silencer",
             "known|other_regulatory", "known|promoter", "known|gene_body"]
    labels = [c for c in order if c in vc.index]
    sizes = [vc[c] for c in labels]; cols = [ALLCOL.get(c, "#ccc") for c in labels]
    tot = sum(vc.values); nov = sum(vc[c] for c in vc.index if str(c).startswith("novel"))
    w, _ = ax.pie(sizes, colors=cols, startangle=90, counterclock=False,
                  wedgeprops=dict(width=0.42, edgecolor="white"))
    ax.text(0, 0.10, f"{nov/tot*100:.1f}%", ha="center", fontsize=18, fontweight="bold", color=RED)
    ax.text(0, -0.16, "NOVEL", ha="center", fontsize=10, color=RED)
    ax.legend(w, [f"{l.replace('|',' | ')} ({vc[l]})" for l in labels],
              loc="upper center", bbox_to_anchor=(0.5, -0.02), fontsize=7.5,
              frameon=False, ncol=2, handlelength=1.2, columnspacing=1.0)
    ax.set_title("FOXP2 top-500 fingerprint composition", fontsize=10, fontweight="bold")


def panel_c(ax):
    df = pd.read_csv(f"{D}/P4_cumulative_novel_by_rank.csv")
    ax.plot(df["rank"], df["cumulative_novel_pct"], color=RED, lw=2)
    ax.axhline(50, ls=":", color="#95a5a6", lw=1)
    cum = df.set_index("rank")["cumulative_novel_pct"]
    for k in (25, 50, 100, 250, 500):
        if k in cum.index:
            ax.scatter([k], [cum[k]], color="#2c3e50", s=22, zorder=3)
            ax.annotate(f"top-{k}: {cum[k]:.0f}%", (k, cum[k]), xytext=(4, 6),
                        textcoords="offset points", fontsize=7.5)
    ax.set_xscale("log"); ax.set_ylim(0, 80)
    ax.set_xlabel("region rank by differential presence (log)", fontsize=9)
    ax.set_ylabel("cumulative % NOVEL", fontsize=9)
    ax.set_title("Novel regions carry the strongest signal", fontsize=10, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)


def panel_d(ax_k, ax_n, cax):
    dk = pd.read_csv(f"{D}/P6_divergence_known.csv", index_col=0)
    dn = pd.read_csv(f"{D}/P6_divergence_novel.csv", index_col=0)
    subs = list(dk.index)
    vmax = max(dk.values.max(), dn.values.max())
    iu = np.triu_indices(len(subs), 1)
    im = None
    for ax, d, ttl in [(ax_k, dk, "KNOWN regions only"), (ax_n, dn, "NOVEL regions only")]:
        v = d.values
        im = ax.imshow(v, cmap="magma_r", vmin=0, vmax=vmax)
        ax.set_xticks(range(len(subs))); ax.set_xticklabels(subs, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(subs))); ax.set_yticklabels(subs, fontsize=8)
        for i in range(len(subs)):
            for j in range(len(subs)):
                ax.text(j, i, f"{v[i,j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if v[i, j] > vmax * 0.5 else "black")
        ax.set_title(f"{ttl}\nmean div = {v[iu].mean():.3f}", fontsize=9, fontweight="bold")
    cb = plt.colorbar(im, cax=cax); cb.set_label("1 − Spearman divergence", fontsize=8)
    cb.ax.tick_params(labelsize=7)


def main_figure():
    fig = plt.figure(figsize=(15.5, 11))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.05], width_ratios=[1.15, 1, 1],
                          hspace=0.42, wspace=0.42, left=0.07, right=0.95, top=0.9, bottom=0.08)
    a = fig.add_subplot(gs[0, 0]); panel_a(a); _letter(a, "a")
    b = fig.add_subplot(gs[0, 1]); panel_b(b); _letter(b, "b")
    c = fig.add_subplot(gs[0, 2]); panel_c(c); _letter(c, "c")
    # bottom row: two heatmaps + slim colorbar
    gsb = gs[1, :].subgridspec(1, 3, width_ratios=[1, 1, 0.06], wspace=0.35)
    ak = fig.add_subplot(gsb[0, 0]); an = fig.add_subplot(gsb[0, 1]); cax = fig.add_subplot(gsb[0, 2])
    panel_d(ak, an, cax); _letter(ak, "d")
    an.text(1.24, -0.28,
            "FOXP2_1–4 diverge ~1.9× more on novel regions (0.23→0.42):\n"
            "the subtypes' distinguishing signal lives in novel elements.",
            transform=an.transAxes, fontsize=8.5, style="italic", ha="center", va="top")
    fig.suptitle("FOXP2's regulatory fingerprint is ~58–66% novel (ML-predicted), not known regulatory elements",
                 fontsize=15, fontweight="bold", y=0.965)
    for e in ("png", "pdf"):
        fig.savefig(f"{D}/FIG_foxp2_novel_main.{e}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def supp_figure():
    df = pd.read_csv(f"{D}/P7_percell.csv")
    order = []
    for st in ["FOXP2_1", "FOXP2_2", "FOXP2_3", "FOXP2_4", "MSN_1", "PVALB_1",
               "CNGA_1", "VIP_1", "D1Pu"]:
        if st in set(df.subtype):
            order.append(st)
    data = [df[df.subtype == st]["novel_frac"].values for st in order]
    isfox = [st.startswith("FOXP2") for st in order]
    fig, ax = plt.subplots(figsize=(11, 6))
    parts = ax.violinplot(data, showmedians=True, showextrema=False, widths=0.85)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(RED if isfox[i] else GREY); pc.set_alpha(0.75)
    parts["cmedians"].set_color("black")
    for i, d in enumerate(data):
        ax.text(i + 1, np.median(d) + 2, f"{np.median(d):.0f}%", ha="center",
                fontsize=8, fontweight="bold")
    other = np.concatenate([data[i] for i in range(len(order)) if not isfox[i]])
    ax.axhline(np.median(other), ls="--", color="#7f8c8d", lw=1)
    ax.set_xticks(range(1, len(order) + 1)); ax.set_xticklabels(order, rotation=45, ha="right", fontsize=9)
    for i, lbl in enumerate(ax.get_xticklabels()):
        if isfox[i]:
            lbl.set_color(RED); lbl.set_fontweight("bold")
    ax.set_ylabel("% of cell's regulatory peaks at NOVEL loci", fontsize=10)
    foxm = np.median(np.concatenate([data[i] for i in range(len(order)) if isfox[i]]))
    ax.set_title("Figure S — Per-cell novel regulatory burden (single-cell)\n"
                 f"FOXP2 cells highest (median {foxm:.0f}%, within the 58–66% range)",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(0, 100); ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.5, -0.06,
             "Caveat: novel loci are called from the cells' own open chromatin, so per-cell overlap is partly circular; "
             "this inflates the novel fraction across all neurons and compresses the FOXP2 lead. FOXP2 is highest and lands\n"
             "in the 58–66% range, corroborating the aggregate metric, but the dramatic outlier status is specific to the "
             "distinguishing fingerprint (main Fig. a). Treat as supporting, not primary, evidence.",
             ha="center", va="top", fontsize=8, style="italic", wrap=True)
    for e in ("png", "pdf"):
        fig.savefig(f"{D}/FIGS_foxp2_percell_supp.{e}", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main_figure()
    supp_figure()
    print(f"wrote FIG_foxp2_novel_main.* and FIGS_foxp2_percell_supp.* -> {D}")
