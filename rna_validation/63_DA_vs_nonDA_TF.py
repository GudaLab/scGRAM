#!/usr/bin/env python3
"""
TF-level DA-vs-non-DA contrast (Track B, TF companion to 62_DA_vs_nonDA.py).
Finds transcription factors whose footprint activity is UNIQUELY HIGH in the
dopaminergic (DA) subtypes and those MISSING/DEPLETED in DA vs the rest of CSPMEI.

Per-subtype TF metric = `occupancy` (fraction of nuclei with a footprint > 0),
read from the pipeline's own cache Joint_TF_Divergence/per_subtype/<sub>.csv.gz
(879 TFs x 55 subtypes). Two reference sets (both):
  - GABAergic non-DA (21)   - ALL non-DA (41)

Outputs -> DA_vs_nonDA_TF/<contrast>/:
  DA_specific_TFs.csv   higher footprint occupancy in DA (delta>0, ranked; FDR)
  DA_absent_TFs.csv     lower  footprint occupancy in DA (delta<0, ranked; FDR)
  DA_vs_nonDA_TF_scatter.png
  DA_specific_TF_heatmap.png / DA_absent_TF_heatmap.png
  summary.txt
New dirs only; nothing overwritten. Env: zeros.
"""
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.colors as mcolors
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
import da_config as C

BASE = "/path/to/data"
PER = f"{BASE}/Joint_TF_Divergence/per_subtype"
OUT = f"{BASE}/DA_vs_nonDA_TF"
TOPN = 40
CLASS_COL = {"DA": "#e6550d", "GABA_nonDA": "#3182bd", "GLUT": "#31a354", "NonN": "#756bb1"}


def log(m): print(m, flush=True)


def build_matrix():
    """Primary metric = DOMINANCE = (n_TF - Mean_Rank): higher-ranked (more active)
    TFs within nuclei get a higher score. occupancy saturates (85-99% everywhere)
    so it cannot discriminate; within-nucleus rank has full dynamic range.
    Returns (dominance motif x subtype, occupancy% motif x subtype, tf_short)."""
    need = C.DA_SUBTYPES + C.ALL_NONDA_SUBTYPES
    dom, occ, short = {}, {}, {}
    for s in need:
        f = f"{PER}/{s}.csv.gz"
        d = pd.read_csv(f).set_index("motif")
        n_tf = len(d)
        dom[s] = (n_tf - d["Mean_Rank"]).astype("float32")     # higher = more dominant/active
        occ[s] = (d["occupancy"] * 100).astype("float32")
        if not short:
            short.update(dict(zip(d.index, d["TF"])))
    M = pd.concat(dom, axis=1).dropna(how="any")
    O = pd.concat(occ, axis=1).reindex(M.index)
    tf_short = pd.Series(short).reindex(M.index)
    log(f"  TF matrix: {M.shape[0]} TFs x {M.shape[1]} subtypes (metric=dominance)")
    return M, O, tf_short


def contrast(M, O, tf_short, ref_subs, name):
    outdir = f"{OUT}/{name}"; os.makedirs(outdir, exist_ok=True)
    da_cols = [s for s in C.DA_SUBTYPES if s in M.columns]
    ref_cols = [s for s in ref_subs if s in M.columns]
    DA = M[da_cols].values; REF = M[ref_cols].values
    da_mean, ref_mean = DA.mean(1), REF.mean(1)
    delta = da_mean - ref_mean
    da_occ = O[da_cols].values.mean(1); ref_occ = O[ref_cols].values.mean(1)
    ps = np.ones(M.shape[0])
    for i in range(M.shape[0]):
        try:
            _, ps[i] = mannwhitneyu(DA[i], REF[i], alternative="two-sided")
        except ValueError:
            ps[i] = 1.0
    fdr = multipletests(ps, method="fdr_bh")[1]
    tab = pd.DataFrame({
        "TF": tf_short.values, "motif": M.index,
        "DA_dominance": da_mean.round(2), "ref_dominance": ref_mean.round(2),
        "delta_dominance": delta.round(2),
        "DA_occ_pct": da_occ.round(1), "ref_occ_pct": ref_occ.round(1),
        "mwu_p": ps, "mwu_FDR": fdr.round(5),
    })
    spec = tab[tab.delta_dominance > 0].sort_values("delta_dominance", ascending=False)
    absent = tab[tab.delta_dominance < 0].sort_values("delta_dominance", ascending=True)
    spec.to_csv(f"{outdir}/DA_specific_TFs.csv", index=False)
    absent.to_csv(f"{outdir}/DA_absent_TFs.csv", index=False)

    scatter(da_mean, ref_mean, fdr, delta, name, outdir, len(ref_cols))
    heatmap(M, tf_short, spec.head(TOPN)["motif"].tolist(), da_cols, ref_cols,
            f"DA-SPECIFIC TFs — more dominant footprint in dopaminergic (vs {name})",
            f"{outdir}/DA_specific_TF_heatmap.png")
    heatmap(M, tf_short, absent.head(TOPN)["motif"].tolist(), da_cols, ref_cols,
            f"DA-DEPLETED TFs — less dominant footprint in dopaminergic (vs {name})",
            f"{outdir}/DA_absent_TF_heatmap.png")
    n_sig_up = int(((tab.delta_dominance > 0) & (tab.mwu_FDR < 0.05)).sum())
    n_sig_dn = int(((tab.delta_dominance < 0) & (tab.mwu_FDR < 0.05)).sum())
    with open(f"{outdir}/summary.txt", "w") as fh:
        fh.write(f"DA vs {name} — TF footprint DOMINANCE contrast\n"
                 f"DA ({len(da_cols)}) vs reference ({len(ref_cols)}); {M.shape[0]} TFs\n"
                 f"metric: dominance = (n_TF - within-nucleus Mean_Rank); higher = more active.\n"
                 f"(occupancy saturates 85-99% and cannot discriminate; reported as context.)\n"
                 f"DA-higher (FDR<0.05): {n_sig_up} | DA-lower (FDR<0.05): {n_sig_dn}\n")
        for lab, tb in [("DA-SPECIFIC (more dominant in DA)", spec), ("DA-DEPLETED (less dominant in DA)", absent)]:
            fh.write(f"\ntop 15 {lab}  (TF / DA_dom / ref_dom / delta / FDR):\n")
            for _, r in tb.head(15).iterrows():
                fh.write(f"  {r.TF:16s} {r.DA_dominance:7.1f} {r.ref_dominance:7.1f} "
                         f"{r.delta_dominance:+8.1f}  FDR={r.mwu_FDR:.3g}\n")
    log(f"[{name}] DA-higher(FDR<.05)={n_sig_up}  DA-lower(FDR<.05)={n_sig_dn}  -> {outdir}")
    return {"contrast": name, "DA_higher_FDR05": n_sig_up, "DA_lower_FDR05": n_sig_dn}


def scatter(da_mean, ref_mean, fdr, delta, name, outdir, n_ref):
    fig, ax = plt.subplots(figsize=(6.5, 6.2))
    sig = fdr < 0.05
    up = sig & (delta > 0); dn = sig & (delta < 0); other = ~sig
    ax.scatter(ref_mean[other], da_mean[other], s=6, c="#cccccc", alpha=0.5, label="n.s.")
    ax.scatter(ref_mean[up], da_mean[up], s=18, c="#e6550d", label=f"DA-higher ({up.sum()})")
    ax.scatter(ref_mean[dn], da_mean[dn], s=18, c="#3182bd", label=f"DA-lower ({dn.sum()})")
    lo = min(da_mean.min(), ref_mean.min()); hi = max(da_mean.max(), ref_mean.max())
    ax.plot([lo, hi], [lo, hi], "--", c="k", lw=0.7)
    ax.set_xlabel(f"mean TF occupancy in reference ({name}, n={n_ref}) [%]")
    ax.set_ylabel("mean TF occupancy in DA subtypes [%]")
    ax.set_title(f"DA vs {name}: per-TF footprint occupancy")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(f"{outdir}/DA_vs_nonDA_TF_scatter.png", dpi=170); plt.close(fig)


def heatmap(M, tf_short, motifs, da_cols, ref_cols, title, path):
    if not motifs:
        return
    cols = da_cols + ref_cols
    H = M.loc[motifs, cols].values
    labels = list(tf_short.reindex(motifs).values)
    n_r, n_c = H.shape; n_da = len(da_cols)
    row_h, col_w = 0.2, 0.24
    left_in, cbar_in, top_in, bottom_in = 1.9, 1.4, 2.0, 2.0
    fig_w = left_in + n_c*col_w + cbar_in
    fig_h = top_in + 0.5 + n_r*row_h + bottom_in
    fig = plt.figure(figsize=(fig_w, fig_h))
    L, R = left_in/fig_w, (left_in+n_c*col_w)/fig_w
    T, Bo = 1-top_in/fig_h, bottom_in/fig_h
    gs = fig.add_gridspec(2, 1, height_ratios=[0.5, n_r*row_h], hspace=0.02,
                          left=L, right=R, top=T, bottom=Bo)
    ax = fig.add_subplot(gs[1, 0]); ax_top = fig.add_subplot(gs[0, 0])
    ax_top.set_xlim(-0.5, n_c-0.5)
    vmax = np.percentile(H, 98)
    im = ax.imshow(H, cmap="viridis", aspect="auto", vmin=np.percentile(H, 2), vmax=vmax)
    strip = np.array([[mcolors.to_rgb(CLASS_COL[C.SUBTYPE_CLASS[c]]) for c in cols]])
    ax_top.imshow(strip, aspect="auto"); ax_top.set_xticks([]); ax_top.set_yticks([])
    ax.set_yticks(range(n_r)); ax.set_yticklabels(labels, fontsize=6)
    ax.set_xticks(range(n_c)); ax.set_xticklabels(cols, rotation=90, fontsize=7)
    for tk, c in zip(ax.get_xticklabels(), cols):
        tk.set_color(CLASS_COL[C.SUBTYPE_CLASS[c]])
    ax.axvline(n_da-0.5, color="white", lw=2.5); ax_top.axvline(n_da-0.5, color="black", lw=2)
    ax_top.text((n_da-1)/2, -0.9, "DA", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax_top.text((n_da+n_c-1)/2, -0.9, "reference", ha="center", va="bottom", fontsize=9, fontweight="bold")
    cax = fig.add_axes([R + 0.35/fig_w, Bo + 0.3*(T-Bo), 0.16/fig_w, 0.34*(T-Bo)])
    cb = fig.colorbar(im, cax=cax); cb.set_label("TF dominance (higher = more active)", fontsize=9)
    fig.suptitle(title + f"\n{n_r} TFs | dominance = n_TF - mean within-nucleus rank | x-labels by class | white line = DA | reference",
                 fontsize=11, fontweight="bold", y=1 - 0.4/fig_h)
    classes = sorted({C.SUBTYPE_CLASS[c] for c in cols}, key=["DA", "GABA_nonDA", "GLUT", "NonN"].index)
    fig.legend(handles=[Patch(color=CLASS_COL[k], label=k) for k in classes],
               loc="lower center", ncol=len(classes), fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, 0.4/fig_h))
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.3); plt.close(fig)
    log(f"  wrote {path}")


def main():
    os.makedirs(OUT, exist_ok=True)
    M, O, tf_short = build_matrix()
    res = [contrast(M, O, tf_short, C.GABA_NONDA_SUBTYPES, "GABAergic_nonDA"),
           contrast(M, O, tf_short, C.ALL_NONDA_SUBTYPES, "ALL_nonDA")]
    pd.DataFrame(res).to_csv(f"{OUT}/contrast_counts.csv", index=False)
    log("DA_vs_nonDA_TF COMPLETE")


if __name__ == "__main__":
    main()
