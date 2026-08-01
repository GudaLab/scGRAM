#!/usr/bin/env python3
"""
Subtype-resolution heatmap of ONLY the DA-specific enhancers whose target gene is
RNA-concordant (higher in DA superclusters) — the 73/120 concordant set.
Per-cluster pseudobulk (log1p CP10K) across DA superclusters (MSN, Midbrain-derived
inhibitory) + interneuron reference (CGE, MGE); rows ordered by ATAC DA-specificity.
Output -> DA_vs_nonDA/rna_validation/subtype/concordant_DAspecific_subtype_heatmap.png
Env: snapatac2_env.
"""
import os, glob
import numpy as np, pandas as pd
import anndata as ad
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = "/path/to/data"
T = f"{D}/unbound_characetrize/transcriptome"
OUT = f"{D}/DA_vs_nonDA/rna_validation/subtype"
CHUNK = 50_000
MARKERS = ["DRD1", "DRD2", "TH", "PPP1R1B", "GAD1", "SLC17A7"]   # orientation only
FILES = {"MSN": ("Supercluster_Medium_spiny_neuron.h5ad", "DA"),
         "MidbrDA": ("Supercluster_Midbrain-derived_inhibitory.h5ad", "DA"),
         "CGE": ("Supercluster_CGE_interneuron.h5ad", "ref"),
         "MGE": ("Supercluster_MGE*interneuron.h5ad", "ref")}


def log(m): print(m, flush=True)


def pseudobulk_file(path, sc_short, genes):
    a = ad.read_h5ad(path, backed="r")
    fn = a.var["feature_name"].astype(str).values
    sym2col = {}
    for i, s in enumerate(fn):
        sym2col.setdefault(s, i)
    cols = [sym2col[g] for g in genes if g in sym2col]
    found = [g for g in genes if g in sym2col]
    cid = a.obs["cluster_id"].astype(str).values
    diss = a.obs["dissection"].astype(str).values
    umis = a.obs["total_UMIs"].values.astype(np.float64)
    sums, cnts, disc = {}, {}, {}
    n = a.n_obs
    for st in range(0, n, CHUNK):
        en = min(st + CHUNK, n)
        X = np.asarray(a[st:en].X[:, cols].todense(), dtype=np.float64)
        lib = umis[st:en][:, None]; lib[lib == 0] = 1
        X = np.log1p(X / lib * 1e4)
        cc = cid[st:en]; dd = diss[st:en]
        for c in np.unique(cc):
            mask = cc == c; key = f"{sc_short}:{c}"
            sums[key] = sums.get(key, 0) + X[mask].sum(0)
            cnts[key] = cnts.get(key, 0) + int(mask.sum())
            disc.setdefault(key, {})
            for dv in dd[mask]:
                disc[key][dv] = disc[key].get(dv, 0) + 1
    rows, meta = {}, {}
    for k in sums:
        rows[k] = sums[k] / max(cnts[k], 1)
        meta[k] = {"supercluster": sc_short, "n_cells": cnts[k],
                   "top_dissection": max(disc[k].items(), key=lambda x: x[1])[0]}
    return pd.DataFrame(rows, index=found).T, pd.DataFrame(meta).T


def label_cluster(row, expr):
    sc = row["supercluster"]
    if sc == "MSN":
        d1, d2 = expr.get("DRD1", np.nan), expr.get("DRD2", np.nan)
        if pd.notna(d1) and pd.notna(d2):
            if d1 > d2 + 0.1: return "D1-MSN"
            if d2 > d1 + 0.1: return "D2-MSN"
            return "MSN-other/eccentric"
        return "MSN"
    if sc == "MidbrDA":
        dd = str(row["top_dissection"]).lower()
        for key, lab in [("collicul", "IC-DA"), ("septal", "Septal-DA"),
                         ("mediodorsal", "MD-DA"), ("midbrain", "Midbrain-DA"),
                         ("substantia", "SN-DA"), ("tegment", "VTA/tegm-DA")]:
            if key in dd: return lab
        return "Midbrain-DA"
    return sc


def main():
    os.makedirs(OUT, exist_ok=True)
    r = pd.read_csv(f"{D}/DA_vs_nonDA/rna_validation/candidate_gene_rna.csv")
    conc = r[(r.direction == "specific") & (r.concordant)].sort_values(
        "atac_delta_pct", ascending=False)
    genes = list(dict.fromkeys(conc.gene.tolist() + MARKERS))
    log(f"concordant DA-specific genes: {len(conc)}")
    exprs, metas = [], []
    for short, (pat, _) in FILES.items():
        hit = glob.glob(f"{T}/{pat}")[0]
        log(f"  pseudobulking {short}: {os.path.basename(hit)}")
        e, m = pseudobulk_file(hit, short, genes)
        exprs.append(e); metas.append(m)
    E = pd.concat(exprs).fillna(0.0); M = pd.concat(metas)
    M["label"] = [label_cluster(M.loc[k], E.loc[k]) for k in M.index]
    keep = M[M.n_cells >= 50].index
    E, M = E.loc[keep], M.loc[keep]

    order_lab = ["D1-MSN", "D2-MSN", "MSN-other/eccentric", "MSN", "IC-DA",
                 "Midbrain-DA", "Septal-DA", "MD-DA", "CGE", "MGE"]
    da_labels = {"D1-MSN", "D2-MSN", "MSN-other/eccentric", "MSN", "IC-DA",
                 "Midbrain-DA", "Septal-DA", "MD-DA", "SN-DA", "VTA/tegm-DA"}
    M["ord"] = M["label"].apply(lambda l: order_lab.index(l) if l in order_lab else 99)
    cl = M.sort_values(["ord", "n_cells"], ascending=[True, False]).index.tolist()
    n_da = sum(M.loc[c, "label"] in da_labels for c in cl)

    g_conc = [g for g in conc.gene if g in E.columns]
    g_mk = [g for g in MARKERS if g in E.columns]
    grow = g_conc + g_mk
    Z = E[grow].T
    Zz = Z.sub(Z.mean(1), 0).div(Z.std(1).replace(0, 1), 0)[cl]
    fig, ax = plt.subplots(figsize=(max(10, len(cl)*0.34), max(9, len(grow)*0.2+1.5)))
    im = ax.imshow(Zz.values, cmap="RdBu_r", aspect="auto", vmin=-2, vmax=2)
    ax.set_xticks(range(len(cl)))
    ax.set_xticklabels([f"{M.loc[c,'label']} [{c}] n={int(M.loc[c,'n_cells'])}" for c in cl],
                       rotation=90, fontsize=6)
    for k, c in enumerate(cl):
        ax.get_xticklabels()[k].set_color("#e6550d" if M.loc[c, "label"] in da_labels else "#3182bd")
    ax.set_yticks(range(len(grow))); ax.set_yticklabels(grow, fontsize=6)
    ax.axvline(n_da-0.5, color="black", lw=2)
    ax.axhline(len(g_conc)-0.5, color="k", lw=1.2)
    cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01); cb.set_label("z(expr) per gene", fontsize=8)
    ax.set_title(f"Concordant DA-specific enhancer target genes (n={len(g_conc)}) — subtype-resolved RNA\n"
                 "DA clusters = orange labels (left of black line) | interneuron ref = blue | "
                 "genes ordered by ATAC DA-specificity; last 6 rows = identity markers",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{OUT}/concordant_DAspecific_subtype_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    E[grow].join(M[["label", "supercluster", "n_cells", "top_dissection"]]).to_csv(
        f"{OUT}/concordant_DAspecific_cluster_expr.csv")
    log(f"wrote concordant_DAspecific_subtype_heatmap.png ({len(g_conc)} genes x {len(cl)} clusters)")


if __name__ == "__main__":
    main()
