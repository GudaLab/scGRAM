#!/usr/bin/env python3
"""
Subtype-resolution enhancer<->RNA validation for the TOP DA hits.
Drops below supercluster to the atlas cluster_id level within the DA superclusters
(Medium spiny neuron, Midbrain-derived inhibitory) + cortical-interneuron reference
(CGE, MGE), so D1- vs D2-MSN and midbrain-DA anatomical types are resolved.

For a focused gene panel (top DA-specific + DA-absent target genes + canonical
identity markers), computes per-cluster pseudobulk (log1p CP10K) and labels each
cluster by markers (D1/D2/eccentric MSN) and dominant dissection (midbrain types).

Outputs -> DA_vs_nonDA/rna_validation/subtype/:
  cluster_expr_matrix.csv         focus genes x labelled clusters
  subtype_expr_heatmap.png        top hits + markers across DA subtypes vs interneurons
  d1_vs_d2_top_hits.csv           D1-MSN vs D2-MSN mean expression for top hits
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
MARKERS = ["DRD1", "DRD2", "DRD3", "TH", "SLC6A3", "PPP1R1B", "PENK", "TAC1",
           "PDYN", "ADORA2A", "FOXP2", "MEIS2", "GAD1", "GAD2", "SLC17A7"]
FILES = {  # short label -> (glob, supercluster group)
    "MSN":  ("Supercluster_Medium_spiny_neuron.h5ad", "DA"),
    "MidbrDA": ("Supercluster_Midbrain-derived_inhibitory.h5ad", "DA"),
    "CGE":  ("Supercluster_CGE_interneuron.h5ad", "ref"),
    "MGE":  ("Supercluster_MGE*interneuron.h5ad", "ref"),
}


def log(m): print(m, flush=True)


def top_genes():
    r = pd.read_csv(f"{D}/DA_vs_nonDA/rna_validation/candidate_gene_rna.csv")
    spec = r[r.direction == "specific"].reindex(
        r[r.direction == "specific"].atac_delta_pct.abs().sort_values(ascending=False).index).head(18)["gene"].tolist()
    absent = r[r.direction == "absent"]["gene"].tolist()
    return spec, absent


def pseudobulk_file(path, sc_short, genes):
    a = ad.read_h5ad(path, backed="r")
    fn = a.var["feature_name"].astype(str).values
    sym2col = {}
    for i, s in enumerate(fn):
        sym2col.setdefault(s, i)
    cols, found = [], []
    for g in genes:
        if g in sym2col:
            cols.append(sym2col[g]); found.append(g)
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
            mask = cc == c
            key = f"{sc_short}:{c}"
            sums[key] = sums.get(key, 0) + X[mask].sum(0)
            cnts[key] = cnts.get(key, 0) + int(mask.sum())
            disc.setdefault(key, {})
            for dv in dd[mask]:
                disc[key][dv] = disc[key].get(dv, 0) + 1
    rows = {}
    meta = {}
    for k in sums:
        rows[k] = sums[k] / max(cnts[k], 1)
        top_diss = max(disc[k].items(), key=lambda x: x[1])[0]
        meta[k] = {"supercluster": sc_short, "n_cells": cnts[k], "top_dissection": top_diss}
    expr = pd.DataFrame(rows, index=found).T   # clusters x genes
    return expr, pd.DataFrame(meta).T


def label_cluster(row, expr):
    sc = row["supercluster"]
    if sc == "MSN":
        d1 = expr.get("DRD1", np.nan); d2 = expr.get("DRD2", np.nan)
        if pd.notna(d1) and pd.notna(d2):
            if d1 > d2 + 0.1:
                return "D1-MSN"
            if d2 > d1 + 0.1:
                return "D2-MSN"
            return "MSN-other/eccentric"
        return "MSN"
    if sc == "MidbrDA":
        dd = str(row["top_dissection"]).lower()
        for key, lab in [("collicul", "IC-DA"), ("septal", "Septal-DA"),
                         ("mediodorsal", "MD-DA"), ("midbrain", "Midbrain-DA"),
                         ("substantia", "SN-DA"), ("tegment", "VTA/tegm-DA")]:
            if key in dd:
                return lab
        return "Midbrain-DA"
    return sc  # CGE / MGE


def main():
    os.makedirs(OUT, exist_ok=True)
    spec, absent = top_genes()
    genes = spec + absent + MARKERS
    exprs, metas = [], []
    for short, (pat, grp) in FILES.items():
        hits = glob.glob(f"{T}/{pat}")
        if not hits:
            log(f"  WARN no file for {short} ({pat})"); continue
        log(f"  pseudobulking {short}: {os.path.basename(hits[0])}")
        e, m = pseudobulk_file(hits[0], short, genes)
        exprs.append(e); metas.append(m)
    E = pd.concat(exprs).fillna(0.0)        # clusters x genes
    M = pd.concat(metas)
    M["label"] = [label_cluster(M.loc[k], E.loc[k]) for k in M.index]
    # drop tiny clusters
    keep = M[M.n_cells >= 50].index
    E, M = E.loc[keep], M.loc[keep]
    E.join(M).to_csv(f"{OUT}/cluster_expr_matrix.csv")
    log(f"  {len(E)} clusters (>=50 cells): " + ", ".join(sorted(M.label.unique())))

    # order clusters: D1-MSN, D2-MSN, MSN-other, midbrain-DA types, CGE, MGE
    order_lab = ["D1-MSN", "D2-MSN", "MSN-other/eccentric", "MSN",
                 "IC-DA", "SIGA-DA", "Septal-DA", "MD-DA", "Midbrain-DA", "SN-DA", "VTA/tegm-DA",
                 "CGE", "MGE"]
    M["ord"] = M["label"].apply(lambda l: order_lab.index(l) if l in order_lab else 99)
    cl_order = M.sort_values(["ord", "n_cells"], ascending=[True, False]).index.tolist()
    da_labels = {"D1-MSN", "D2-MSN", "MSN-other/eccentric", "MSN", "IC-DA", "SIGA-DA",
                 "Septal-DA", "MD-DA", "Midbrain-DA", "SN-DA", "VTA/tegm-DA"}
    n_da = sum(M.loc[c, "label"] in da_labels for c in cl_order)

    # gene rows: specific, absent, markers
    grow = [g for g in spec if g in E.columns] + [g for g in absent if g in E.columns] + \
           [g for g in MARKERS if g in E.columns]
    Z = E[grow].T                      # genes x clusters
    Zz = Z.sub(Z.mean(1), 0).div(Z.std(1).replace(0, 1), 0)
    Zz = Zz[cl_order]
    fig, ax = plt.subplots(figsize=(max(9, len(cl_order)*0.34), max(7, len(grow)*0.22+1)))
    im = ax.imshow(Zz.values, cmap="RdBu_r", aspect="auto", vmin=-2, vmax=2)
    ax.set_xticks(range(len(cl_order)))
    ax.set_xticklabels([f"{M.loc[c,'label']} [{c.split(':')[0]}:{c.split(':')[1]}] n={int(M.loc[c,'n_cells'])}"
                        for c in cl_order], rotation=90, fontsize=6)
    for k, c in enumerate(cl_order):
        ax.get_xticklabels()[k].set_color("#e6550d" if M.loc[c, "label"] in da_labels else "#3182bd")
    ax.set_yticks(range(len(grow))); ax.set_yticklabels(grow, fontsize=6.5)
    ax.axvline(n_da-0.5, color="black", lw=2)
    ax.axhline(len(spec)-0.5, color="k", lw=1); ax.axhline(len(spec)+len(absent)-0.5, color="k", lw=1)
    cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01); cb.set_label("z(expr) per gene", fontsize=8)
    ax.set_title("Subtype-resolved RNA of top DA-hit genes (DA clusters = orange labels | interneuron ref = blue)\n"
                 "rows: DA-specific hits / DA-absent hits / identity markers (black lines); black column line = DA | reference",
                 fontsize=10)
    fig.tight_layout(); fig.savefig(f"{OUT}/subtype_expr_heatmap.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # D1 vs D2 table for top hits
    d1 = [c for c in cl_order if M.loc[c, "label"] == "D1-MSN"]
    d2 = [c for c in cl_order if M.loc[c, "label"] == "D2-MSN"]
    if d1 and d2:
        tab = pd.DataFrame({
            "gene": grow,
            "D1_MSN": E.loc[d1, grow].mean().round(3).values,
            "D2_MSN": E.loc[d2, grow].mean().round(3).values,
        })
        tab["log2FC_D1_vs_D2"] = np.log2((tab.D1_MSN+1e-3)/(tab.D2_MSN+1e-3)).round(3)
        tab.to_csv(f"{OUT}/d1_vs_d2_top_hits.csv", index=False)
        log("D1 vs D2 (top hits):\n" + tab.to_string(index=False))
    log("SUBTYPE RESOLUTION COMPLETE -> " + OUT)


if __name__ == "__main__":
    main()
