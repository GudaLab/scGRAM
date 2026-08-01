#!/usr/bin/env python3
"""
Top-25 DA-enriched SILENCERS -> subtype-resolved RNA validation (silencer analog of
the concordant DA-specific enhancer heatmap). For silencers the functional prediction
is INVERTED: a silencer more accessible in DA should REPRESS its target, so concordance
= target RNA LOWER in DA superclusters (negative log2FC).

Ranks all informative silencer regions by DA-specificity (DA - GABAergic-nonDA presence),
takes top 25, annotates target genes (gene.gtf), pseudobulks per atlas cluster, and
plots genes x DA/interneuron clusters.
Outputs -> DA_vs_nonDA/rna_validation/subtype/:
  top25_silencers.csv, silencer_subtype_heatmap.png
Env: snapatac2_env.
"""
import os, glob, importlib.util
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
import da_config as C

D = "/path/to/data"
OUT = f"{D}/DA_vs_nonDA/rna_validation/subtype"
MARKERS = ["DRD1", "DRD2", "TH", "PPP1R1B", "GAD1", "SLC17A7"]
TOPN = 25


def load(path):
    s = importlib.util.spec_from_file_location("m", path)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


def log(m): print(m, flush=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    m62 = load(f"{D}/62_DA_vs_nonDA.py")
    m64 = load(f"{D}/64_annotate_DA_regions.py")
    m67 = load(f"{D}/67_concordant_subtype_heatmap.py")

    # 1. region x subtype presence + rank silencers by DA-specificity (vs GABAergic non-DA)
    M, meta = m62.build_matrix()
    sil = meta.region_class == "silencer"
    da = M[[s for s in C.DA_SUBTYPES if s in M.columns]].values
    ref = M[[s for s in C.GABA_NONDA_SUBTYPES if s in M.columns]].values
    dfr = pd.DataFrame({"coord": meta["coord"].values,
                        "DA_mean_pct": da.mean(1), "ref_mean_pct": ref.mean(1)})
    dfr["delta_pct"] = dfr.DA_mean_pct - dfr.ref_mean_pct
    dfr = dfr[sil.values].copy()
    ps = []
    for i in np.where(sil.values)[0]:
        try: ps.append(mannwhitneyu(da[i], ref[i], alternative="two-sided")[1])
        except ValueError: ps.append(1.0)
    dfr["mwu_p"] = ps
    dfr["mwu_FDR"] = multipletests(dfr.mwu_p, method="fdr_bh")[1]
    log(f"informative silencers: {len(dfr)}; DA-enriched (delta>0): {(dfr.delta_pct>0).sum()}")
    top = dfr.sort_values("delta_pct", ascending=False).head(TOPN).reset_index(drop=True)

    # 2. annotate each silencer to a target gene
    idx = m64.build_index(m64.load_genes())
    tgt = []
    for c in top.coord:
        chrom, s, e = c.split(":")[0], *map(int, c.split(":")[1].split("-"))
        mid = (s + e) // 2
        body = m64.body_overlap(idx, chrom, mid)
        pgn, pdist, _ = m64.nearest(idx, chrom, mid, pc_only=True)
        tgt.append(body if body else (pgn if abs(pdist) <= 25000 else pgn))
    top["target_gene"] = tgt
    top.to_csv(f"{OUT}/top25_silencers.csv", index=False)
    log("top silencers (coord / target / delta / FDR):\n" +
        top[["coord", "target_gene", "delta_pct", "mwu_FDR"]].to_string(index=False))

    # 3. pseudobulk target genes per atlas cluster
    genes = list(dict.fromkeys(list(top.target_gene) + MARKERS))
    exprs, metas = [], []
    for short, (pat, _) in m67.FILES.items():
        hit = glob.glob(f"{m67.T}/{pat}")[0]
        log(f"  pseudobulking {short}")
        ee, mm = m67.pseudobulk_file(hit, short, genes); exprs.append(ee); metas.append(mm)
    E = pd.concat(exprs).fillna(0.0); MM = pd.concat(metas)
    MM["label"] = [m67.label_cluster(MM.loc[k], E.loc[k]) for k in MM.index]
    keep = MM[MM.n_cells >= 50].index; E, MM = E.loc[keep], MM.loc[keep]

    order_lab = ["D1-MSN", "D2-MSN", "MSN-other/eccentric", "MSN", "IC-DA",
                 "Midbrain-DA", "Septal-DA", "MD-DA", "CGE", "MGE"]
    da_labels = {"D1-MSN", "D2-MSN", "MSN-other/eccentric", "MSN", "IC-DA",
                 "Midbrain-DA", "Septal-DA", "MD-DA", "SN-DA", "VTA/tegm-DA"}
    MM["ord"] = MM["label"].apply(lambda l: order_lab.index(l) if l in order_lab else 99)
    cl = MM.sort_values(["ord", "n_cells"], ascending=[True, False]).index.tolist()
    n_da = sum(MM.loc[c, "label"] in da_labels for c in cl)
    da_cl = [c for c in cl if MM.loc[c, "label"] in da_labels]
    ref_cl = [c for c in cl if MM.loc[c, "label"] not in da_labels]

    # 4. concordance: silencer -> repression -> DA RNA lower (log2FC < 0)
    rows = []
    for _, r in top.iterrows():
        g = r.target_gene
        if g not in E.columns:
            rows.append({**r, "RNA_DA": np.nan, "RNA_ref": np.nan, "RNA_log2FC_DA": np.nan,
                         "concordant_repression": np.nan}); continue
        da_e = E.loc[da_cl, g].mean(); ref_e = E.loc[ref_cl, g].mean()
        lfc = np.log2((da_e + 1e-3) / (ref_e + 1e-3))
        rows.append({"coord": r.coord, "target_gene": g, "ATAC_delta_pct": round(r.delta_pct, 2),
                     "mwu_FDR": round(r.mwu_FDR, 4), "RNA_DA": round(da_e, 3),
                     "RNA_ref": round(ref_e, 3), "RNA_log2FC_DA": round(lfc, 3),
                     "concordant_repression": bool(lfc < 0)})
    res = pd.DataFrame(rows)
    res.to_csv(f"{OUT}/top25_silencers.csv", index=False)
    nconc = int(res.concordant_repression.sum())
    log(f"\nCONCORDANCE (silencer->repression, RNA lower in DA): "
        f"{nconc}/{res.concordant_repression.notna().sum()} "
        f"({100*res.concordant_repression.mean():.0f}%); median RNA log2FC={res.RNA_log2FC_DA.median():+.2f}")

    # 5. heatmap
    grow = [g for g in top.target_gene if g in E.columns] + [g for g in MARKERS if g in E.columns]
    grow = list(dict.fromkeys(grow))
    n_sil = sum(g in set(top.target_gene) for g in grow) - sum(g in MARKERS and g in set(top.target_gene) for g in grow)
    Z = E[grow].T; Zz = Z.sub(Z.mean(1), 0).div(Z.std(1).replace(0, 1), 0)[cl]
    fig, ax = plt.subplots(figsize=(max(10, len(cl)*0.34), max(7, len(grow)*0.26+1.5)))
    im = ax.imshow(Zz.values, cmap="RdBu_r", aspect="auto", vmin=-2, vmax=2)
    ax.set_xticks(range(len(cl)))
    ax.set_xticklabels([f"{MM.loc[c,'label']} [{c}] n={int(MM.loc[c,'n_cells'])}" for c in cl],
                       rotation=90, fontsize=6)
    for k, c in enumerate(cl):
        ax.get_xticklabels()[k].set_color("#e6550d" if MM.loc[c, "label"] in da_labels else "#3182bd")
    ax.set_yticks(range(len(grow))); ax.set_yticklabels(grow, fontsize=7)
    ax.axvline(n_da-0.5, color="black", lw=2)
    ax.axhline(len(grow)-len(MARKERS)-0.5, color="k", lw=1.2)
    cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01); cb.set_label("z(expr) per gene", fontsize=8)
    ax.set_title("Top-25 DA-enriched SILENCERS — target-gene RNA across subtypes\n"
                 "silencer prediction = REPRESSION -> concordant if BLUE in DA clusters (orange labels) / RED in interneurons (blue)\n"
                 "genes ordered by silencer DA-specificity; last rows = identity markers",
                 fontsize=9.5)
    fig.tight_layout()
    fig.savefig(f"{OUT}/silencer_subtype_heatmap.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    log("wrote silencer_subtype_heatmap.png")


if __name__ == "__main__":
    main()
