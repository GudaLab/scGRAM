#!/usr/bin/env python3
"""
FOXP2 analog of the novel-element RNA validation. Top NOVEL (ML-predicted) FOXP2
regulatory elements (from the FOXP2 group master; FOXP2-exclusive by construction)
-> candidate target genes -> subtype-resolved RNA.

FOXP2 has no dedicated atlas supercluster (FOXP2 marks striatal / eccentric MSNs),
so the RNA 'in-group' is defined transparently by the FOXP2 MARKER itself: atlas
clusters with above-median FOXP2 expression = FOXP2-high (cells of interest),
the rest = FOXP2-low (reference). Concordant = target gene higher in FOXP2-high.

Outputs -> DA_vs_nonDA/rna_validation/subtype/:
  novel_FOXP2_targets.csv, novel_FOXP2_cluster_expr.csv,
  novel_FOXP2_subtype_heatmap.png, novel_FOXP2_subtype_heatmap_concordant.png
Env: snapatac2_env.
"""
import os, glob, importlib.util
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = "/path/to/data"
OUT = f"{D}/DA_vs_nonDA/rna_validation/subtype"
GROUP = "FOXP2"
MARK = ["FOXP2", "DRD1", "DRD2", "PPP1R1B", "GAD1", "SLC17A7"]
REG_CLASSES = {"enhancer", "dual", "silencer", "other_regulatory", "promoter"}
TOPN = 45
POS_COL, NEG_COL = "#238b45", "#888888"


def load(path):
    s = importlib.util.spec_from_file_location("m", path)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


def log(m): print(m, flush=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    m64 = load(f"{D}/64_annotate_DA_regions.py")
    m67 = load(f"{D}/67_concordant_subtype_heatmap.py")

    # 1. top NOVEL FOXP2 regulatory elements straight from the FOXP2 master
    f = f"{D}/Joint_Differential/regulatory/{GROUP}_joint_master.csv.gz"
    d = pd.read_csv(f, usecols=lambda c: c in ("region_id", "basket", "region_class")
                    or c.startswith("Pct_"))
    pct = [c for c in d.columns if c.startswith("Pct_")]
    nv = d[d.basket == "novel"].copy()
    nv["region_class"] = nv["region_class"].str.replace("novel|", "", regex=False)
    nv = nv[nv.region_class.isin(REG_CLASSES)]
    nv["FOXP2_max_pct"] = nv[pct].max(1)
    nv["coord"] = nv["region_id"].str.split("|").str[-1]
    top = nv.sort_values("FOXP2_max_pct", ascending=False).drop_duplicates("coord").head(TOPN).reset_index(drop=True)
    log(f"FOXP2 novel regulatory elements: {len(nv)} (classes {nv.region_class.value_counts().to_dict()}); "
        f"top FOXP2_max={top.FOXP2_max_pct.max():.1f}%")

    # 2. annotate target genes
    idx = m64.build_index(m64.load_genes())
    genes, dists = [], []
    for c in top.coord:
        chrom, s, e = c.split(":")[0], *map(int, c.split(":")[1].split("-"))
        mid = (s + e) // 2
        body = m64.body_overlap(idx, chrom, mid)
        pgn, pd_, _ = m64.nearest(idx, chrom, mid, pc_only=True)
        genes.append(body if body else pgn); dists.append(0 if body else pd_)
    top["target_gene"] = genes; top["dist_to_TSS"] = dists

    # 3. pseudobulk target genes + FOXP2 marker across atlas clusters
    panel = list(dict.fromkeys(list(top.target_gene) + MARK))
    exprs, metas = [], []
    for short, (pat, _) in m67.FILES.items():
        hit = glob.glob(f"{m67.T}/{pat}")[0]
        log(f"  pseudobulking {short}")
        ee, mm = m67.pseudobulk_file(hit, short, panel); exprs.append(ee); metas.append(mm)
    E = pd.concat(exprs).fillna(0.0); MM = pd.concat(metas)
    MM["label"] = [m67.label_cluster(MM.loc[k], E.loc[k]) for k in MM.index]
    keep = MM[MM.n_cells >= 50].index; E, MM = E.loc[keep], MM.loc[keep]

    # 4. define FOXP2-high (in-group) vs FOXP2-low (reference) by the FOXP2 marker
    fx = E["FOXP2"] if "FOXP2" in E.columns else pd.Series(0.0, index=E.index)
    thr = fx.median()
    MM["foxp2"] = fx
    MM["ingroup"] = fx >= thr
    cl = MM.sort_values("foxp2", ascending=False).index.tolist()   # FOXP2-high -> low
    n_pos = int(MM["ingroup"].sum())
    pos = MM[MM.ingroup].index; neg = MM[~MM.ingroup].index
    log(f"FOXP2-high clusters: {n_pos}/{len(MM)} (FOXP2 median={thr:.2f}); "
        f"top labels: {MM.loc[cl[:6],'label'].tolist()}")

    # 5. concordance: FOXP2 novel element -> target higher in FOXP2-high clusters
    rec = []
    for _, r in top.iterrows():
        g = r.target_gene
        if g not in E.columns:
            continue
        pe, ne = E.loc[pos, g].mean(), E.loc[neg, g].mean()
        lfc = np.log2((pe + 1e-3) / (ne + 1e-3))
        rec.append({"coord": r.coord, "region_class": r.region_class, "target_gene": g,
                    "dist_to_TSS": int(r.dist_to_TSS), "FOXP2_max_pct": round(r.FOXP2_max_pct, 1),
                    "RNA_FOXP2hi": round(pe, 3), "RNA_FOXP2lo": round(ne, 3),
                    "RNA_log2FC": round(lfc, 3), "concordant": bool(lfc > 0)})
    res = pd.DataFrame(rec)
    res.to_csv(f"{OUT}/novel_FOXP2_targets.csv", index=False)
    E.join(MM[["label", "n_cells", "foxp2", "ingroup"]]).to_csv(f"{OUT}/novel_FOXP2_cluster_expr.csv")
    log(f"\nCONCORDANCE (FOXP2 novel element -> target higher in FOXP2-high RNA): "
        f"{int(res.concordant.sum())}/{len(res)} ({100*res.concordant.mean():.0f}%); "
        f"median log2FC = {res.RNA_log2FC.median():+.2f}")

    # 6. heatmaps (full + concordant)
    def draw(gene_list, path, title):
        grow = [g for g in gene_list if g in E.columns] + [g for g in MARK if g in E.columns]
        grow = list(dict.fromkeys(grow)); n_mk = sum(g in MARK for g in grow)
        Z = E[grow].T; Zz = Z.sub(Z.mean(1), 0).div(Z.std(1).replace(0, 1), 0)[cl]
        fig, ax = plt.subplots(figsize=(max(11, len(cl) * 0.34), max(8, len(grow) * 0.26 + 2)))
        im = ax.imshow(Zz.values, cmap="RdBu_r", aspect="auto", vmin=-2, vmax=2)
        ax.set_xticks(range(len(cl)))
        ax.set_xticklabels([f"{MM.loc[c,'label']} [{c}] FOXP2={MM.loc[c,'foxp2']:.1f} n={int(MM.loc[c,'n_cells'])}"
                            for c in cl], rotation=90, fontsize=6)
        for k, c in enumerate(cl):
            ax.get_xticklabels()[k].set_color(POS_COL if MM.loc[c, "ingroup"] else NEG_COL)
        ax.set_yticks(range(len(grow))); ax.set_yticklabels(grow, fontsize=7)
        ax.axvline(n_pos - 0.5, color="black", lw=2.5)
        ax.axhline(len(grow) - n_mk - 0.5, color="k", lw=1.2)
        cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01); cb.set_label("z(expr) per gene", fontsize=8)
        ax.set_title(title, fontsize=9.5)
        fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
        log(f"wrote {path} ({len(grow)} rows)")

    draw(list(res.target_gene),
         f"{OUT}/novel_FOXP2_subtype_heatmap.png",
         "Target genes of top NOVEL FOXP2 regulatory elements — RNA by FOXP2-high vs FOXP2-low clusters\n"
         "novel enhancer active in FOXP2 -> predicted target HIGHER in FOXP2-high (red left / blue right = concordant)\n"
         "FOXP2-high clusters = green labels (left of black line) | FOXP2-low = grey | last rows = markers")
    conc = res[res.concordant]
    draw(list(conc.target_gene),
         f"{OUT}/novel_FOXP2_subtype_heatmap_concordant.png",
         f"CONCORDANT novel FOXP2 elements only (n={len(conc)}): target genes higher in FOXP2-high clusters\n"
         "(FOXP2-high = green labels, left of black line | markers last)")


if __name__ == "__main__":
    main()
