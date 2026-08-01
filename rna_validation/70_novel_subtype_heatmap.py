#!/usr/bin/env python3
"""
Subtype-resolution enhancer<->RNA heatmap for NOVEL (ML-predicted) DA regulatory
elements — the novel-element analog of concordant_DAspecific_subtype_heatmap.

Novel regions are group-specific coordinates, so they cannot pass the cross-group
DA-specific prevalence gate. Here we instead rank NOVEL regulatory elements by their
accessibility in DA subtypes (DA presence, ~absent in the reference), annotate each to
a candidate target gene, and test whether those genes are DA-enriched in independent
snRNA-seq (Human Brain Cell Atlas) at subtype resolution.

Outputs -> DA_vs_nonDA/rna_validation/subtype/:
  novel_DA_targets.csv          top novel elements: coord, class, DA presence, target gene, RNA log2FC, concordant
  novel_DA_subtype_heatmap.png  target genes x DA/interneuron clusters
Env: snapatac2_env.
"""
import os, glob, importlib.util
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import da_config as C

D = "/path/to/data"
OUT = f"{D}/DA_vs_nonDA/rna_validation/subtype"
MARK = ["DRD1", "DRD2", "TH", "PPP1R1B", "GAD1", "SLC17A7"]
REG_CLASSES = {"enhancer", "silencer", "dual", "other_regulatory", "promoter"}
TOPN = 45
CLASS_COL = {"D1-MSN": "#e6550d", "D2-MSN": "#fd8d3c", "MSN-other/eccentric": "#fdae6b",
             "IC-DA": "#8c2d04", "Midbrain-DA": "#d94801", "CGE": "#3182bd", "MGE": "#6baed6"}


def load(path):
    s = importlib.util.spec_from_file_location("m", path)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


def log(m): print(m, flush=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    m62 = load(f"{D}/62_DA_vs_nonDA.py")
    m64 = load(f"{D}/64_annotate_DA_regions.py")
    m67 = load(f"{D}/67_concordant_subtype_heatmap.py")

    # 1. NOVEL regulatory elements straight from the DA-group masters (DA-exclusive by
    #    construction: novel coords are group-specific, so absent from the reference).
    rows = []
    for g in C.DA_GROUPS:
        f = f"{D}/Joint_Differential/regulatory/{g}_joint_master.csv.gz"
        d = pd.read_csv(f, usecols=lambda c: c in ("region_id", "basket", "region_class")
                        or c.startswith("Pct_"))
        pct = [c for c in d.columns if c.startswith("Pct_")]
        nv = d[d.basket == "novel"].copy()
        if not len(nv):
            continue
        nv["DA_max_pct"] = nv[pct].max(1)
        nv["region_class"] = nv["region_class"].str.replace("novel|", "", regex=False)
        nv["coord"] = nv["region_id"].str.split("|").str[-1]
        nv["group"] = g
        rows.append(nv[["coord", "region_class", "DA_max_pct", "group"]])
    allnv = pd.concat(rows)
    allnv = allnv.sort_values("DA_max_pct", ascending=False).drop_duplicates("coord")
    log(f"DA novel regulatory elements: {len(allnv)} "
        f"(classes: {allnv.region_class.value_counts().to_dict()}); DA_max top={allnv.DA_max_pct.max():.1f}%")
    top = allnv.head(TOPN).reset_index(drop=True)
    top["ref_mean_pct"] = 0.0   # DA-exclusive by construction

    # 2. annotate each novel element to a target gene (within-gene or nearest PC-TSS <50kb)
    idx = m64.build_index(m64.load_genes())
    genes, dists = [], []
    for c in top.coord:
        chrom, s, e = c.split(":")[0], *map(int, c.split(":")[1].split("-"))
        mid = (s + e) // 2
        body = m64.body_overlap(idx, chrom, mid)
        pgn, pd_, _ = m64.nearest(idx, chrom, mid, pc_only=True)
        genes.append(body if body else pgn); dists.append(0 if body else pd_)
    top["target_gene"] = genes; top["dist_to_TSS"] = dists

    # 3. pseudobulk target genes across atlas clusters
    panel = list(dict.fromkeys(list(top.target_gene) + MARK))
    exprs, metas = [], []
    for short, (pat, _) in m67.FILES.items():
        hit = glob.glob(f"{m67.T}/{pat}")[0]
        log(f"  pseudobulking {short}")
        ee, mm = m67.pseudobulk_file(hit, short, panel); exprs.append(ee); metas.append(mm)
    E = pd.concat(exprs).fillna(0.0); MM = pd.concat(metas)
    MM["label"] = [m67.label_cluster(MM.loc[k], E.loc[k]) for k in MM.index]
    keep = MM[MM.n_cells >= 50].index; E, MM = E.loc[keep], MM.loc[keep]

    order_lab = ["D1-MSN", "D2-MSN", "MSN-other/eccentric", "IC-DA", "Midbrain-DA", "CGE", "MGE"]
    da_labels = {"D1-MSN", "D2-MSN", "MSN-other/eccentric", "IC-DA", "Midbrain-DA", "SN-DA",
                 "Midbrain-DA", "Septal-DA", "MD-DA"}
    MM["ord"] = MM["label"].apply(lambda l: order_lab.index(l) if l in order_lab else 99)
    cl = MM.sort_values(["ord", "n_cells"], ascending=[True, False]).index.tolist()
    n_da = sum(MM.loc[c, "label"] in da_labels for c in cl)
    da_cl = [c for c in cl if MM.loc[c, "label"] in da_labels]
    ref_cl = [c for c in cl if MM.loc[c, "label"] not in da_labels]

    # 4. concordance: DA-active novel enhancer -> predict target higher in DA (log2FC>0)
    rec = []
    for _, r in top.iterrows():
        g = r.target_gene
        if g not in E.columns:
            continue
        da_e = E.loc[da_cl, g].mean(); ref_e = E.loc[ref_cl, g].mean()
        lfc = np.log2((da_e + 1e-3) / (ref_e + 1e-3))
        rec.append({"coord": r.coord, "region_class": r.region_class, "target_gene": g,
                    "dist_to_TSS": int(r.dist_to_TSS), "DA_max_pct": round(r.DA_max_pct, 1),
                    "ref_mean_pct": round(r.ref_mean_pct, 1), "RNA_DA": round(da_e, 3),
                    "RNA_ref": round(ref_e, 3), "RNA_log2FC_DA": round(lfc, 3),
                    "concordant": bool(lfc > 0)})
    res = pd.DataFrame(rec)
    res.to_csv(f"{OUT}/novel_DA_targets.csv", index=False)
    log(f"\nCONCORDANCE (novel DA element -> target higher in DA RNA): "
        f"{int(res.concordant.sum())}/{len(res)} ({100*res.concordant.mean():.0f}%); "
        f"median RNA log2FC = {res.RNA_log2FC_DA.median():+.2f}")

    E.join(MM[["label", "n_cells"]]).to_csv(f"{OUT}/novel_DA_cluster_expr.csv")

    # 5. heatmaps (full + concordant-only)
    def draw(gene_list, path, title):
        grow = [g for g in gene_list if g in E.columns] + [g for g in MARK if g in E.columns]
        grow = list(dict.fromkeys(grow)); n_mk = sum(g in MARK for g in grow)
        Z = E[grow].T; Zz = Z.sub(Z.mean(1), 0).div(Z.std(1).replace(0, 1), 0)[cl]
        fig, ax = plt.subplots(figsize=(max(11, len(cl) * 0.34), max(8, len(grow) * 0.26 + 2)))
        im = ax.imshow(Zz.values, cmap="RdBu_r", aspect="auto", vmin=-2, vmax=2)
        ax.set_xticks(range(len(cl)))
        ax.set_xticklabels([f"{MM.loc[c,'label']} [{c}] n={int(MM.loc[c,'n_cells'])}" for c in cl],
                           rotation=90, fontsize=6)
        for k, c in enumerate(cl):
            ax.get_xticklabels()[k].set_color("#a63603" if MM.loc[c, "label"] in da_labels else "#08519c")
        ax.set_yticks(range(len(grow))); ax.set_yticklabels(grow, fontsize=7)
        ax.axvline(n_da - 0.5, color="black", lw=2.5)
        ax.axhline(len(grow) - n_mk - 0.5, color="k", lw=1.2)
        cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01); cb.set_label("z(expr) per gene", fontsize=8)
        ax.set_title(title, fontsize=9.5)
        fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
        log(f"wrote {path} ({len(grow)} rows)")

    draw(list(res.target_gene),
         f"{OUT}/novel_DA_subtype_heatmap.png",
         "Target genes of top NOVEL (ML-predicted) DA regulatory elements — subtype-resolved RNA\n"
         "novel enhancer active in DA -> predicted target HIGHER in DA (red left / blue right = concordant)\n"
         "DA clusters = orange labels (left of black line) | interneuron ref = blue | last rows = markers")
    conc = res[res.concordant]
    draw(list(conc.target_gene),
         f"{OUT}/novel_DA_subtype_heatmap_concordant.png",
         f"CONCORDANT novel DA elements only (n={len(conc)}): target genes with RNA HIGHER in DA\n"
         "(subset of the top-45 novel DA elements; DA clusters orange, left of black line | markers last)")


if __name__ == "__main__":
    main()
