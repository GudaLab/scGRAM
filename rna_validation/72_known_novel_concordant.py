#!/usr/bin/env python3
"""
Stitched KNOWN + NOVEL concordant enhancer<->RNA heatmap, per cell type (DA, FOXP2).

For a cell type C, both baskets are selected the same way and validated the same way:
  KNOWN : top-N known regulatory regions by C-specificity (C presence - reference presence),
          from the informative region x subtype matrix (shared coordinate catalog).
  NOVEL : top-N novel regulatory elements by C presence, straight from C's group masters
          (novel coords are group-specific -> C-exclusive by construction).
Each element -> candidate target gene (gene.gtf). Target-gene RNA pseudobulked at atlas
cluster resolution; concordant = target higher in the C in-group than the reference
(log2FC>0). The figure stacks the concordant KNOWN block over the concordant NOVEL block
(+ identity markers), columns = clusters split into C in-group vs reference.

Outputs -> DA_vs_nonDA/rna_validation/subtype/known_novel_concordant_<C>.png (+ _table.csv)
Env: snapatac2_env.
"""
import os, glob, importlib.util
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import da_config as C

D = "/path/to/data"
OUT = f"{D}/DA_vs_nonDA/rna_validation/subtype"
REG = {"enhancer", "dual", "silencer", "other_regulatory", "promoter"}
N = 40
ALL_SUB = C.DA_SUBTYPES + C.ALL_NONDA_SUBTYPES
DA_LABELS = {"D1-MSN", "D2-MSN", "MSN-other/eccentric", "IC-DA", "Midbrain-DA",
             "SN-DA", "Septal-DA", "MD-DA"}

CFG = {
    "DA": dict(subtypes=C.DA_SUBTYPES, novel_groups=C.DA_GROUPS, mode="DA",
               markers=["DRD1", "DRD2", "TH", "PPP1R1B", "GAD1", "SLC17A7"]),
    "FOXP2": dict(subtypes=["FOXP2_1", "FOXP2_2", "FOXP2_3", "FOXP2_4"], novel_groups=["FOXP2"],
                  mode="FOXP2", markers=["FOXP2", "DRD1", "DRD2", "PPP1R1B", "GAD1", "SLC17A7"]),
}


def load(p):
    s = importlib.util.spec_from_file_location("m", p); m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m); return m


def log(m): print(m, flush=True)


m62 = load(f"{D}/62_DA_vs_nonDA.py")
m64 = load(f"{D}/64_annotate_DA_regions.py")
m67 = load(f"{D}/67_concordant_subtype_heatmap.py")
IDX = None


def target_gene(coord):
    chrom, s, e = coord.split(":")[0], *map(int, coord.split(":")[1].split("-"))
    mid = (s + e) // 2
    body = m64.body_overlap(IDX, chrom, mid)
    if body:
        return body
    pgn, pd_, _ = m64.nearest(IDX, chrom, mid, pc_only=True)
    return pgn


def select_known(M, meta, subs):
    csub = [s for s in subs if s in M.columns]
    rsub = [s for s in M.columns if s not in csub]
    cm = M[csub].values.mean(1); rm = M[rsub].values.mean(1)
    t = pd.DataFrame({"coord": meta["coord"].values, "region_class": meta["region_class"].values,
                      "basket": meta["basket"].values, "C_mean": cm, "ref_mean": rm})
    t["delta"] = t.C_mean - t.ref_mean
    t = t[(t.basket == "known") & (t.region_class.isin(REG)) & (t.C_mean >= 10)]
    return t.sort_values("delta", ascending=False).head(N)[["coord", "region_class"]].assign(source="known")


def select_novel(groups):
    rows = []
    for g in groups:
        f = f"{D}/Joint_Differential/regulatory/{g}_joint_master.csv.gz"
        d = pd.read_csv(f, usecols=lambda c: c in ("region_id", "basket", "region_class") or c.startswith("Pct_"))
        pct = [c for c in d.columns if c.startswith("Pct_")]
        nv = d[d.basket == "novel"].copy()
        nv["region_class"] = nv["region_class"].str.replace("novel|", "", regex=False)
        nv = nv[nv.region_class.isin(REG)]
        nv["pres"] = nv[pct].max(1); nv["coord"] = nv["region_id"].str.split("|").str[-1]
        rows.append(nv[["coord", "region_class", "pres"]])
    a = pd.concat(rows).sort_values("pres", ascending=False).drop_duplicates("coord").head(N)
    return a[["coord", "region_class"]].assign(source="novel")


def analyze(name):
    cfg = CFG[name]
    M, meta = m62.build_matrix()
    elems = pd.concat([select_known(M, meta, cfg["subtypes"]), select_novel(cfg["novel_groups"])],
                      ignore_index=True)
    elems["target_gene"] = elems.coord.map(target_gene)
    log(f"[{name}] elements: {len(elems)} (known {sum(elems.source=='known')}, novel {sum(elems.source=='novel')})")

    panel = list(dict.fromkeys(list(elems.target_gene) + cfg["markers"]))
    exprs, metas = [], []
    for short, (pat, _) in m67.FILES.items():
        hit = glob.glob(f"{m67.T}/{pat}")[0]; log(f"  pseudobulk {short}")
        ee, mm = m67.pseudobulk_file(hit, short, panel); exprs.append(ee); metas.append(mm)
    E = pd.concat(exprs).fillna(0.0); MM = pd.concat(metas)
    MM["label"] = [m67.label_cluster(MM.loc[k], E.loc[k]) for k in MM.index]
    MM = MM[MM.n_cells >= 50]; E = E.loc[MM.index]

    if cfg["mode"] == "DA":
        MM["ingroup"] = MM.label.isin(DA_LABELS)
        order_lab = ["D1-MSN", "D2-MSN", "MSN-other/eccentric", "IC-DA", "Midbrain-DA", "CGE", "MGE"]
        MM["k"] = MM.label.map(lambda l: order_lab.index(l) if l in order_lab else 99)
        cl = MM.sort_values(["k", "n_cells"], ascending=[True, False]).index.tolist()
        xcolor = lambda c: "#a63603" if MM.loc[c, "ingroup"] else "#08519c"
    else:
        fx = E["FOXP2"] if "FOXP2" in E.columns else pd.Series(0.0, index=E.index)
        MM["ingroup"] = fx >= fx.median(); MM["foxp2"] = fx
        cl = MM.sort_values("foxp2", ascending=False).index.tolist()
        xcolor = lambda c: "#238b45" if MM.loc[c, "ingroup"] else "#888888"
    ing = MM[MM.ingroup].index; ref = MM[~MM.ingroup].index; n_in = len(ing)

    # concordance per element
    def lfc(g):
        if g not in E.columns:
            return np.nan
        return np.log2((E.loc[ing, g].mean() + 1e-3) / (E.loc[ref, g].mean() + 1e-3))
    elems["RNA_log2FC"] = elems.target_gene.map(lfc)
    elems["concordant"] = elems.RNA_log2FC > 0
    elems.to_csv(f"{OUT}/known_novel_concordant_{name}_table.csv", index=False)

    kg = list(dict.fromkeys(elems[(elems.source == "known") & elems.concordant & elems.target_gene.isin(E.columns)].target_gene))
    ng = list(dict.fromkeys(elems[(elems.source == "novel") & elems.concordant & elems.target_gene.isin(E.columns)].target_gene))
    mk = [g for g in cfg["markers"] if g in E.columns]
    log(f"[{name}] concordant: known {len(kg)}, novel {len(ng)} | "
        f"known rate {elems[elems.source=='known'].concordant.mean():.0%}, "
        f"novel rate {elems[elems.source=='novel'].concordant.mean():.0%}")

    grow = kg + ng + mk
    Z = E[grow].T; Zz = Z.sub(Z.mean(1), 0).div(Z.std(1).replace(0, 1), 0)[cl]
    fig, ax = plt.subplots(figsize=(max(12, len(cl) * 0.34), max(9, len(grow) * 0.24 + 2)))
    im = ax.imshow(Zz.values, cmap="RdBu_r", aspect="auto", vmin=-2, vmax=2)
    ax.set_xticks(range(len(cl)))
    ax.set_xticklabels([f"{MM.loc[c,'label']} [{c}] n={int(MM.loc[c,'n_cells'])}" for c in cl],
                       rotation=90, fontsize=5.5)
    for k, c in enumerate(cl):
        ax.get_xticklabels()[k].set_color(xcolor(c))
    ax.set_yticks(range(len(grow))); ax.set_yticklabels(grow, fontsize=7)
    ax.axvline(n_in - 0.5, color="black", lw=2.5)
    b1, b2 = len(kg), len(kg) + len(ng)
    for y in (b1, b2):
        ax.axhline(y - 0.5, color="k", lw=1.4)
    inlab = "DA neurons" if cfg["mode"] == "DA" else "FOXP2-high"
    reflab = "interneuron reference" if cfg["mode"] == "DA" else "FOXP2-low"
    for lab, y0, y1, col in [("KNOWN\nconcordant", 0, b1, "#6a51a3"),
                             ("NOVEL\nconcordant", b1, b2, "#08519c"),
                             ("markers", b2, len(grow), "#333")]:
        ax.text(1.008, 1 - (y0 + y1) / 2 / len(grow), lab, transform=ax.transAxes, rotation=90,
                ha="left", va="center", fontsize=10, fontweight="bold", color=col)
    cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.035); cb.set_label("z(expr) per gene", fontsize=8)
    ax.set_title(f"{name}: KNOWN + NOVEL concordant regulatory elements — subtype-resolved RNA\n"
                 f"left of black line = {inlab} | right = {reflab} | "
                 f"rows: concordant KNOWN targets / concordant NOVEL targets / markers",
                 fontsize=10)
    fig.tight_layout(); fig.savefig(f"{OUT}/known_novel_concordant_{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"wrote known_novel_concordant_{name}.png ({len(grow)} rows x {len(cl)} clusters)")


def main():
    global IDX
    IDX = m64.build_index(m64.load_genes())
    for name in ("DA", "FOXP2"):
        analyze(name)
    log("DONE")


if __name__ == "__main__":
    main()
