#!/usr/bin/env python3
"""
24_joint_crossgroup.py
======================
JOINT cross-group analysis on the known+novel joint master tables produced by
23_joint_differential.py, in two versions (genebody / regulatory).

Mirrors the cross_group_brain panels on the combined set, adding the known/novel
dimension:
  C1  1 - Spearman divergence clustermap (subtype x top-variable-region matrix)
  C2  PCA of regulatory fingerprints (+ scree)
  C3  t-SNE
  C4  region-CLASS composition per subtype (known|* vs novel|* stacked)  <- known-vs-novel view
  C6  concordance of significant regions across groups (joint pairwise)
  C7  Shannon diversity of region-class composition
  C8  conserved vs divergent regions
  C9  intra- vs inter-group divergence
  KN  known-vs-novel contribution to each subtype's top-variable fingerprint  <- NEW

Input  : Joint_Differential/<version>/<group>_joint_master.csv.gz
         Joint_Differential/<version>/<group>_pairwise/<A>_vs_<B>.csv.gz
Output : Joint_CrossGroup/<version>/  (panels + figure_tables/)
Env    : snapatac2_env (sklearn).  TOP_PER_GROUP = 500.
"""
from __future__ import annotations
import os, sys, glob, argparse, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

BASE = "/path/to/data"
DIFF = f"{BASE}/Joint_Differential"
OUT = f"{BASE}/Joint_CrossGroup"
TOP_PER_GROUP = 500
plt.rcParams.update({"font.size": 9, "savefig.dpi": 300, "savefig.bbox": "tight",
                     "pdf.fonttype": 42})


def log(m):
    print(f"[cross] {m}", flush=True)


def save(fig, stem):
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def basket_of(region_class):
    return "known" if str(region_class).startswith("known") else "novel"


def run_version(version):
    vdir = f"{DIFF}/{version}"
    outdir = f"{OUT}/{version}"
    tdir = f"{outdir}/figure_tables"
    os.makedirs(tdir, exist_ok=True)
    masters = sorted(glob.glob(f"{vdir}/*_joint_master.csv.gz"))
    if not masters:
        log(f"no joint masters for {version}; run 23 first"); return
    log(f"=== version={version}: {len(masters)} groups ===")

    # ---- build subtype x region presence matrix from top-variable regions ----
    region_union = {}          # region_id -> (chrom,start,end,region_class)
    subtype_pct = {}           # subtype -> {region_id: pct}
    subtype_group = {}         # subtype -> group
    class_by_subtype = {}      # subtype -> Counter of region_class (full table)
    sig_regions = {}           # group -> set(region_id) significant in any pair
    novel_frac = {}            # subtype -> frac of top-variable that is novel

    for mf in masters:
        grp = os.path.basename(mf).replace("_joint_master.csv.gz", "")
        df = pd.read_csv(mf, low_memory=False, dtype={"Chromosome": str})
        pcts = [c for c in df.columns if c.startswith("Pct_")]
        subs = [c[4:] for c in pcts]
        # composition over FULL joint table
        for s in subs:
            class_by_subtype.setdefault(s, {})
            subtype_group[s] = grp
        for cls, n in df["region_class"].value_counts().items():
            for s in subs:
                class_by_subtype[s][cls] = class_by_subtype[s].get(cls, 0) + int(n)
        # top variable regions for the fingerprint
        top = df.nlargest(TOP_PER_GROUP, "Diff_pct")
        nnovel = 0
        for _, r in top.iterrows():
            rid = r["region_id"]
            region_union[rid] = (r["Chromosome"], r["Start"], r["End"], r["region_class"])
            if basket_of(r["region_class"]) == "novel":
                nnovel += 1
            for s in subs:
                subtype_pct.setdefault(s, {})[rid] = r[f"Pct_{s}"]
        for s in subs:
            novel_frac[s] = nnovel / max(1, len(top))
        # significant regions (union across this group's joint pairwise)
        sig = set()
        for pf in glob.glob(f"{vdir}/{grp}_pairwise/*_vs_*.csv.gz"):
            pw = pd.read_csv(pf, usecols=lambda c: c in ("Chromosome", "Start", "End", "sig_FDR005", "region_class", "basket"),
                             low_memory=False, dtype={"Chromosome": str})
            if "sig_FDR005" in pw:
                pw = pw[pw["sig_FDR005"].fillna(False).astype(bool)]
                if len(pw):
                    sig.update((pw["Chromosome"].astype(str) + ":" +
                                pw["Start"].astype(str) + "-" + pw["End"].astype(str)).tolist())
        sig_regions[grp] = sig

    subtypes = sorted(subtype_pct.keys())
    regions = sorted(region_union.keys())
    ridx = {r: i for i, r in enumerate(regions)}
    M = np.zeros((len(subtypes), len(regions)))
    for i, s in enumerate(subtypes):
        for rid, p in subtype_pct[s].items():
            M[i, ridx[rid]] = p
    log(f"matrix {M.shape} ({len(subtypes)} subtypes x {len(regions)} top-variable regions)")

    # ---- C9 / C1: divergence matrix (1 - Spearman) ----
    rho, _ = spearmanr(M.T)
    if np.isscalar(rho):
        rho = np.array([[1.0, rho], [rho, 1.0]])
    div = 1.0 - rho
    np.fill_diagonal(div, 0.0)
    divdf = pd.DataFrame(div, index=subtypes, columns=subtypes)
    divdf.to_csv(f"{tdir}/C1_divergence_matrix.csv")

    groups_of = [subtype_group[s] for s in subtypes]
    intra, inter = [], []
    for i in range(len(subtypes)):
        for j in range(i + 1, len(subtypes)):
            (intra if groups_of[i] == groups_of[j] else inter).append(div[i, j])
    pd.DataFrame({"pair_type": ["intra"] * len(intra) + ["inter"] * len(inter),
                 "divergence": intra + inter}).to_csv(f"{tdir}/C9_intra_inter.csv", index=False)

    # C1 clustermap
    try:
        cond = squareform(div, checks=False)
        Z = linkage(cond, method="average")
        order = dendrogram(Z, no_plot=True)["leaves"]
        fig, ax = plt.subplots(figsize=(min(22, 0.28 * len(subtypes) + 4),
                                        min(20, 0.28 * len(subtypes) + 4)), constrained_layout=True)
        im = ax.imshow(div[np.ix_(order, order)], cmap="magma_r", aspect="auto")
        ax.set_xticks(range(len(subtypes))); ax.set_xticklabels([subtypes[o] for o in order], rotation=90, fontsize=6)
        ax.set_yticks(range(len(subtypes))); ax.set_yticklabels([subtypes[o] for o in order], fontsize=6)
        fig.colorbar(im, ax=ax, label="1 - Spearman", shrink=0.6)
        ax.set_title(f"C1 joint divergence clustermap ({version}) — top {TOP_PER_GROUP} variable/group")
        save(fig, f"{outdir}/C1_divergence_clustermap")
    except Exception as e:
        log(f"C1 skipped: {e}")

    # C2 PCA
    try:
        pca = PCA(n_components=min(10, M.shape[0] - 1))
        co = pca.fit_transform(M)
        fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
        gcolors = {g: c for g, c in zip(sorted(set(groups_of)),
                   plt.cm.tab20(np.linspace(0, 1, len(set(groups_of)))))}
        for g in sorted(set(groups_of)):
            idx = [i for i, gg in enumerate(groups_of) if gg == g]
            ax.scatter(co[idx, 0], co[idx, 1], s=45, color=gcolors[g], label=g, edgecolor="k", linewidth=0.3)
        cen = {}
        for g in set(groups_of):
            idx = [i for i, gg in enumerate(groups_of) if gg == g]
            cen[g] = (co[idx, 0].mean(), co[idx, 1].mean())
            ax.annotate(g, cen[g], fontsize=7, weight="bold")
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
        ax.set_title(f"C2 PCA joint fingerprints ({version})")
        save(fig, f"{outdir}/C2_PCA")
        pd.DataFrame({"subtype": subtypes, "group": groups_of, "PC1": co[:, 0], "PC2": co[:, 1],
                      "novel_frac_of_top": [novel_frac.get(s, np.nan) for s in subtypes]}).to_csv(
            f"{tdir}/C2_PCA_coords.csv", index=False)
    except Exception as e:
        log(f"C2 skipped: {e}")

    # C3 t-SNE
    try:
        if len(subtypes) > 5:
            perp = min(30, max(2, len(subtypes) // 3))
            ts = TSNE(n_components=2, metric="precomputed", init="random",
                      perplexity=perp, random_state=42).fit_transform(div)
            fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
            for g in sorted(set(groups_of)):
                idx = [i for i, gg in enumerate(groups_of) if gg == g]
                ax.scatter(ts[idx, 0], ts[idx, 1], s=45, label=g, edgecolor="k", linewidth=0.3)
                ax.annotate(g, (ts[idx, 0].mean(), ts[idx, 1].mean()), fontsize=7, weight="bold")
            ax.set_title(f"C3 t-SNE joint divergence ({version}, perplexity={perp})")
            ax.set_xticks([]); ax.set_yticks([])
            save(fig, f"{outdir}/C3_tSNE")
    except Exception as e:
        log(f"C3 skipped: {e}")

    # C4 region-class composition (known|* vs novel|*)
    try:
        classes = sorted({c for d in class_by_subtype.values() for c in d})
        comp = pd.DataFrame(0, index=subtypes, columns=classes, dtype=float)
        for s in subtypes:
            tot = sum(class_by_subtype[s].values()) or 1
            for c, n in class_by_subtype[s].items():
                comp.loc[s, c] = n / tot * 100
        comp = comp.loc[[subtypes[o] for o in order]] if 'order' in dir() else comp
        comp.to_csv(f"{tdir}/C4_region_class_composition.csv")
        palette = {c: ("#%02x%02x%02x" % tuple(int(x*255) for x in plt.cm.tab20(i % 20)[:3]))
                   for i, c in enumerate(classes)}
        fig, ax = plt.subplots(figsize=(max(10, 0.3 * len(subtypes) + 3), 6.5), constrained_layout=True)
        bottom = np.zeros(len(comp))
        for c in classes:
            ax.bar(comp.index, comp[c], bottom=bottom, label=c, color=palette[c])
            bottom += comp[c].values
        ax.set_xticklabels(comp.index, rotation=90, fontsize=6)
        ax.set_ylabel("% of subtype's joint regions")
        ax.set_title(f"C4 region-class composition — known vs novel ({version})")
        ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=6, frameon=False)
        save(fig, f"{outdir}/C4_region_class_composition")
    except Exception as e:
        log(f"C4 skipped: {e}")

    # C6 concordance (shared significant regions across groups)
    try:
        gs = sorted(sig_regions.keys())
        C = pd.DataFrame(0, index=gs, columns=gs, dtype=int)
        for a in gs:
            for b in gs:
                C.loc[a, b] = len(sig_regions[a] & sig_regions[b])
        C.to_csv(f"{tdir}/C6_concordance_matrix.csv")
        fig, ax = plt.subplots(figsize=(11, 9), constrained_layout=True)
        im = ax.imshow(np.log1p(C.values), cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(gs))); ax.set_xticklabels(gs, rotation=90, fontsize=7)
        ax.set_yticks(range(len(gs))); ax.set_yticklabels(gs, fontsize=7)
        fig.colorbar(im, ax=ax, label="log(1+shared sig regions)", shrink=0.6)
        ax.set_title(f"C6 concordance of significant regions ({version})")
        save(fig, f"{outdir}/C6_concordance_heatmap")
    except Exception as e:
        log(f"C6 skipped: {e}")

    # C7 Shannon diversity of region-class composition
    try:
        rows = []
        for s in subtypes:
            tot = sum(class_by_subtype[s].values()) or 1
            p = np.array([n / tot for n in class_by_subtype[s].values() if n > 0])
            H = -(p * np.log(p)).sum()
            rows.append({"subtype": s, "group": subtype_group[s], "shannon": H,
                         "n_regions": tot, "novel_frac_top": novel_frac.get(s, np.nan)})
        sh = pd.DataFrame(rows)
        sh.to_csv(f"{tdir}/C7_shannon.csv", index=False)
        fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
        sh2 = sh.sort_values("shannon")
        ax.scatter(sh2["n_regions"], sh2["shannon"], s=30, c=sh2["novel_frac_top"], cmap="coolwarm")
        ax.set_xscale("log"); ax.set_xlabel("total joint regions (log)"); ax.set_ylabel("Shannon (region-class)")
        sm = ax.collections[0]; fig.colorbar(sm, ax=ax, label="novel fraction of top-variable")
        ax.set_title(f"C7 regulatory complexity ({version})")
        save(fig, f"{outdir}/C7_complexity")
    except Exception as e:
        log(f"C7 skipped: {e}")

    # C8 conserved vs divergent (presence across subtypes at >5%)
    try:
        present = (M > 5).sum(axis=0)
        nsub = len(subtypes)
        universal = int((present > 0.8 * nsub).sum())
        shared = int(((present >= 0.2 * nsub) & (present <= 0.8 * nsub)).sum())
        specific = int((present < 0.2 * nsub).sum())
        cd = pd.DataFrame({"category": ["universal(>80%)", "shared(20-80%)", "specific(<20%)"],
                           "n_regions": [universal, shared, specific]})
        cd.to_csv(f"{tdir}/C8_conserved_divergent.csv", index=False)
        fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
        ax.bar(cd["category"], cd["n_regions"], color=["#4C72B0", "#DD8452", "#C44E52"])
        ax.set_ylabel("regions"); ax.set_title(f"C8 conserved vs divergent ({version})")
        for i, v in enumerate(cd["n_regions"]):
            ax.text(i, v, str(v), ha="center", va="bottom", fontsize=9)
        save(fig, f"{outdir}/C8_conserved_vs_divergent")
    except Exception as e:
        log(f"C8 skipped: {e}")

    # C9 intra vs inter figure
    try:
        fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
        ax.boxplot([intra, inter], tick_labels=[f"intra (n={len(intra)})", f"inter (n={len(inter)})"], showfliers=False)
        ax.set_ylabel("1 - Spearman divergence")
        ax.set_title(f"C9 intra vs inter-group divergence ({version})")
        save(fig, f"{outdir}/C9_intra_vs_inter")
    except Exception as e:
        log(f"C9 skipped: {e}")

    # KN known-vs-novel contribution to fingerprint
    try:
        kn = pd.DataFrame({"subtype": subtypes, "group": groups_of,
                           "novel_frac_of_top": [novel_frac.get(s, np.nan) for s in subtypes]})
        kn = kn.sort_values("novel_frac_of_top", ascending=False)
        kn.to_csv(f"{tdir}/KN_novel_fraction_of_fingerprint.csv", index=False)
        fig, ax = plt.subplots(figsize=(max(10, 0.28 * len(subtypes) + 3), 5.5), constrained_layout=True)
        ax.bar(kn["subtype"], kn["novel_frac_of_top"] * 100, color="#C44E52")
        ax.set_xticklabels(kn["subtype"], rotation=90, fontsize=6)
        ax.set_ylabel("% of top-variable fingerprint that is NOVEL")
        ax.set_title(f"KN novel contribution to cross-group fingerprint ({version})")
        save(fig, f"{outdir}/KN_novel_fraction_fingerprint")
    except Exception as e:
        log(f"KN skipped: {e}")

    # index
    figs = sorted(glob.glob(f"{outdir}/*.png"))
    pd.DataFrame({"figure": [os.path.basename(f) for f in figs]}).to_csv(
        f"{outdir}/FIGURES_INDEX.csv", index=False)
    log(f"version {version} done: {len(figs)} panels -> {outdir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="both", choices=["genebody", "regulatory", "both"])
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    for v in (["genebody", "regulatory"] if args.version == "both" else [args.version]):
        run_version(v)
    log("JOINT CROSSGROUP COMPLETE")


if __name__ == "__main__":
    main()
