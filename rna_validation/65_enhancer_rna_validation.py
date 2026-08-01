#!/usr/bin/env python3
"""
Enhancer<->RNA validation of DA-specific / DA-absent regulatory regions.
Independent test: do the candidate TARGET GENES of DA-specific enhancers show
DA-enriched RNA (and DA-absent enhancers' genes show DA-depleted RNA) in the
Human Brain Cell Atlas snRNA-seq (Siletti et al.)?

RNA: All_neurons.h5ad (2.48M nuclei x 59,236 genes, raw counts, ENSG var),
grouped by supercluster_term. DA superclusters = Medium spiny neuron,
Eccentric medium spiny neuron, Midbrain-derived inhibitory (== our DA set).
Expression = mean over cells of log1p(CP10K).

Candidate target genes: from DA_vs_nonDA/*/DA_(specific|absent)_regions_annotated.csv,
using within_gene (region inside gene) or nearest protein-coding TSS < 25 kb.

Outputs -> DA_vs_nonDA/rna_validation/:
  candidate_gene_rna.csv       per gene: ATAC delta, DA vs nonDA RNA, log2FC, concordant
  validation_summary.txt
  scatter_atac_vs_rna.png      ATAC enhancer DA-specificity vs RNA DA log2FC
  supercluster_expr_heatmap.png
Env: snapatac2_env (anndata/scanpy).
"""
import os, re, glob
import numpy as np, pandas as pd
import anndata as ad
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = "/path/to/data"
H5 = f"{D}/unbound_characetrize/transcriptome/All_neurons.h5ad"
GTF = f"{D}/gene.gtf"
OUT = f"{D}/DA_vs_nonDA/rna_validation"
DA_SC = {"Medium spiny neuron", "Eccentric medium spiny neuron", "Midbrain-derived inhibitory"}
MAX_TSS = 25000
CHUNK = 200_000


def log(m): print(m, flush=True)


def sym2ensg():
    m = {}
    pn = re.compile(r'gene_name "([^"]+)"'); pi = re.compile(r'gene_id "([^"]+)"')
    for ln in open(GTF):
        if ln.startswith("#") or "\tgene\t" not in ln:
            continue
        a = ln.split("\t")[8]
        nm = pn.search(a); gi = pi.search(a)
        if nm and gi:
            m[nm.group(1)] = gi.group(1).split(".")[0]
    return m


def candidates():
    """Return dict gene -> (direction, best_atac_delta, example_coord)."""
    rows = []
    for path in glob.glob(f"{D}/DA_vs_nonDA/*/DA_specific_regions_annotated.csv") + \
                glob.glob(f"{D}/DA_vs_nonDA/*/DA_absent_regions_annotated.csv"):
        direction = "specific" if "specific" in path else "absent"
        df = pd.read_csv(path)
        for _, r in df.iterrows():
            g = r.get("within_gene")
            if not isinstance(g, str) or g == "" or g.startswith("ENSG"):
                if abs(r.get("pc_dist_to_TSS", 1e9)) <= MAX_TSS:
                    g = r["nearest_pc_gene"]
                else:
                    continue
            rows.append((g, direction, float(r["delta_pct"]), r["coord"]))
    best = {}
    for g, d, delta, coord in rows:
        key = (g, d)
        if key not in best or abs(delta) > abs(best[key][1]):
            best[key] = (d, delta, coord)
    # if a gene appears in both directions, keep the stronger |delta|
    out = {}
    for (g, d), (d2, delta, coord) in best.items():
        if g not in out or abs(delta) > abs(out[g][1]):
            out[g] = (d, delta, coord)
    return out


def pseudobulk(genes_ensg):
    a = ad.read_h5ad(H5, backed="r")
    var_ens = pd.Index([v.split(".")[0] for v in a.var_names])
    col = {e: i for i, e in enumerate(var_ens)}
    cols = [col[e] for e in genes_ensg if e in col]
    found = [e for e in genes_ensg if e in col]
    log(f"  {len(found)}/{len(genes_ensg)} candidate genes found in RNA var")
    sc = a.obs["supercluster_term"].astype(str).values
    umis = a.obs["total_UMIs"].values.astype(np.float64)
    scs = sorted(set(sc))
    sums = {s: np.zeros(len(cols)) for s in scs}
    cnts = {s: 0 for s in scs}
    n = a.n_obs
    for start in range(0, n, CHUNK):
        end = min(start + CHUNK, n)
        X = a[start:end].X[:, cols]
        X = np.asarray(X.todense(), dtype=np.float64)
        lib = umis[start:end][:, None]; lib[lib == 0] = 1
        X = np.log1p(X / lib * 1e4)
        chsc = sc[start:end]
        for s in np.unique(chsc):
            mask = chsc == s
            sums[s] += X[mask].sum(0); cnts[s] += int(mask.sum())
        log(f"    rows {end}/{n}")
    expr = pd.DataFrame({s: sums[s] / max(cnts[s], 1) for s in scs}, index=found)
    ncells = pd.Series(cnts)
    return expr, ncells


def main():
    os.makedirs(OUT, exist_ok=True)
    cand = candidates()
    s2e = sym2ensg()
    gene_ensg = {g: s2e[g] for g in cand if g in s2e}
    log(f"candidates: {len(cand)} genes, {len(gene_ensg)} mapped to ENSG")
    expr, ncells = pseudobulk(list(gene_ensg.values()))
    ens2sym = {v: k for k, v in gene_ensg.items()}
    expr.index = [ens2sym.get(e, e) for e in expr.index]
    expr.to_csv(f"{OUT}/supercluster_expr_matrix.csv")

    da_cols = [c for c in expr.columns if c in DA_SC]
    ref_cols = [c for c in expr.columns if c not in DA_SC]
    da_e = expr[da_cols].mean(1); ref_e = expr[ref_cols].mean(1)
    log2fc = np.log2((da_e + 1e-3) / (ref_e + 1e-3))
    rec = []
    for g in expr.index:
        d, delta, coord = cand[g]
        pred_up = d == "specific"
        obs_up = log2fc[g] > 0
        rec.append({"gene": g, "direction": d, "atac_delta_pct": round(delta, 2),
                    "enhancer_coord": coord, "RNA_DA_expr": round(da_e[g], 3),
                    "RNA_nonDA_expr": round(ref_e[g], 3), "RNA_log2FC_DA": round(log2fc[g], 3),
                    "concordant": bool(pred_up == obs_up)})
    res = pd.DataFrame(rec).sort_values(["direction", "atac_delta_pct"],
                                        ascending=[True, False])
    res.to_csv(f"{OUT}/candidate_gene_rna.csv", index=False)

    spec = res[res.direction == "specific"]; absent = res[res.direction == "absent"]
    with open(f"{OUT}/validation_summary.txt", "w") as fh:
        fh.write("Enhancer<->RNA validation (Human Brain Cell Atlas, All_neurons)\n")
        fh.write(f"DA superclusters: {sorted(DA_SC & set(expr.columns))}\n")
        fh.write(f"non-DA superclusters: {len(ref_cols)}\n\n")
        for lab, t in [("DA-SPECIFIC (predict RNA up in DA)", spec),
                       ("DA-ABSENT (predict RNA down in DA)", absent)]:
            if len(t):
                fh.write(f"{lab}: n={len(t)}, concordant={t.concordant.sum()} "
                         f"({100*t.concordant.mean():.0f}%), median RNA log2FC={t.RNA_log2FC_DA.median():+.2f}\n")
        fh.write("\nper-gene:\n")
        for _, r in res.iterrows():
            fh.write(f"  {r.gene:12s} {r.direction:8s} ATACd={r.atac_delta_pct:+6.1f} "
                     f"RNA_DA={r.RNA_DA_expr:5.2f} nonDA={r.RNA_nonDA_expr:5.2f} "
                     f"log2FC={r.RNA_log2FC_DA:+5.2f} {'OK' if r.concordant else 'x'}\n")
    log(open(f"{OUT}/validation_summary.txt").read())

    # scatter: ATAC DA-specificity vs RNA DA log2FC
    fig, ax = plt.subplots(figsize=(6.4, 6))
    for d, c in [("specific", "#e6550d"), ("absent", "#3182bd")]:
        t = res[res.direction == d]
        ax.scatter(t.atac_delta_pct, t.RNA_log2FC_DA, c=c, s=28, label=f"DA-{d} (n={len(t)})")
    for _, r in res.iterrows():
        ax.annotate(r.gene, (r.atac_delta_pct, r.RNA_log2FC_DA), fontsize=5.5,
                    xytext=(2, 2), textcoords="offset points")
    ax.axhline(0, c="k", lw=0.6); ax.axvline(0, c="k", lw=0.6)
    ax.set_xlabel("ATAC enhancer DA-specificity  (DA - ref presence %, delta)")
    ax.set_ylabel("RNA log2FC  (DA vs non-DA superclusters)")
    ax.set_title("Enhancer accessibility vs target-gene RNA\n(upper-right / lower-left = concordant)")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(f"{OUT}/scatter_atac_vs_rna.png", dpi=170); plt.close(fig)

    # heatmap: candidate genes x supercluster (z per gene), DA cols first
    order = da_cols + [c for c in expr.columns if c not in da_cols]
    Z = expr[order].sub(expr[order].mean(1), 0).div(expr[order].std(1).replace(0, 1), 0)
    Z = Z.loc[list(spec.gene) + list(absent.gene)]
    fig, ax = plt.subplots(figsize=(max(8, len(order)*0.32), max(6, len(Z)*0.16+1)))
    im = ax.imshow(Z.values, cmap="RdBu_r", aspect="auto", vmin=-2, vmax=2)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=90, fontsize=6)
    for k, c in enumerate(order):
        if c in DA_SC: ax.get_xticklabels()[k].set_color("#e6550d")
    ax.set_yticks(range(len(Z))); ax.set_yticklabels(Z.index, fontsize=5)
    ax.axhline(len(spec)-0.5, color="k", lw=1.5)
    ax.axvline(len(da_cols)-0.5, color="k", lw=1.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01); cb.set_label("z(expr)", fontsize=8)
    ax.set_title("Target-gene RNA across superclusters (z per gene)\nDA superclusters = orange labels, left of black line", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{OUT}/supercluster_expr_heatmap.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    log("VALIDATION COMPLETE -> " + OUT)


if __name__ == "__main__":
    main()
