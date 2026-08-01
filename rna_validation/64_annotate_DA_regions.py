#!/usr/bin/env python3
"""Annotate DA-specific / DA-absent regulatory regions to nearest gene (TSS) and
gene-body overlap, using gene.gtf (GENCODE v44 / GRCh38). Adds nearest protein-
coding gene for interpretability. Writes *_annotated.csv next to each table."""
import re, glob
import numpy as np, pandas as pd

D = "/path/to/data"
GTF = f"{D}/gene.gtf"


def load_genes():
    rows = []
    pat_name = re.compile(r'gene_name "([^"]+)"'); pat_type = re.compile(r'gene_type "([^"]+)"')
    with open(GTF) as fh:
        for ln in fh:
            if ln.startswith("#"):
                continue
            f = ln.split("\t")
            if len(f) < 9 or f[2] != "gene":
                continue
            chrom, s, e, strand, attr = f[0], int(f[3]), int(f[4]), f[6], f[8]
            nm = pat_name.search(attr); tp = pat_type.search(attr)
            tss = s if strand == "+" else e
            rows.append((chrom, s, e, tss, nm.group(1) if nm else "NA",
                         tp.group(1) if tp else "NA"))
    g = pd.DataFrame(rows, columns=["chrom", "start", "end", "tss", "gene", "gtype"])
    return g


def build_index(g):
    idx = {}
    for chrom, sub in g.groupby("chrom"):
        sub = sub.sort_values("tss")
        idx[chrom] = {"tss": sub["tss"].values, "gene": sub["gene"].values,
                      "gtype": sub["gtype"].values, "start": sub["start"].values,
                      "end": sub["end"].values}
    return idx


def nearest(idx, chrom, mid, pc_only=False):
    if chrom not in idx:
        return ("NA", np.nan, "NA")
    d = idx[chrom]
    tss, gene, gtype = d["tss"], d["gene"], d["gtype"]
    if pc_only:
        m = gtype == "protein_coding"
        tss, gene, gtype = tss[m], gene[m], gtype[m]
        if len(tss) == 0:
            return ("NA", np.nan, "NA")
    j = np.searchsorted(tss, mid)
    cands = [k for k in (j-1, j) if 0 <= k < len(tss)]
    best = min(cands, key=lambda k: abs(int(tss[k]) - mid))
    return (gene[best], int(mid - tss[best]), gtype[best])


def body_overlap(idx, chrom, mid):
    if chrom not in idx:
        return ""
    d = idx[chrom]
    hit = np.where((d["start"] <= mid) & (d["end"] >= mid))[0]
    if len(hit) == 0:
        return ""
    # prefer protein_coding among overlaps
    genes = [(d["gene"][k], d["gtype"][k]) for k in hit]
    pc = [gn for gn, gt in genes if gt == "protein_coding"]
    return (pc[0] if pc else genes[0][0])


def annotate(path, idx):
    df = pd.read_csv(path)
    if "coord" not in df.columns or len(df) == 0:
        return None
    out = []
    for c in df["coord"]:
        m = re.match(r"(chr[\w]+):(\d+)-(\d+)", str(c))
        if not m:
            out.append(("NA", np.nan, "NA", "NA", np.nan)); continue
        chrom, s, e = m.group(1), int(m.group(2)), int(m.group(3))
        mid = (s + e) // 2
        gn, dist, gt = nearest(idx, chrom, mid)
        pgn, pdist, _ = nearest(idx, chrom, mid, pc_only=True)
        body = body_overlap(idx, chrom, mid)
        out.append((gn, dist, gt, pgn, pdist, body))
    ann = pd.DataFrame(out, columns=["nearest_gene", "dist_to_TSS", "nearest_gtype",
                                     "nearest_pc_gene", "pc_dist_to_TSS", "within_gene"])
    res = pd.concat([df.reset_index(drop=True), ann], axis=1)
    outp = path.replace(".csv", "_annotated.csv")
    res.to_csv(outp, index=False)
    return res


def main():
    print("loading gene models ..."); g = load_genes()
    print(f"  {len(g)} genes ({(g.gtype=='protein_coding').sum()} protein-coding)")
    idx = build_index(g)
    for path in sorted(glob.glob(f"{D}/DA_vs_nonDA/*/DA_specific_regions.csv") +
                       glob.glob(f"{D}/DA_vs_nonDA/*/DA_absent_regions.csv")):
        res = annotate(path, idx)
        if res is None or len(res) == 0:
            print(f"\n{path}: (empty)"); continue
        tag = path.split("/DA_vs_nonDA/")[1]
        print(f"\n===== {tag} (n={len(res)}) =====")
        cols = ["region_class", "coord", "nearest_pc_gene", "pc_dist_to_TSS",
                "within_gene", "DA_mean_pct", "ref_mean_pct", "delta_pct"]
        print(res.head(20)[cols].to_string(index=False))
    print("\nDONE — *_annotated.csv written")


if __name__ == "__main__":
    main()
