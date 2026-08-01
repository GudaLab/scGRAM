#!/usr/bin/env python3
"""Annotate Fig 6 / Suppl 5 case-study coordinates to genes:
  (a) protein-coding gene bodies overlapping the region, and
  (b) the nearest protein-coding TSS to the region midpoint (with distance).
Uses GENCODE gene.gtf (gene features only).
"""
import re

GTF = "/path/to/data/gene.gtf"

# (label, chrom, start, end)
REGIONS = [
    # Supplementary Fig. 5
    ("S5a VIP",        "chr1", 193456815, 193500541),
    ("S5b L6B #1",     "chr5", 113161157, 113171999),
    ("S5b L6B #2",     "chr2", 195451200, 195496793),
    ("S5c ITL4/45 #1", "chr14", 37707309, 37878140),
    ("S5c ITL4/45 #2", "chr14", 37771203, 37915485),
    ("S5c ITL #chr7a", "chr7", 25613810, 25621418),
    ("S5c ITL #chr7b", "chr7", 25613545, 25621234),
    ("S5c ITL #chr13", "chr13", 60753801, 60760799),
    # Figure 6 (a few top-ranked)
    ("F6 top 76.3%",   "chr5", 88659268, 88682400),
    ("F6 FOXP2 66.1%", "chr5", 51378199, 51390638),
    ("F6 53.8%",       "chr1", 193531881, 193716398),
    ("F6 51.6%",       "chr2", 187925925, 188049556),
]

need_chroms = {r[1] for r in REGIONS}

# ---- parse gene features (protein-coding) on needed chromosomes ----
genes = {c: [] for c in need_chroms}   # chrom -> list of (start,end,strand,name,gtype)
name_re = re.compile(r'gene_name "([^"]+)"')
type_re = re.compile(r'gene_type "([^"]+)"')

with open(GTF) as f:
    for line in f:
        if line[0] == "#":
            continue
        # fast reject before splitting
        tab1 = line.find("\t")
        chrom = line[:tab1]
        if chrom not in need_chroms:
            continue
        parts = line.split("\t")
        if parts[2] != "gene":
            continue
        start, end, strand, attr = int(parts[3]), int(parts[4]), parts[6], parts[8]
        gt = type_re.search(attr)
        if not gt or gt.group(1) != "protein_coding":
            continue
        nm = name_re.search(attr)
        genes[chrom].append((start, end, strand, nm.group(1) if nm else "?", "protein_coding"))

for c in genes:
    genes[c].sort()

def annotate(chrom, s, e):
    mid = (s + e) // 2
    gl = genes.get(chrom, [])
    # overlapping gene bodies
    overlaps = [g for g in gl if not (g[1] < s or g[0] > e)]
    # nearest protein-coding TSS to midpoint
    best = None
    for (gs, ge, strand, name, gt) in gl:
        tss = gs if strand == "+" else ge
        dist = abs(tss - mid)
        if best is None or dist < best[0]:
            best = (dist, name, strand, tss)
    return overlaps, best

print(f"{'region':16s} {'locus':30s}  overlap_gene(s)                | nearest_TSS (dist)")
print("-"*110)
for label, chrom, s, e in REGIONS:
    ov, best = annotate(chrom, s, e)
    ov_names = ", ".join(sorted({g[3] for g in ov})) if ov else "(none — intergenic)"
    locus = f"{chrom}:{s:,}-{e:,}"
    nb = f"{best[1]} ({best[0]/1000:.1f} kb, {best[2]})" if best else "?"
    print(f"{label:16s} {locus:30s}  {ov_names:30s} | {nb}")
