#!/usr/bin/env python3
"""Build a table of absolute paths to prediction directories per (group, subtype).
Output: predictions/neural_20groups_paths.tsv  (also human-readable stdout)"""
import os
from collections import defaultdict

PRED_ROOT = "/path/to/scgram/predictions"

def group_of(source):
    if source.startswith("ITL23_rows_part"): return "ITL23"
    if source.startswith("cspmei_ASCT_rows_part"): return "ASCT"
    if source == "L6B_rows": return "L6B"
    if source == "ITL34_rows": return "ITL34"
    if source == "ITL4_rows": return "ITL4"
    if source == "ITL45_rows": return "ITL45"
    if source == "CHO_rows": return "CHO"
    if source == "AMY_rows": return "AMY"
    if source in ("Dopaminergic","MSN","FOXP2","PVALB","VIP","CNGA","BFEXA",
                  "BNGA","CNMIX","MGC_1","MGC_2","PV_ChCs"):
        return source
    return None

GROUPS = ["ITL23","ASCT","L6B","ITL34","ITL4","ITL45","CHO","AMY",
          "Dopaminergic","MSN","FOXP2","PVALB","VIP","CNGA","BFEXA",
          "BNGA","CNMIX","MGC_1","MGC_2","PV_ChCs"]

# (group, subtype) -> list of (absolute_dir, n_csvs)
paths = defaultdict(list)

for source in sorted(os.listdir(PRED_ROOT)):
    src_dir = os.path.join(PRED_ROOT, source)
    if not os.path.isdir(src_dir): continue
    g = group_of(source)
    if g is None: continue
    for subtype in sorted(os.listdir(src_dir)):
        st_dir = os.path.join(src_dir, subtype)
        if not os.path.isdir(st_dir): continue
        n = sum(1 for _ in os.scandir(st_dir) if _.name.endswith("_predictions.csv"))
        if n > 0:
            paths[(g, subtype)].append((os.path.abspath(st_dir), n))

# Sort by group order, then subtype
group_order = {g: i for i, g in enumerate(GROUPS)}
ordered = sorted(paths.items(), key=lambda kv: (group_order.get(kv[0][0], 999), kv[0][1]))

# TSV output
tsv_path = os.path.join(PRED_ROOT, "neural_20groups_paths.tsv")
with open(tsv_path, "w") as f:
    f.write("group\tsubtype\tn_dirs\ttotal_cells\tabsolute_paths\n")
    for (g, st), plist in ordered:
        total = sum(n for _, n in plist)
        joined = ";".join(p for p, _ in plist)
        f.write(f"{g}\t{st}\t{len(plist)}\t{total}\t{joined}\n")

# Human-readable stdout
for (g, st), plist in ordered:
    total = sum(n for _, n in plist)
    print(f"\n[{g}] {st}  ({len(plist)} dir(s), {total} cells)")
    for p, n in plist:
        print(f"  {p}  ({n} cells)")

print(f"\n\nSaved TSV: {tsv_path}")
