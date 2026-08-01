#!/usr/bin/env python3
"""Roll up per-source-subtype TSV to per-canonical-group-subtype table for the 20 neural groups."""
import csv
from collections import defaultdict

TSV = "/path/to/scgram/predictions/per_subtype_summary.tsv"

# Source-part -> canonical group name
# MBGA is a re-aggregation of cells already predicted under standalone sources
# (MDGA, ICGA_1, ICGA_2, SEPGA) + existing CNMIX + a truly-new CTXMIX subtype.
# To avoid double-counting, we only surface MBGA's CTXMIX rows and drop its
# duplicates.
def group_of(source, subtype=None):
    if source.startswith("ITL23_rows_part"): return "ITL23"
    if source.startswith("cspmei_ASCT_rows_part"): return "ASCT"
    if source == "L6B_rows": return "L6B"
    if source == "ITL34_rows": return "ITL34"
    if source == "ITL4_rows": return "ITL4"
    if source == "ITL45_rows": return "ITL45"
    if source == "CHO_rows": return "CHO"
    if source == "AMY_rows": return "AMY"
    if source in ("Dopaminergic","MSN","FOXP2","PVALB","VIP","CNGA","BFEXA",
                  "BNGA","CNMIX","MGC_1","MGC_2","PV_ChCs",
                  "MDGA","ICGA_1","ICGA_2","SEPGA","SIGA"):
        return source
    if source == "MBGA":
        # only CTXMIX is net-new; all other MBGA subtypes duplicate standalone predictions
        if subtype == "CTXMIX":
            return "CTXMIX"
        return None
    return None

# Ordered group list — 20 original + 6 new = 26 groups
GROUPS = ["ITL23","ASCT","L6B","ITL34","ITL4","ITL45","CHO","AMY",
          "Dopaminergic","MSN","FOXP2","PVALB","VIP","CNGA","BFEXA",
          "BNGA","CNMIX","MGC_1","MGC_2","PV_ChCs",
          "MDGA","ICGA_1","ICGA_2","SEPGA","SIGA","CTXMIX"]

# (group, subtype) -> aggregate
agg = defaultdict(lambda: {"n_cells": 0, "total_peaks": 0, "enhancer": 0,
                           "silencer": 0, "both": 0, "neither": 0})

with open(TSV) as f:
    r = csv.DictReader(f, delimiter="\t")
    for row in r:
        g = group_of(row["source"], row["subtype"])
        if g is None: continue
        key = (g, row["subtype"])
        a = agg[key]
        a["n_cells"] += int(row["n_cells"])
        a["total_peaks"] += int(row["total_peaks"])
        a["enhancer"] += int(row["enhancer"])
        a["silencer"] += int(row["silencer"])
        a["both"] += int(row["both"])
        a["neither"] += int(row["neither"])

# Emit ordered by group
group_order = {g: i for i, g in enumerate(GROUPS)}
ordered = sorted(agg.items(), key=lambda kv: (group_order.get(kv[0][0], 999), kv[0][1]))

def pct(num, den):
    return f"{100.0 * num / den:.6f}" if den > 0 else "0.000000"

print("group\tsubtype\tn_cells\ttotal_peaks\tenhancer\tsilencer\tboth\tneither\tpct_enh\tpct_sil\tpct_both")
totals_g = defaultdict(lambda: {"n_cells":0,"total_peaks":0,"enhancer":0,
                                "silencer":0,"both":0,"neither":0,"subtypes":0})
grand = {"n_cells":0,"total_peaks":0,"enhancer":0,"silencer":0,"both":0,"neither":0}

for (g, st), a in ordered:
    tp = a["total_peaks"]
    print(f"{g}\t{st}\t{a['n_cells']}\t{tp}\t{a['enhancer']}\t{a['silencer']}\t{a['both']}\t{a['neither']}"
          f"\t{pct(a['enhancer'], tp)}\t{pct(a['silencer'], tp)}\t{pct(a['both'], tp)}")
    tg = totals_g[g]
    tg["subtypes"] += 1
    for k in ["n_cells","total_peaks","enhancer","silencer","both","neither"]:
        tg[k] += a[k]
        grand[k] += a[k]

# Group-level totals
print("\n\n# GROUP TOTALS")
print("group\tn_subtypes\tn_cells\ttotal_peaks\tenhancer\tsilencer\tboth\tneither\tpct_enh\tpct_sil\tpct_both")
for g in GROUPS:
    if g not in totals_g: continue
    t = totals_g[g]
    tp = t["total_peaks"]
    print(f"{g}\t{t['subtypes']}\t{t['n_cells']}\t{tp}\t{t['enhancer']}\t{t['silencer']}\t{t['both']}\t{t['neither']}"
          f"\t{pct(t['enhancer'], tp)}\t{pct(t['silencer'], tp)}\t{pct(t['both'], tp)}")

print(f"\nGRAND TOTAL")
print(f"n_subtypes={sum(t['subtypes'] for t in totals_g.values())}")
for k in ["n_cells","total_peaks","enhancer","silencer","both","neither"]:
    print(f"{k}={grand[k]}")
tp = grand["total_peaks"]
print(f"pct_enh={pct(grand['enhancer'], tp)}")
print(f"pct_sil={pct(grand['silencer'], tp)}")
print(f"pct_both={pct(grand['both'], tp)}")
