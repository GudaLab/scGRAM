#!/usr/bin/env python3
"""Walk predictions/<source>/<subtype>/*.csv and aggregate per-subtype counts."""
import os, glob, csv, sys, json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

PRED_ROOT = "/path/to/scgram/predictions"

# Map each raw source-part -> canonical group name
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
                  "BNGA","CNMIX","MGC_2","PV_ChCs","MGC_1"): return source
    return source

def count_one(csv_path):
    """Return (n_peaks, {cat: cnt})."""
    cnt = defaultdict(int)
    n_peaks = 0
    with open(csv_path) as f:
        next(f, None)  # header
        for line in f:
            fields = line.rstrip("\n").split(",")
            if len(fields) < 4: continue
            cnt[fields[3]] += 1
            n_peaks += 1
    return n_peaks, dict(cnt)

def process_subtype(args):
    source, subtype = args
    subtype_dir = os.path.join(PRED_ROOT, source, subtype)
    files = glob.glob(os.path.join(subtype_dir, "*_predictions.csv"))
    n_cells = len(files)
    total_peaks = 0
    cat_totals = defaultdict(int)
    for fp in files:
        n_p, cnt = count_one(fp)
        total_peaks += n_p
        for k, v in cnt.items():
            cat_totals[k] += v
    return source, subtype, n_cells, total_peaks, dict(cat_totals)

def main():
    # Enumerate (source, subtype) pairs
    pairs = []
    for source in sorted(os.listdir(PRED_ROOT)):
        source_dir = os.path.join(PRED_ROOT, source)
        if not os.path.isdir(source_dir): continue
        for subtype in sorted(os.listdir(source_dir)):
            if os.path.isdir(os.path.join(source_dir, subtype)):
                pairs.append((source, subtype))

    print(f"# subtypes to aggregate: {len(pairs)}", file=sys.stderr)

    # Group under canonical group name
    # (group, subtype) -> [n_cells, total_peaks, cat_totals]
    by_gs = defaultdict(lambda: [0, 0, defaultdict(int)])

    with ProcessPoolExecutor(max_workers=12) as ex:
        futs = {ex.submit(process_subtype, p): p for p in pairs}
        for i, fut in enumerate(as_completed(futs), 1):
            source, subtype, n_cells, total_peaks, cat_totals = fut.result()
            g = group_of(source)
            entry = by_gs[(g, subtype)]
            entry[0] += n_cells
            entry[1] += total_peaks
            for k, v in cat_totals.items():
                entry[2][k] += v
            if i % 5 == 0 or i == len(pairs):
                print(f"  {i}/{len(pairs)} done", file=sys.stderr)

    # Write JSON
    out = {}
    for (g, st), (n, tp, cats) in sorted(by_gs.items()):
        out.setdefault(g, {})[st] = {
            "n_cells": n,
            "total_peaks": tp,
            "enhancer": cats.get("enhancer", 0),
            "silencer": cats.get("silencer", 0),
            "enhancer_silencer": cats.get("enhancer_silencer", 0),
            "neither": cats.get("neither", 0),
        }
    with open("/path/to/scgram/predictions/per_subtype_summary.json", "w") as f:
        json.dump(out, f, indent=2)
    print("wrote per_subtype_summary.json", file=sys.stderr)

if __name__ == "__main__":
    main()
