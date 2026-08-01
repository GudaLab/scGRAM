#!/usr/bin/env python3
"""
Generate master tables for single-subtype groups (BFEXA, CNMIX, BNGA, PV_ChCs, MGC_1, MGC_2).

These groups have only 1 subtype each, so pairwise differential analysis isn't meaningful.
Instead, we:
  1. Read all 5 per-cell data sources
  2. Count region presence across cells
  3. Filter by MIN_CELLS_GLOBAL
  4. Save a master table with Pct_<subtype> column
  5. Also save a regulatory_only variant (excluding gene_body-only regions)

Output matches the format expected by 14_cross_group_brain.py so these groups
can be included as standalone "celltypes" in the cross-group analysis.

Usage:
  nohup $HOME/.conda/envs/snapatac2_env/bin/python \
    /path/to/data/14_single_subtype_master_tables.py \
    >> /path/to/data/14_single_subtype.out 2>&1 &
"""

import os, sys, glob, time
import pandas as pd
import numpy as np
from collections import defaultdict

BASE_DIR = "/path/to/data"

# Process order: smallest first so we can verify quickly
SINGLE_GROUPS = ["CTXMIX", "MDGA", "SEPGA", "ICGA_1", "SIGA", "ICGA_2"]

DATA_SOURCES = [
    {"name":"enhancer", "suffix":"_sorted_scBAMs_enhancer_csv",
     "pattern":"*_overlaps_enhancer_summary.csv", "tag":"enhancer",
     "is_bed":False, "type_col":None},
    {"name":"gtf", "suffix":"_sorted_scBAMs_GTF_csv",
     "pattern":"*_overlaps_GTF_summary.csv", "tag":"gene_body",
     "is_bed":False, "type_col":None},
    {"name":"promoter", "suffix":"_sorted_scBAMs_PROMOTER_csv",
     "pattern":"*_overlaps_PROMOTER_summary.csv", "tag":"promoter",
     "is_bed":False, "type_col":None},
    {"name":"regulome", "suffix":"_sorted_scBAMs_REGULOME_csv",
     "pattern":"*_overlaps_REGULOME_summary.csv", "tag":None,
     "is_bed":False, "type_col":"Regulome_type"},
    {"name":"unknown", "suffix":"_sorted_scBAMs_UNKNOWN_TFBS",
     "pattern":"*_BD_uncharacterized.bed", "tag":"uncharacterized",
     "is_bed":True, "type_col":None},
]

MIN_CELLS_GLOBAL = 10

# --- helpers ---
def cell_id(fp):
    b = os.path.splitext(os.path.basename(fp))[0]
    i = b.find("_BD_")
    return b[:i] if i > 0 else b

def _read_csv_regions(fp):
    try:
        df = pd.read_csv(fp, usecols=["Chromosome","Start","End"],
                         low_memory=False, dtype={"Chromosome":str})
    except: return set()
    if df.empty: return set()
    df["Start"] = pd.to_numeric(df["Start"], errors="coerce")
    df["End"]   = pd.to_numeric(df["End"],   errors="coerce")
    df = df.dropna(subset=["Start","End"])
    if df.empty: return set()
    return set(zip(df["Chromosome"], df["Start"].astype(int), df["End"].astype(int)))

def _read_regulome(fp):
    try: df = pd.read_csv(fp, low_memory=False, dtype={"Chromosome":str})
    except: return {}
    if df.empty or "Chromosome" not in df.columns: return {}
    df["Start"] = pd.to_numeric(df["Start"], errors="coerce")
    df["End"]   = pd.to_numeric(df["End"],   errors="coerce")
    df = df.dropna(subset=["Chromosome","Start","End"])
    if df.empty: return {}
    out = {}
    if "Regulome_type" in df.columns:
        for c,s,e,t in zip(df["Chromosome"], df["Start"].astype(int),
                           df["End"].astype(int), df["Regulome_type"].fillna("regulome")):
            ts = str(t).strip()
            out[(c,s,e)] = ts if ts not in ("","nan","None") else "regulome"
    else:
        for c,s,e in zip(df["Chromosome"], df["Start"].astype(int), df["End"].astype(int)):
            out[(c,s,e)] = "regulome"
    return out

def _read_bed_peaks(fp):
    try:
        df = pd.read_csv(fp, sep="\t", header=None, comment="#",
                         usecols=[6,7,8], engine="c", dtype={6:str})
    except: return set()
    if df.empty: return set()
    df.columns = ["Chromosome","Start","End"]
    df["Start"] = pd.to_numeric(df["Start"], errors="coerce")
    df["End"]   = pd.to_numeric(df["End"],   errors="coerce")
    df = df.dropna()
    if df.empty: return set()
    return set(zip(df["Chromosome"], df["Start"].astype(int), df["End"].astype(int)))

def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def process_group(pfx):
    t0 = time.time()
    log(f"=== {pfx} ===")

    # Find the one subtype directory
    subtype = None
    for src in DATA_SOURCES:
        d = os.path.join(BASE_DIR, f"{pfx}{src['suffix']}")
        if os.path.isdir(d):
            subs = [s for s in os.listdir(d) if os.path.isdir(os.path.join(d, s))]
            if subs:
                subtype = subs[0]
                break
    if subtype is None:
        log(f"  SKIP {pfx}: no subtype directory found")
        return False

    out_dir = os.path.join(BASE_DIR, f"{pfx}_differential_unified_celltypes")
    os.makedirs(out_dir, exist_ok=True)
    reg_dir = os.path.join(out_dir, "regulatory_only")
    os.makedirs(reg_dir, exist_ok=True)

    # Skip if already processed
    reg_master = os.path.join(reg_dir, "regulatory_only_master_table.csv")
    if os.path.exists(reg_master):
        log(f"  SKIP {pfx}: regulatory_only_master_table.csv already exists")
        return True

    # Build cell -> {source -> filepath} mapping
    cell_files = defaultdict(dict)
    for src in DATA_SOURCES:
        d = os.path.join(BASE_DIR, f"{pfx}{src['suffix']}", subtype)
        if not os.path.isdir(d):
            continue
        files = glob.glob(os.path.join(d, src["pattern"]))
        for fp in files:
            cell_files[cell_id(fp)][src["name"]] = fp

    n_cells_total = len(cell_files)
    log(f"  {pfx} subtype={subtype}, {n_cells_total:,} unique cells")

    if n_cells_total == 0:
        log(f"  SKIP {pfx}: no cells")
        return False

    # Aggregate regions across cells
    global_counts = defaultdict(int)
    region_type_map = defaultdict(set)

    src_by_name = {s["name"]: s for s in DATA_SOURCES}
    n_processed = 0
    report_every = max(100, n_cells_total // 20)

    for cid, srcs in cell_files.items():
        cell_regs = set()
        cell_types = defaultdict(set)

        for sname, fp in srcs.items():
            src = src_by_name[sname]
            if src["is_bed"]:
                for reg in _read_bed_peaks(fp):
                    cell_regs.add(reg); cell_types[reg].add(src["tag"])
            elif src["type_col"]:
                for reg, rtype in _read_regulome(fp).items():
                    cell_regs.add(reg); cell_types[reg].add(rtype)
            else:
                for reg in _read_csv_regions(fp):
                    cell_regs.add(reg); cell_types[reg].add(src["tag"])

        for reg in cell_regs:
            global_counts[reg] += 1
            region_type_map[reg].update(cell_types[reg])

        n_processed += 1
        if n_processed % report_every == 0:
            log(f"  {pfx}: {n_processed:,}/{n_cells_total:,} cells processed "
                f"({len(global_counts):,} unique regions so far)")

    log(f"  {pfx}: {n_processed:,} cells done, {len(global_counts):,} total regions")

    # Filter by MIN_CELLS_GLOBAL
    filtered = [r for r, c in global_counts.items() if c >= MIN_CELLS_GLOBAL]
    log(f"  {pfx}: {len(filtered):,} regions >= {MIN_CELLS_GLOBAL} cells")

    if not filtered:
        log(f"  SKIP {pfx}: no regions survived filter")
        return False

    # Build master table
    rows = []
    for reg in sorted(filtered):
        c, s, e = reg
        types = sorted(region_type_map[reg])
        rtype = ";".join(types)
        cnt = global_counts[reg]
        pct = cnt / n_cells_total * 100.0

        rows.append({
            "Chromosome": c, "Start": s, "End": e,
            "Region_Type": rtype, "All_Types": rtype,
            f"Count_{subtype}": cnt,
            f"Pct_{subtype}": pct,
            "Max_pct": pct, "Min_pct": pct,
            "Diff_pct": pct,   # proxy: use Pct as "Diff" so nlargest works for ranking
            "log2FC": 0.0,
            "Max_Celltype": subtype,
            "ExclusiveCelltype": subtype,
        })

    master = pd.DataFrame(rows).sort_values("Diff_pct", ascending=False)

    # Save full master table
    master_path = os.path.join(out_dir, "unified_master_table_celltypes.csv")
    master.to_csv(master_path, index=False)
    log(f"  Saved: {master_path} ({len(master):,} rows)")

    # Save filtered regions CSV (for consistency with multi-subtype pipeline)
    filt_rows = []
    for reg in sorted(filtered):
        c, s, e = reg
        filt_rows.append({
            "Chromosome": c, "Start": s, "End": e,
            "Region_Type": ";".join(sorted(region_type_map[reg])),
            "Global_Count": global_counts[reg]
        })
    pd.DataFrame(filt_rows).to_csv(
        os.path.join(out_dir, f"filtered_regions_ge{MIN_CELLS_GLOBAL}.csv"), index=False)

    # Save regulatory_only version (drop regions that are ONLY gene_body)
    def _is_regulatory(rtype_str):
        types_set = set(rtype_str.split(";"))
        non_genebody = {t for t in types_set if t != "gene_body"}
        return len(non_genebody) > 0

    reg_only = master[master["Region_Type"].apply(_is_regulatory)].copy()
    reg_only.to_csv(reg_master, index=False)
    log(f"  Saved: {reg_master} ({len(reg_only):,} rows)")

    # Summary stats
    log(f"  {pfx} DONE in {(time.time()-t0)/60:.1f} min  |  "
        f"cells={n_cells_total:,} regions={len(master):,} reg_only={len(reg_only):,}")
    return True


# ---- Main ----
log(f"Processing {len(SINGLE_GROUPS)} single-subtype groups: {SINGLE_GROUPS}")
done = 0
for pfx in SINGLE_GROUPS:
    try:
        if process_group(pfx):
            done += 1
    except Exception as e:
        log(f"  ERROR {pfx}: {e}")
        import traceback
        traceback.print_exc()
        continue

log(f"COMPLETE: {done}/{len(SINGLE_GROUPS)} groups processed")
