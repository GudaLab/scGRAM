#!/usr/bin/env python3
"""
Build a detailed summary table of all celltypes/subtypes entering the differential
analysis (CRC excluded), combining:
  - Metadata.tsv QC metrics (TSSe, UQ, doublet_scores, log10UQ, TN, UM)
  - Number of cells actually processed (overlap summary files present on disk)

Output: celltype_subtype_summary.csv  (+ printed to stdout)
"""
import os, glob
import numpy as np
import pandas as pd

BASE = "/path/to/data"

# group -> list of (enhancer_csv dir prefix(es))   subtypes auto-discovered from dirs
# multi-part groups list all parts; their subtype cell counts are summed across parts
GROUPS = {
    # --- newest single-subtype additions (2026-07) ---
    "MDGA":         ["MDGA"],
    "ICGA_1":       ["ICGA_1"],
    "ICGA_2":       ["ICGA_2"],
    "SEPGA":        ["SEPGA"],
    "SIGA":         ["SIGA"],
    "CTXMIX":       ["CTXMIX"],
    # --- new celltypes ---
    "ITL23":        [f"ITL23_rows_part_{p:03d}" for p in (1,2,3,4)],
    "ASCT":         [f"cspmei_ASCT_rows_part_{p:03d}" for p in range(1,9)],
    "L6B":          ["L6B_rows"],
    "ITL34":        ["ITL34_rows"],
    "ITL4":         ["ITL4_rows"],
    "ITL45":        ["ITL45_rows"],
    "CHO":          ["CHO_rows"],
    "AMY":          ["AMY_rows"],
    # --- existing brain multi-subtype ---
    "Dopaminergic": ["Dopaminergic"],
    "MSN":          ["MSN"],
    "FOXP2":        ["FOXP2"],
    "PVALB":        ["PVALB"],
    "VIP":          ["VIP"],
    "CNGA":         ["CNGA"],
    # --- existing single-subtype ---
    "BFEXA":        ["BFEXA"],
    "BNGA":         ["BNGA"],
    "CNMIX":        ["CNMIX"],
    "MGC_1":        ["MGC_1"],
    "MGC_2":        ["MGC_2"],
    "PV_ChCs":      ["PV_ChCs"],
}

def processed_counts(prefixes):
    """subtype -> number of *_summary.csv (enhancer) summed across all prefixes/parts"""
    out = {}
    for pfx in prefixes:
        d = os.path.join(BASE, f"{pfx}_sorted_scBAMs_enhancer_csv")
        if not os.path.isdir(d):
            continue
        for sub in os.listdir(d):
            sd = os.path.join(d, sub)
            if not os.path.isdir(sd):
                continue
            n = len(glob.glob(os.path.join(sd, "*_overlaps_enhancer_summary.csv")))
            out[sub] = out.get(sub, 0) + n
    return out

print("Loading Metadata.tsv ...", flush=True)
meta = pd.read_csv(os.path.join(BASE, "Metadata.tsv"), sep="\t",
                   dtype={"barcode": str, "sample": str, "celltype": str,
                          "subclass": str, "cellclass": str}, low_memory=False)
for c in ["TN","UM","PP","UQ","CM","TSSe","doublet_scores","log10UQ"]:
    meta[c] = pd.to_numeric(meta[c], errors="coerce")

# cspmei ASCT selection (sample,barcode) — the subset actually run
cspmei = pd.read_csv(os.path.join(BASE, "cspmei_ASCT_rows.txt"), sep="\t",
                     dtype=str)
cspmei_keys = set(zip(cspmei["sample"], cspmei["barcode"]))

def qc_block(df):
    if len(df) == 0:
        return dict(TSSe_med=np.nan, UQ_med=np.nan, log10UQ_med=np.nan,
                    dbl_med=np.nan, dbl_hi_pct=np.nan, TN_med=np.nan, UM_med=np.nan)
    return dict(
        TSSe_med   = round(df["TSSe"].median(), 2),
        UQ_med     = int(df["UQ"].median()),
        log10UQ_med= round(df["log10UQ"].median(), 3),
        dbl_med    = round(df["doublet_scores"].median(), 4),
        dbl_hi_pct = round((df["doublet_scores"] > 0.5).mean()*100, 2),
        TN_med     = int(df["TN"].median()),
        UM_med     = int(df["UM"].median()),
    )

rows = []
for group, prefixes in GROUPS.items():
    proc = processed_counts(prefixes)
    # subtypes = union of discovered (on disk) and metadata labels for this group's subs
    subs = sorted(proc.keys()) if proc else []
    if not subs:
        print(f"  WARN {group}: no processed subtypes found on disk")
    for sub in subs:
        sub_meta = meta[meta["celltype"] == sub]
        if group == "ASCT":
            # restrict population to cspmei-selected cells
            sel = sub_meta[[ (s,b) in cspmei_keys
                             for s,b in zip(sub_meta["sample"], sub_meta["barcode"]) ]]
            n_assigned_full = len(sub_meta)
            n_selected = len(sel)
            qc_src = sel
        else:
            n_assigned_full = len(sub_meta)
            n_selected = len(sub_meta)
            qc_src = sub_meta
        cellclass = sub_meta["cellclass"].mode().iat[0] if len(sub_meta) else ""
        subclass  = sub_meta["subclass"].mode().iat[0] if len(sub_meta) else ""
        qc = qc_block(qc_src)
        n_proc = proc.get(sub, 0)
        rows.append(dict(
            Celltype=group, Subtype=sub, CellClass=cellclass, SubClass=subclass,
            N_assigned=n_assigned_full, N_selected=n_selected, N_analyzed=n_proc,
            Pct_analyzed=round(n_proc/n_selected*100,1) if n_selected else np.nan,
            **qc))

df = pd.DataFrame(rows)
out_csv = os.path.join(BASE, "celltype_subtype_summary.csv")
df.to_csv(out_csv, index=False)

# group totals
grp = df.groupby("Celltype", sort=False).agg(
    Subtypes=("Subtype","nunique"),
    N_assigned=("N_assigned","sum"),
    N_selected=("N_selected","sum"),
    N_analyzed=("N_analyzed","sum")).reset_index()
grp["Pct_analyzed"] = round(grp["N_analyzed"]/grp["N_selected"]*100,1)
grp.to_csv(os.path.join(BASE,"celltype_group_summary.csv"), index=False)

pd.set_option("display.width", 200, "display.max_columns", 30, "display.max_rows", 200)
print("\n================ PER-SUBTYPE SUMMARY ================")
print(df.to_string(index=False))
print("\n================ PER-GROUP TOTALS ================")
print(grp.to_string(index=False))
print(f"\nSaved: {out_csv}")
print(f"Saved: {os.path.join(BASE,'celltype_group_summary.csv')}")
print(f"\nGRAND TOTAL: {len(df)} subtypes across {df['Celltype'].nunique()} celltypes | "
      f"analyzed cells = {df['N_analyzed'].sum():,}")
