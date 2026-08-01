#!/usr/bin/env python3
"""
22_joint_analysis_known_novel.py
================================
JOINT ANALYSIS of KNOWN regulatory regions (curated-reference overlaps, already
aggregated in <group>_differential_unified_celltypes/unified_master_table_celltypes.csv)
with NOVEL ML-predicted regions (uncharacterized open+TF-bound peaks predicted by
the enhancer/silencer model, per-cell *_BD_predictions.csv).

Every region in every output is tagged by BASKET:
    known|<class>            e.g. known|enhancer, known|gene_body, known|silencer
    novel|enhancer           model category == 'enhancer'
    novel|silencer           model category == 'silencer'
    novel|dual               model category == 'enhancer_silencer'
    novel|nonregulatory      model category == 'neither'

Design decisions (user-confirmed 2026-07-02):
  * 'neither' novel peaks KEPT as novel|nonregulatory (full denominator).
  * KNOWN occupancy reused from existing unified master tables (fast, consistent).
  * Novel regulatory call = the model's written `category` field, as-is.
  * MIN_CELLS_GLOBAL = 10 applied to novel peaks (reproducibility filter).

Phases (each checkpointed; re-run resumes):
  1. Per-subtype NOVEL aggregation (streaming, sharded-parallel) -> novel_agg/<subtype>.csv.gz
  2. Per-group JOINT master table (known + novel, tagged)       -> Joint_Analysis_known_novel/<group>_joint_master_table.csv.gz
  3. Comprehensive summaries + overlap(rediscovery) + figures

Env: `zeros` conda env.  Base: /path/to/data
"""
from __future__ import annotations
import os, sys, csv, glob, gzip, json, time, argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd

BASE = "/path/to/data"
PATHS_TSV = f"{BASE}/unbound_characetrize/predictions/neural_20groups_paths.tsv"
OUT = f"{BASE}/Joint_Analysis_known_novel"
NOVEL_AGG = f"{OUT}/novel_agg"
CKPT = f"{OUT}/.checkpoints"
MONITOR = f"{OUT}/monitor.log"
MIN_CELLS_GLOBAL = 10
PSEUDOCOUNT = 0.1
N_WORKERS = 16           # modest footprint on shared <node>
SHARD_SIZE = 400         # cells per shard task
BIN_SIZE = 1000          # novel peaks merged into 1 kb midpoint bins (consensus loci).
                         # Raw per-cell Genrich peak coords almost never match exactly across
                         # cells, so exact-coordinate occupancy is meaningless; binning to a
                         # standard scATAC consensus resolution restores recurrence. 500 bp /
                         # 2 kb are alternatives (2 kb over-merges distinct elements).
CATS = ["enhancer", "silencer", "enhancer_silencer", "neither"]
CAT2CLASS = {"enhancer": "novel|enhancer", "silencer": "novel|silencer",
             "enhancer_silencer": "novel|dual", "neither": "novel|nonregulatory"}

for d in (OUT, NOVEL_AGG, CKPT):
    os.makedirs(d, exist_ok=True)


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(MONITOR, "a") as fh:
        fh.write(line + "\n")


def simplify_known(rt: str) -> str:
    rt = str(rt).lower()
    if "gene_body" in rt and "enhancer" not in rt and "silencer" not in rt and "promoter" not in rt:
        return "known|gene_body"
    if "enhancer" in rt:
        return "known|enhancer"
    if "silencer" in rt:
        return "known|silencer"
    if "promoter" in rt:
        return "known|promoter"
    if "gene_body" in rt:
        return "known|gene_body"
    return "known|other_regulatory"


# ---------------------------------------------------------------- load roster
def load_roster():
    rows = list(csv.DictReader(open(PATHS_TSV), delimiter="\t"))
    roster = []  # (group, subtype, [dirs], n_cells)
    for r in rows:
        roster.append((r["group"], r["subtype"], r["absolute_paths"].split(";"),
                       int(r["total_cells"])))
    return roster


# ---------------------------------------------------------------- PHASE 1
def _shard_worker(files):
    """Aggregate one shard of prediction CSVs into a compact dict, keyed by 1 kb
    midpoint bin (chrom, midpoint//BIN_SIZE).
    key -> [n_cells, c_enh, c_sil, c_dual, c_neither, sum_penh, sum_psil, sum_ntfs,
            min_start, max_end]  (a peak/cell contributes once per bin per cell)"""
    agg = {}
    for fp in files:
        seen = set()
        try:
            with open(fp) as fh:
                rd = csv.reader(fh)
                next(rd, None)  # header
                for row in rd:
                    if len(row) < 7:
                        continue
                    try:
                        st = int(row[1]); en = int(row[2])
                        penh = float(row[4]); psil = float(row[5]); ntf = float(row[6])
                    except ValueError:
                        continue
                    key = (row[0], (st + en) // 2 // BIN_SIZE)
                    cat = row[3]
                    a = agg.get(key)
                    if a is None:
                        a = [0, 0, 0, 0, 0, 0.0, 0.0, 0.0, st, en]
                        agg[key] = a
                    if key not in seen:
                        a[0] += 1
                        seen.add(key)
                    if cat == "enhancer":            a[1] += 1
                    elif cat == "silencer":          a[2] += 1
                    elif cat == "enhancer_silencer": a[3] += 1
                    else:                            a[4] += 1
                    a[5] += penh; a[6] += psil; a[7] += ntf
                    if st < a[8]: a[8] = st
                    if en > a[9]: a[9] = en
        except FileNotFoundError:
            continue
    return agg


def _merge_into(master, part):
    for k, a in part.items():
        m = master.get(k)
        if m is None:
            master[k] = a[:]
        else:
            for i in range(8):
                m[i] += a[i]
            if a[8] < m[8]: m[8] = a[8]
            if a[9] > m[9]: m[9] = a[9]


def aggregate_novel_subtype(subtype, dirs, n_cells_reported):
    done = f"{CKPT}/novel_{subtype}.done"
    outf = f"{NOVEL_AGG}/{subtype}.csv.gz"
    if os.path.exists(done) and os.path.exists(outf):
        log(f"  [skip] novel agg {subtype} (checkpoint present)")
        return
    files = []
    for d in dirs:
        files.extend(glob.glob(f"{d}/*_predictions.csv"))
    log(f"  novel agg {subtype}: {len(files)} cells across {len(dirs)} dirs")
    shards = [files[i:i + SHARD_SIZE] for i in range(0, len(files), SHARD_SIZE)]
    master = {}
    n_done = 0
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(_shard_worker, s): len(s) for s in shards}
        for fut in as_completed(futs):
            _merge_into(master, fut.result())
            n_done += futs[fut]
            if n_done % 4000 < SHARD_SIZE:
                log(f"    {subtype}: {n_done}/{len(files)} cells, {len(master):,} unique novel peaks")
    # write
    n_cells = len(files)
    rows = []
    for (chrom, binidx), a in master.items():
        ncell, cenh, csil, cdual, cneither, spenh, spsil, sntf, mns, mxe = a
        cat_counts = {"enhancer": cenh, "silencer": csil, "enhancer_silencer": cdual, "neither": cneither}
        consensus = max(cat_counts, key=cat_counts.get)
        # total category observations across all contributing cells (>= ncell)
        ntot = cenh + csil + cdual + cneither
        rows.append((chrom, int(mns), int(mxe), ncell, round(ncell / n_cells * 100, 6),
                     consensus, cenh, csil, cdual, cneither,
                     round(spenh / ntot, 5), round(spsil / ntot, 5), round(sntf / ntot, 3)))
    df = pd.DataFrame(rows, columns=["Chromosome", "Start", "End", "n_cells", "Pct",
                                     "consensus_category", "c_enh", "c_sil", "c_dual", "c_neither",
                                     "mean_p_enhancer", "mean_p_silencer", "mean_n_tfs"])
    df.to_csv(outf, index=False, compression="gzip")
    with open(done, "w") as fh:
        json.dump({"subtype": subtype, "n_cells": n_cells, "n_unique_novel_peaks": len(df)}, fh)
    log(f"  [done] novel agg {subtype}: {len(df):,} unique peaks, n_cells={n_cells}")


def phase1(roster, only=None):
    log("=== PHASE 1: novel per-subtype aggregation ===")
    for grp, sub, dirs, nc in roster:
        if only and grp not in only and sub not in only:
            continue
        aggregate_novel_subtype(sub, dirs, nc)


# ---------------------------------------------------------------- PHASE 2
def phase2(roster, only=None):
    log("=== PHASE 2: per-group JOINT master tables ===")
    groups = {}
    for grp, sub, dirs, nc in roster:
        groups.setdefault(grp, []).append(sub)
    for grp, subs in groups.items():
        if only and grp not in only:
            continue
        outf = f"{OUT}/{grp}_joint_master_table.csv.gz"
        done = f"{CKPT}/joint_{grp}.done"
        if os.path.exists(done) and os.path.exists(outf):
            log(f"  [skip] joint master {grp}")
            continue
        # ---- KNOWN side (reuse unified master table) ----
        kf = f"{BASE}/{grp}_differential_unified_celltypes/unified_master_table_celltypes.csv"
        known = pd.read_csv(kf, low_memory=False, dtype={"Chromosome": str})
        pct_cols = [c for c in known.columns if c.startswith("Pct_")]
        ksub = [c[4:] for c in pct_cols]
        kd = pd.DataFrame({
            "Chromosome": known["Chromosome"], "Start": known["Start"], "End": known["End"],
            "basket": "known",
            "region_class": known["Region_Type"].map(simplify_known),
            "consensus_category": known["Region_Type"].map(simplify_known),
            "mean_p_enhancer": np.nan, "mean_p_silencer": np.nan, "mean_n_tfs": np.nan,
        })
        for c in pct_cols:
            kd[c] = known[c]

        # ---- NOVEL side (combine subtypes, filter n_cells>=10) — fully vectorized ----
        parts = []
        for sub in subs:
            nf = f"{NOVEL_AGG}/{sub}.csv.gz"
            if not os.path.exists(nf):
                log(f"    WARN missing novel agg for {sub}; skipping")
                continue
            nd = pd.read_csv(nf, dtype={"Chromosome": str})
            nd = nd[nd["n_cells"] >= MIN_CELLS_GLOBAL].copy()
            if nd.empty:
                continue
            # bin key on midpoint so different subtypes' spans collapse to same locus
            mid = ((nd["Start"] + nd["End"]) // 2 // BIN_SIZE).astype(np.int64)
            nd["binkey"] = nd["Chromosome"] + ":" + mid.astype(str)
            nd["subtype"] = sub
            parts.append(nd)
        if parts:
            allnov = pd.concat(parts, ignore_index=True)
            # per-locus coordinate span (min start, max end across contributing subtypes)
            coord = allnov.groupby("binkey").agg(
                Chromosome=("Chromosome", "first"), Start=("Start", "min"), End=("End", "max")).reset_index()
            # weighted probs & mean n_tfs (weight = n_cells)
            w = allnov["n_cells"].values
            allnov["_wpenh"] = allnov["mean_p_enhancer"] * w
            allnov["_wpsil"] = allnov["mean_p_silencer"] * w
            allnov["_wntf"] = allnov["mean_n_tfs"] * w
            wsum = allnov.groupby("binkey").agg(
                W=("n_cells", "sum"), spenh=("_wpenh", "sum"), spsil=("_wpsil", "sum"),
                sntf=("_wntf", "sum")).reset_index()
            wsum["mean_p_enhancer"] = wsum["spenh"] / wsum["W"]
            wsum["mean_p_silencer"] = wsum["spsil"] / wsum["W"]
            wsum["mean_n_tfs"] = wsum["sntf"] / wsum["W"]
            # consensus category = n_cells-weighted majority vote across subtypes
            catvote = (allnov.groupby(["binkey", "consensus_category"])["n_cells"].sum()
                       .reset_index().sort_values("n_cells", ascending=False)
                       .drop_duplicates("binkey")[["binkey", "consensus_category"]])
            # per-subtype Pct pivot
            pctpiv = allnov.pivot_table(index="binkey", columns="subtype", values="Pct",
                                        aggfunc="max").reset_index()
            pctpiv.columns = ["binkey"] + [f"Pct_{c}" for c in pctpiv.columns[1:]]
            nd_out = (coord.merge(wsum[["binkey", "mean_p_enhancer", "mean_p_silencer", "mean_n_tfs"]], on="binkey")
                            .merge(catvote, on="binkey").merge(pctpiv, on="binkey"))
            nd_out["basket"] = "novel"
            nd_out["region_class"] = nd_out["consensus_category"].map(CAT2CLASS)
            nd_out = nd_out.drop(columns="binkey")
        else:
            nd_out = pd.DataFrame(columns=kd.columns)

        # ---- combine, align Pct columns, compute Max/Min/Diff/log2FC across subtypes ----
        joint = pd.concat([kd, nd_out], ignore_index=True, sort=False)
        all_pct = [f"Pct_{s}" for s in subs]
        for c in all_pct:
            if c not in joint.columns:
                joint[c] = np.nan
        pctvals = joint[all_pct].fillna(0.0).values
        joint["Max_pct"] = pctvals.max(axis=1)
        joint["Min_pct"] = pctvals.min(axis=1)
        joint["Diff_pct"] = joint["Max_pct"] - joint["Min_pct"]
        joint["log2FC"] = np.log2((joint["Max_pct"] + PSEUDOCOUNT) / (joint["Min_pct"] + PSEUDOCOUNT))
        joint["Max_Celltype"] = np.array(all_pct)[pctvals.argmax(axis=1)]
        joint["Max_Celltype"] = joint["Max_Celltype"].str.replace("Pct_", "", regex=False)
        joint["region_id"] = (joint["region_class"] + "|" + joint["Chromosome"] + ":" +
                              joint["Start"].astype(str) + "-" + joint["End"].astype(str))
        front = ["region_id", "Chromosome", "Start", "End", "basket", "region_class",
                 "consensus_category", "Max_pct", "Min_pct", "Diff_pct", "log2FC", "Max_Celltype",
                 "mean_p_enhancer", "mean_p_silencer", "mean_n_tfs"]
        joint = joint[front + all_pct]
        joint.to_csv(outf, index=False, compression="gzip")
        with open(done, "w") as fh:
            json.dump({"group": grp, "n_rows": len(joint),
                       "known": int((joint.basket == "known").sum()),
                       "novel": int((joint.basket == "novel").sum())}, fh)
        log(f"  [done] joint {grp}: {len(joint):,} rows "
            f"(known {int((joint.basket=='known').sum()):,} / novel {int((joint.basket=='novel').sum()):,})")


# ---------------------------------------------------------------- PHASE 3
def interval_overlap_count(a, b):
    """Count rows in `a` (Chromosome,Start,End) that overlap any interval in `b`.
    Simple per-chrom sweep. Returns boolean mask over a."""
    mask = np.zeros(len(a), dtype=bool)
    b_by = {c: g[["Start", "End"]].sort_values("Start").values for c, g in b.groupby("Chromosome")}
    a_idx = a.reset_index(drop=True)
    for c, g in a_idx.groupby("Chromosome"):
        bv = b_by.get(c)
        if bv is None:
            continue
        starts = bv[:, 0]; ends = bv[:, 1]
        order = np.argsort(starts); starts = starts[order]; ends = ends[order]
        maxend = np.maximum.accumulate(ends)
        for i, (s, e) in zip(g.index, g[["Start", "End"]].values):
            j = np.searchsorted(starts, e, side="right")
            if j > 0 and maxend[j - 1] > s:
                mask[i] = True
    return mask


def phase3(roster, only=None):
    log("=== PHASE 3: comprehensive joint summaries ===")
    groups = {}
    for grp, sub, dirs, nc in roster:
        groups.setdefault(grp, []).append(sub)
    comp_rows, overlap_rows = [], []
    top_novel = []
    for grp, subs in groups.items():
        if only and grp not in only:
            continue
        jf = f"{OUT}/{grp}_joint_master_table.csv.gz"
        if not os.path.exists(jf):
            continue
        j = pd.read_csv(jf, low_memory=False, dtype={"Chromosome": str})
        # composition per group
        for basket in ["known", "novel"]:
            sub_j = j[j.basket == basket]
            for cls, n in sub_j["region_class"].value_counts().items():
                comp_rows.append({"group": grp, "basket": basket, "region_class": cls,
                                  "n_regions": int(n), "pct_of_group": round(n / len(j) * 100, 3)})
        # rediscovery: novel-regulatory peaks overlapping KNOWN regions of this group
        known = j[j.basket == "known"][["Chromosome", "Start", "End"]]
        novel_reg = j[(j.basket == "novel") & (j.region_class != "novel|nonregulatory")]
        if len(novel_reg) and len(known):
            m = interval_overlap_count(novel_reg[["Chromosome", "Start", "End"]], known)
            overlap_rows.append({"group": grp, "n_novel_regulatory": len(novel_reg),
                                 "n_overlapping_known": int(m.sum()),
                                 "pct_rediscovered": round(m.sum() / len(novel_reg) * 100, 2)})
        # top novel differential peaks (multi-subtype groups only)
        if len(subs) >= 2:
            nv = j[(j.basket == "novel") & (j.region_class != "novel|nonregulatory")]
            for _, r in nv.nlargest(15, "Diff_pct").iterrows():
                top_novel.append({"group": grp, "region_id": r["region_id"],
                                  "consensus_category": r["consensus_category"],
                                  "Max_pct": r["Max_pct"], "Min_pct": r["Min_pct"],
                                  "Diff_pct": r["Diff_pct"], "log2FC": r["log2FC"],
                                  "Max_Celltype": r["Max_Celltype"],
                                  "mean_p_enhancer": r["mean_p_enhancer"],
                                  "mean_p_silencer": r["mean_p_silencer"]})
        log(f"  phase3 {grp}: {len(j):,} joint rows summarised")
    pd.DataFrame(comp_rows).to_csv(f"{OUT}/composition_known_vs_novel_per_group.csv", index=False)
    pd.DataFrame(overlap_rows).to_csv(f"{OUT}/novel_vs_known_rediscovery_per_group.csv", index=False)
    pd.DataFrame(top_novel).to_csv(f"{OUT}/top_novel_differential_peaks.csv", index=False)
    log("  wrote composition / rediscovery / top-novel tables")
    make_figures()


def make_figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    comp = pd.read_csv(f"{OUT}/composition_known_vs_novel_per_group.csv")
    if comp.empty:
        return
    # stacked composition: known vs novel-reg vs novel-nonreg per group
    comp["macro"] = np.where(comp.basket == "known", "known",
                    np.where(comp.region_class == "novel|nonregulatory", "novel_nonregulatory", "novel_regulatory"))
    piv = comp.groupby(["group", "macro"])["n_regions"].sum().unstack(fill_value=0)
    piv = piv.loc[piv.sum(axis=1).sort_values(ascending=False).index]
    order = ["known", "novel_regulatory", "novel_nonregulatory"]
    order = [c for c in order if c in piv.columns]
    colors = {"known": "#4C72B0", "novel_regulatory": "#C44E52", "novel_nonregulatory": "#BBBBBB"}
    fig, ax = plt.subplots(figsize=(13, 6), constrained_layout=True)
    bottom = np.zeros(len(piv))
    for c in order:
        ax.bar(piv.index, piv[c], bottom=bottom, label=c, color=colors[c])
        bottom += piv[c].values
    ax.set_xticklabels(piv.index, rotation=45, ha="right")
    ax.set_ylabel("Regions (union across subtypes)")
    ax.set_title("Joint Analysis | known vs novel regulatory region content per group")
    ax.legend(frameon=False)
    fig.savefig(f"{OUT}/fig_composition_known_vs_novel.png", dpi=300)
    fig.savefig(f"{OUT}/fig_composition_known_vs_novel.pdf")
    plt.close(fig)
    log("  wrote fig_composition_known_vs_novel.(png|pdf)")


def write_readme():
    txt = f"""JOINT ANALYSIS — known regulatory regions + novel ML-predicted regions
=====================================================================
Base: {OUT}

BASKET TAGS (region_class column, and prefix of region_id):
  known|enhancer, known|silencer, known|promoter, known|gene_body, known|other_regulatory
  novel|enhancer, novel|silencer, novel|dual, novel|nonregulatory

INPUTS
  known : <group>_differential_unified_celltypes/unified_master_table_celltypes.csv (reused)
  novel : unbound_characetrize/predictions/<dir>/<subtype>/*_BD_predictions.csv (ML model)

FILTERS
  novel peaks kept only if present in >= {MIN_CELLS_GLOBAL} cells of a subtype (MIN_CELLS_GLOBAL)
  novel regulatory call = model 'category' field (enhancer/silencer/enhancer_silencer/neither)

OUTPUTS
  novel_agg/<subtype>.csv.gz            per-subtype novel peak aggregation (Pct, category, mean probs)
  <group>_joint_master_table.csv.gz     per-group union of known+novel, tagged, with Max/Min/Diff/log2FC
  composition_known_vs_novel_per_group.csv
  novel_vs_known_rediscovery_per_group.csv   novel-regulatory peaks overlapping this group's KNOWN regions
  top_novel_differential_peaks.csv           top novel differential peaks per multi-subtype group
  fig_composition_known_vs_novel.(png|pdf)
"""
    open(f"{OUT}/README.txt", "w").write(txt)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all", choices=["1", "2", "3", "all"])
    ap.add_argument("--only", default=None, help="comma-sep group/subtype filter (testing)")
    args = ap.parse_args()
    only = set(args.only.split(",")) if args.only else None
    roster = load_roster()
    log(f"roster: {len(roster)} subtypes, {len(set(r[0] for r in roster))} groups; only={only}")
    if args.phase in ("1", "all"):
        phase1(roster, only)
    if args.phase in ("2", "all"):
        phase2(roster, only)
    if args.phase in ("3", "all"):
        phase3(roster, only)
        write_readme()
    log("ALL REQUESTED PHASES COMPLETE")


if __name__ == "__main__":
    main()
