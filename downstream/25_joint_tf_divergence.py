#!/usr/bin/env python3
"""
25_joint_tf_divergence.py
=========================
JOINT TF occupancy divergence computed over ALL footprints (KNOWN + NOVEL regions
combined, NO region-type limitation), using the same method and parameters as the
known-region analysis (16_TF_divergence_all_subtypes.py). Delivered into
Joint_TF_Divergence/ for parity with the joint co-occurrence deliverable.

Input = the per-nucleus percent-bound matrices
  *BD_results/<subtype>/*_<subtype>_aggregated_data_percent_values_converted.csv
which aggregate bound/total over EVERY footprint in the cell (known regulatory +
gene-body + novel/uncharacterized alike). No region restriction is applied.

Metric (identical to the original):
  within-nucleus rank of each TF by percent-bound (rank 1 = highest), ties -> average;
  Divergence_STD = SD across nuclei of a TF's within-nucleus rank (ddof=1);
  Mean_Rank, CV = Divergence_STD/Mean_Rank; Hartigan's dip test (<=1000 sampled nuclei,
  random_state=42) -> bimodal at BH-FDR < 0.05.

Because the percent-bound matrix already spans all footprints, this reproduces/validates
the original TF_Divergence while being delivered as the explicit joint (known+novel,
all-region) divergence.

Phases (checkpointed):
  1  per-subtype divergence -> per_subtype/<subtype>.csv.gz
  2  cross-subtype (convergent-divergent, subtype x TF matrix, between-subtype variability,
     bimodal counts) + comparison to original + figure

Env: `zeros` (diptest, scipy, matplotlib). Base: /path/to/data
"""
from __future__ import annotations
import os, glob, json, time, argparse
from collections import Counter
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from statsmodels.stats.multitest import multipletests
try:
    from diptest import diptest
except Exception:
    diptest = None

BASE = "/path/to/data"
SUMMARY = f"{BASE}/celltype_subtype_summary.csv"
KNOWN_TFDIV = f"{BASE}/TF_Divergence"
OUT = f"{BASE}/Joint_TF_Divergence"
PER = f"{OUT}/per_subtype"
CROSS = f"{OUT}/Cross"
CKPT = f"{OUT}/.checkpoints"
MONITOR = f"{OUT}/monitor.log"
DIP_SAMPLE = 1000
for d in (OUT, PER, CROSS, CKPT):
    os.makedirs(d, exist_ok=True)


def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True); open(MONITOR, "a").write(line + "\n")


def resolve_matrices(sub):
    pat = f"{BASE}/*BD_results/{sub}/*_{sub}_aggregated_data_percent_values_converted.csv"
    return sorted(g for g in glob.glob(pat) if "CRC" not in g)


def load_matrix(sub):
    """Return (tf_names, matrix [n_tf x n_cell]) over all footprints; NaN->0."""
    files = resolve_matrices(sub)
    if not files:
        return None, None
    mats, ref = [], None
    for f in files:
        df = pd.read_csv(f, index_col=0)
        if ref is None:
            ref = df.index
        else:
            df = df.reindex(ref)
        mats.append(df.values.astype(np.float32))
    matrix = np.nan_to_num(np.concatenate(mats, axis=1), nan=0.0)
    return list(ref), matrix


def divergence_subtype(sub):
    done = f"{CKPT}/tf_{sub}.done"
    outf = f"{PER}/{sub}.csv.gz"
    if os.path.exists(done) and os.path.exists(outf):
        log(f"  [skip] {sub}"); return
    tf_names, M = load_matrix(sub)
    if M is None:
        pd.DataFrame().to_csv(outf); json.dump({"subtype": sub, "n_cells": 0}, open(done, "w")); return
    tf_short = [str(t).split("_MA")[0] for t in tf_names]
    n_tf, n_nuc = M.shape
    # within-nucleus ranks (axis=0 = across TFs within each nucleus); rank1 = highest percent
    ranks = rankdata(-M, method="average", axis=0).astype(np.float32)
    div_std = np.std(ranks, axis=1, ddof=1)
    mean_rank = np.mean(ranks, axis=1)
    cv = np.where(mean_rank > 0, div_std / mean_rank, 0.0)
    occ = (M > 0).mean(axis=1)
    dip_p = np.ones(n_tf)
    if diptest is not None and n_nuc >= 20:
        rng = np.random.default_rng(42)
        idx = rng.choice(n_nuc, size=min(n_nuc, DIP_SAMPLE), replace=False)
        for i in range(n_tf):
            row = ranks[i, idx]
            if np.ptp(row) == 0:
                continue
            try:
                dip_p[i] = diptest(row)[1]
            except Exception:
                pass
    dip_fdr = multipletests(dip_p, method="fdr_bh")[1] if n_nuc >= 20 else dip_p
    df = pd.DataFrame({
        "TF": tf_short, "motif": tf_names, "Divergence_STD": div_std, "Mean_Rank": mean_rank,
        "CV": cv, "occupancy": occ, "dip_p": dip_p, "dip_FDR": dip_fdr,
        "bimodal": dip_fdr < 0.05, "N_nuclei": n_nuc,
    }).sort_values("Divergence_STD", ascending=False).reset_index(drop=True)
    df.to_csv(outf, index=False, compression="gzip")
    json.dump({"subtype": sub, "n_cells": int(n_nuc), "n_bimodal": int((dip_fdr < 0.05).sum())},
              open(done, "w"))
    log(f"  [done] {sub}: n={n_nuc}, bimodal={int((dip_fdr<0.05).sum())}")


def phase1(subs, only):
    log("=== PHASE 1: all-footprint per-subtype TF divergence ===")
    for s in subs:
        if only and s not in only:
            continue
        divergence_subtype(s)


def phase2(subs):
    log("=== PHASE 2: cross-subtype + comparison ===")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    per = {}
    for s in subs:
        f = f"{PER}/{s}.csv.gz"
        if os.path.exists(f):
            d = pd.read_csv(f)
            if len(d):
                per[s] = d.set_index("motif")   # motif is unique (short TF name is not)
    if not per:
        log("  no per-subtype tables"); return
    top_ctr = Counter()
    for s, d in per.items():
        top = d.sort_values("Divergence_STD", ascending=False).head(25)
        top_ctr.update([str(m).split("_MA")[0] for m in top.index.tolist()])
    conv = pd.DataFrame(top_ctr.most_common(), columns=["TF", "n_subtypes_top25"])
    conv.to_csv(f"{CROSS}/convergent_divergent_TFs.csv", index=False)
    allTF = sorted(set().union(*[set(d.index) for d in per.values()]))
    mat = pd.DataFrame(index=list(per.keys()), columns=allTF, dtype=float)
    for s, d in per.items():
        mat.loc[s, d.index] = d["Divergence_STD"].values
    mat = mat.fillna(0.0)
    mat.to_csv(f"{CROSS}/divergence_subtype_by_TF.csv")
    mat.std(axis=0).sort_values(ascending=False).head(50).to_csv(
        f"{CROSS}/top_between_subtype_variable_TFs.csv", header=["between_subtype_SD"])
    pd.DataFrame([{"subtype": s, "n_bimodal": int(d["bimodal"].sum()),
                   "n_cells": int(d["N_nuclei"].iloc[0])} for s, d in per.items()]).to_csv(
        f"{CROSS}/bimodal_counts_per_subtype.csv", index=False)
    # comparison to original TF_Divergence
    cmp = [f"JOINT (all-footprint) convergent-divergent TFs (top-25 in >=N subtypes):",
           conv.head(15).to_string(index=False), ""]
    kf = f"{KNOWN_TFDIV}/Cross_Celltype_Comparison/convergent_divergent_TFs.csv"
    if os.path.exists(kf):
        kn = pd.read_csv(kf); kc = [c for c in kn.columns if "TF" in c][0]
        shared = sorted(set(conv["TF"].head(15)) & set(kn[kc].head(15)))
        cmp += [f"Original TF_Divergence top-15: {sorted(set(kn[kc].head(15)))}",
                f"Joint top-15: {sorted(set(conv['TF'].head(15)))}",
                f"SHARED: {shared}"]
    open(f"{OUT}/joint_vs_original_TF_divergence_comparison.txt", "w").write("\n".join(cmp))
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    top = conv.head(20)
    ax.barh(top["TF"][::-1], top["n_subtypes_top25"][::-1], color="#C44E52")
    ax.set_xlabel("# subtypes where TF is top-25 divergent (all footprints)")
    ax.set_title("Convergent TF divergence across CSPMEI subtypes (joint, all-region)")
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/fig_convergent_divergent_TFs.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    log(f"  wrote cross-subtype + comparison ({len(per)} subtypes)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all", choices=["1", "2", "all"])
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    only = set(args.only.split(",")) if args.only else None
    subs = list(pd.read_csv(SUMMARY)["Subtype"])
    log(f"start: {len(subs)} subtypes, only={only}")
    if args.phase in ("1", "all"):
        phase1(subs, only)
    if args.phase in ("2", "all"):
        phase2(subs)
    log("JOINT TF DIVERGENCE COMPLETE")


if __name__ == "__main__":
    main()
