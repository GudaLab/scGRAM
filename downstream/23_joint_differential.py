#!/usr/bin/env python3
"""
23_joint_differential.py
========================
JOINT per-group differential on KNOWN + NOVEL regions, in two versions:
  genebody   : known(all region types incl gene_body) + novel(all incl nonregulatory)
  regulatory : known(drop gene_body-only)             + novel(enhancer/silencer/dual only)

Every region tagged by basket in `region_class` and `region_id` prefix:
  known|{enhancer,silencer,promoter,gene_body,other_regulatory}
  novel|{enhancer,silencer,dual,nonregulatory}

KNOWN side reuses the finished per-group differential:
  <group>_differential_unified_celltypes/unified_master_table_celltypes.csv  (Count_/Pct_)
  <group>_differential_unified_celltypes/pairwise/<A>_vs_<B>.csv              (Fisher, reused)
  regulatory_only/ equivalents for the 'regulatory' version.
NOVEL side aggregated in Joint_Analysis_known_novel/novel_agg/<subtype>.csv.gz
  (per-subtype 1 kb-bin loci; n_cells, Pct, consensus_category, mean probs).
  Novel pairwise Fisher is computed here (2x2 occupied/unoccupied per subtype pair).

Denominators (nCells per subtype): known = N_analyzed (celltype_subtype_summary.csv);
novel = total prediction cells (neural_20groups_paths.tsv).

Outputs -> Joint_Differential/<version>/<group>_joint_master.csv.gz
                                        /<group>_pairwise/<A>_vs_<B>.csv.gz
                                        /<group>_pairwise_summary.csv
Checkpointed per (group,version). Env: `zeros`.
"""
from __future__ import annotations
import os, sys, csv, glob, json, time, argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

BASE = "/path/to/data"
NOVEL_AGG = f"{BASE}/Joint_Analysis_known_novel/novel_agg"
PATHS_TSV = f"{BASE}/unbound_characetrize/predictions/neural_20groups_paths.tsv"
SUMMARY = f"{BASE}/celltype_subtype_summary.csv"
OUT = f"{BASE}/Joint_Differential"
CKPT = f"{OUT}/.checkpoints"
MONITOR = f"{OUT}/monitor.log"
MIN_CELLS_GLOBAL = 10
BIN_SIZE = 1000
PSEUDOCOUNT = 0.1
N_WORKERS = 16
CAT2CLASS = {"enhancer": "novel|enhancer", "silencer": "novel|silencer",
             "enhancer_silencer": "novel|dual", "neither": "novel|nonregulatory"}
NOVEL_REG = {"novel|enhancer", "novel|silencer", "novel|dual"}

for d in (OUT, CKPT, f"{OUT}/genebody", f"{OUT}/regulatory"):
    os.makedirs(d, exist_ok=True)


def log(m):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {m}"
    print(line, flush=True)
    open(MONITOR, "a").write(line + "\n")


def simplify_known(rt):
    rt = str(rt).lower()
    if "enhancer" in rt and "gene_body" not in rt: return "known|enhancer"
    if "silencer" in rt and "gene_body" not in rt: return "known|silencer"
    if "promoter" in rt and "gene_body" not in rt: return "known|promoter"
    if rt == "gene_body": return "known|gene_body"
    if "enhancer" in rt: return "known|enhancer"
    if "silencer" in rt: return "known|silencer"
    if "promoter" in rt: return "known|promoter"
    if "gene_body" in rt: return "known|gene_body"
    return "known|other_regulatory"


def roster():
    rows = list(csv.DictReader(open(PATHS_TSV), delimiter="\t"))
    groups = {}
    novel_n = {}
    for r in rows:
        groups.setdefault(r["group"], []).append(r["subtype"])
        novel_n[r["subtype"]] = int(r["total_cells"])
    s = pd.read_csv(SUMMARY)
    known_n = dict(zip(s["Subtype"], s["N_analyzed"]))
    return groups, known_n, novel_n


# ---------------------------------------------------------------- novel merge
def load_novel_group(subs, novel_n):
    """Merge per-subtype novel aggregations to 1 kb loci; return long table with
    Count_/Pct_ per subtype + consensus_category + probs."""
    parts = []
    for sub in subs:
        nf = f"{NOVEL_AGG}/{sub}.csv.gz"
        if not os.path.exists(nf):
            continue
        nd = pd.read_csv(nf, dtype={"Chromosome": str})
        nd = nd[nd["n_cells"] >= MIN_CELLS_GLOBAL].copy()
        if nd.empty:
            continue
        mid = ((nd["Start"] + nd["End"]) // 2 // BIN_SIZE).astype(np.int64)
        nd["binkey"] = nd["Chromosome"] + ":" + mid.astype(str)
        nd["subtype"] = sub
        parts.append(nd)
    if not parts:
        return pd.DataFrame()
    allnov = pd.concat(parts, ignore_index=True)
    coord = allnov.groupby("binkey").agg(Chromosome=("Chromosome", "first"),
                                         Start=("Start", "min"), End=("End", "max")).reset_index()
    w = allnov["n_cells"].values
    for c, s in [("_wpenh", "mean_p_enhancer"), ("_wpsil", "mean_p_silencer"), ("_wntf", "mean_n_tfs")]:
        allnov[c] = allnov[s] * w
    wsum = allnov.groupby("binkey").agg(W=("n_cells", "sum"), spenh=("_wpenh", "sum"),
                                        spsil=("_wpsil", "sum"), sntf=("_wntf", "sum")).reset_index()
    wsum["mean_p_enhancer"] = wsum.spenh / wsum.W
    wsum["mean_p_silencer"] = wsum.spsil / wsum.W
    wsum["mean_n_tfs"] = wsum.sntf / wsum.W
    catvote = (allnov.groupby(["binkey", "consensus_category"])["n_cells"].sum().reset_index()
               .sort_values("n_cells", ascending=False).drop_duplicates("binkey")[["binkey", "consensus_category"]])
    countpiv = allnov.pivot_table(index="binkey", columns="subtype", values="n_cells", aggfunc="sum")
    out = coord.merge(wsum[["binkey", "mean_p_enhancer", "mean_p_silencer", "mean_n_tfs"]], on="binkey").merge(
        catvote, on="binkey")
    for sub in subs:
        cc = f"Count_{sub}"
        out[cc] = out["binkey"].map(countpiv[sub] if sub in countpiv.columns else {}).fillna(0).astype(int)
        out[f"Pct_{sub}"] = out[cc] / novel_n[sub] * 100
    out["region_class"] = out["consensus_category"].map(CAT2CLASS)
    out["basket"] = "novel"
    return out.drop(columns="binkey")


# ---------------------------------------------------------------- novel Fisher
def _fisher_chunk(args):
    a_occ, a_tot, b_occ, b_tot = args
    out = np.empty(len(a_occ))
    for i in range(len(a_occ)):
        try:
            _, p = fisher_exact([[a_occ[i], a_tot - a_occ[i]], [b_occ[i], b_tot - b_occ[i]]])
        except Exception:
            p = 1.0
        out[i] = p
    return out


def novel_pairwise(novel_df, subs, novel_n, ex):
    """Compute novel Fisher for every subtype pair; return {(A,B): DataFrame}."""
    res = {}
    for i in range(len(subs)):
        for j in range(i + 1, len(subs)):
            A, B = subs[i], subs[j]
            nA, nB = novel_n[A], novel_n[B]
            ca = novel_df[f"Count_{A}"].values.astype(int)
            cb = novel_df[f"Count_{B}"].values.astype(int)
            keep = (ca + cb) > 0
            sub = novel_df[keep].reset_index(drop=True)
            caa, cbb = ca[keep], cb[keep]
            # parallel fisher over chunks
            idx = np.array_split(np.arange(len(sub)), max(1, min(N_WORKERS, len(sub) // 5000 + 1)))
            futs = [ex.submit(_fisher_chunk, (caa[k], nA, cbb[k], nB)) for k in idx if len(k)]
            pv = np.concatenate([f.result() for f in futs]) if futs else np.array([])
            if len(pv) == 0:
                res[(A, B)] = pd.DataFrame(); continue
            fdr = multipletests(pv, method="fdr_bh")[1]
            pA = caa / nA * 100; pB = cbb / nB * 100
            df = pd.DataFrame({
                "Chromosome": sub["Chromosome"], "Start": sub["Start"], "End": sub["End"],
                "Region_Type": sub["region_class"],
                f"Count_{A}": caa, f"Pct_{A}": pA, f"Count_{B}": cbb, f"Pct_{B}": pB,
                f"nCells_{A}": nA, f"nCells_{B}": nB,
                "Diff_pct": np.abs(pA - pB),
                "log2FC": np.log2((np.maximum(pA, pB) + PSEUDOCOUNT) / (np.minimum(pA, pB) + PSEUDOCOUNT)),
                "p_value": pv, "FDR": fdr,
                "sig_FDR005": fdr < 0.05, "sig_FDR001": fdr < 0.01, "sig_FDR0001": fdr < 0.001,
                "direction": np.where(pA >= pB, f"up_in_{A}", f"up_in_{B}"),
                "basket": "novel",
            })
            res[(A, B)] = df
    return res


# ---------------------------------------------------------------- per group/version
def build_group_version(group, subs, version, known_n, novel_n, ex):
    tag = f"{group}|{version}"
    done = f"{CKPT}/diff_{group}_{version}.done"
    outdir = f"{OUT}/{version}"
    if os.path.exists(done):
        log(f"  [skip] {tag}"); return
    # ---- KNOWN master ----
    d = f"{BASE}/{group}_differential_unified_celltypes"
    if version == "regulatory":
        kf = f"{d}/regulatory_only/regulatory_only_master_table.csv"
        kpair_dir = f"{d}/regulatory_only/pairwise"
    else:
        kf = f"{d}/unified_master_table_celltypes.csv"
        kpair_dir = f"{d}/pairwise"
    known = pd.read_csv(kf, low_memory=False, dtype={"Chromosome": str})
    known["region_class"] = known["Region_Type"].map(simplify_known)
    known["basket"] = "known"
    # ---- NOVEL master ----
    novel = load_novel_group(subs, novel_n)
    if not novel.empty and version == "regulatory":
        novel = novel[novel["region_class"].isin(NOVEL_REG)].copy()
    # ---- assemble joint master ----
    pcts = [f"Pct_{s}" for s in subs]
    cnts = [f"Count_{s}" for s in subs]
    cols_keep = ["Chromosome", "Start", "End", "basket", "region_class"] + \
                [c for c in cnts + pcts if c in known.columns]
    kd = known[cols_keep].copy()
    for c in ["mean_p_enhancer", "mean_p_silencer", "mean_n_tfs"]:
        kd[c] = np.nan
    if not novel.empty:
        nd = novel[["Chromosome", "Start", "End", "basket", "region_class"] + cnts + pcts +
                   ["mean_p_enhancer", "mean_p_silencer", "mean_n_tfs"]].copy()
        joint = pd.concat([kd, nd], ignore_index=True, sort=False)
    else:
        joint = kd
    for c in pcts:
        if c not in joint: joint[c] = 0.0
    joint[pcts] = joint[pcts].fillna(0.0)
    for c in cnts:
        if c not in joint: joint[c] = 0
    joint[cnts] = joint[cnts].fillna(0).astype(int)
    pv = joint[pcts].values
    joint["Max_pct"] = pv.max(axis=1); joint["Min_pct"] = pv.min(axis=1)
    joint["Diff_pct"] = joint["Max_pct"] - joint["Min_pct"]
    joint["log2FC"] = np.log2((joint["Max_pct"] + PSEUDOCOUNT) / (joint["Min_pct"] + PSEUDOCOUNT))
    joint["Max_Celltype"] = np.array(subs)[pv.argmax(axis=1)]
    joint["region_id"] = (joint["region_class"] + "|" + joint["Chromosome"] + ":" +
                          joint["Start"].astype(str) + "-" + joint["End"].astype(str))
    front = ["region_id", "Chromosome", "Start", "End", "basket", "region_class",
             "Max_pct", "Min_pct", "Diff_pct", "log2FC", "Max_Celltype",
             "mean_p_enhancer", "mean_p_silencer", "mean_n_tfs"]
    joint = joint[front + cnts + pcts]
    joint.to_csv(f"{outdir}/{group}_joint_master.csv.gz", index=False, compression="gzip")

    # ---- pairwise: reuse known + compute novel ----
    if len(subs) >= 2:
        pdir = f"{outdir}/{group}_pairwise"
        os.makedirs(pdir, exist_ok=True)
        novel_res = novel_pairwise(novel, subs, novel_n, ex) if not novel.empty else {}
        summ = []
        for i in range(len(subs)):
            for j in range(i + 1, len(subs)):
                A, B = subs[i], subs[j]
                kp = f"{kpair_dir}/{A}_vs_{B}.csv"
                kp2 = f"{kpair_dir}/{B}_vs_{A}.csv"
                kdf = None
                for cand in (kp, kp2):
                    if os.path.exists(cand):
                        kdf = pd.read_csv(cand, low_memory=False, dtype={"Chromosome": str})
                        kdf["basket"] = "known"
                        break
                nd = novel_res.get((A, B), pd.DataFrame())
                comb = pd.concat([x for x in (kdf, nd) if x is not None and not x.empty],
                                 ignore_index=True, sort=False)
                comb.to_csv(f"{pdir}/{A}_vs_{B}.csv.gz", index=False, compression="gzip")
                s = {"Comparison": f"{A}_vs_{B}", "Total_regions": len(comb),
                     "nCells_A": known_n.get(A), "nCells_B": known_n.get(B)}
                for lv, col in [("0.05", "sig_FDR005"), ("0.01", "sig_FDR001"), ("0.001", "sig_FDR0001")]:
                    if col in comb:
                        s[f"sig_FDR{lv}"] = int(comb[col].fillna(False).astype(bool).sum())
                        s[f"sig_FDR{lv}_known"] = int(comb.loc[comb.basket == "known", col].fillna(False).astype(bool).sum())
                        s[f"sig_FDR{lv}_novel"] = int(comb.loc[comb.basket == "novel", col].fillna(False).astype(bool).sum())
                summ.append(s)
        pd.DataFrame(summ).to_csv(f"{outdir}/{group}_pairwise_summary.csv", index=False)
    with open(done, "w") as fh:
        json.dump({"group": group, "version": version, "n_rows": int(len(joint)),
                   "known": int((joint.basket == "known").sum()),
                   "novel": int((joint.basket == "novel").sum())}, fh)
    log(f"  [done] {tag}: {len(joint):,} rows "
        f"(known {int((joint.basket=='known').sum()):,} / novel {int((joint.basket=='novel').sum()):,})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="both", choices=["genebody", "regulatory", "both"])
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    only = set(args.only.split(",")) if args.only else None
    groups, known_n, novel_n = roster()
    versions = ["genebody", "regulatory"] if args.version == "both" else [args.version]
    log(f"=== JOINT DIFFERENTIAL === groups={len(groups)} versions={versions} only={only}")
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        for grp, subs in groups.items():
            if only and grp not in only:
                continue
            for ver in versions:
                try:
                    build_group_version(grp, subs, ver, known_n, novel_n, ex)
                except Exception as e:
                    import traceback
                    log(f"  ERROR {grp}|{ver}: {e}\n{traceback.format_exc()}")
    log("JOINT DIFFERENTIAL COMPLETE")


if __name__ == "__main__":
    main()
