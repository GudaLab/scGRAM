#!/usr/bin/env python3
"""
26_joint_tf_cooccur.py
======================
JOINT TF co-occurrence — co-repressed / "co-zero" TF modules computed over ALL
footprints (KNOWN + NOVEL regions combined, NO region-type limitation), using the
same method and parameters as the known-region analysis (16_occur_cell_all_subtypes.py).

Input = the per-nucleus percent-bound matrices
  *BD_results/<subtype>/*_<subtype>_aggregated_data_percent_values_converted.csv
which aggregate bound/total over EVERY footprint in the cell (known regulatory +
gene-body + novel/uncharacterized regions alike — ~1.69M footprints/cell). This is
the all-region, not-region-restricted, TF-matrix-based representation the user asked
for ("inclusive of both known + novel regions but not with any region limitation").

  Z[tf, cell] = True  <=>  TF percent-bound == 0 in that cell (jointly-unbound module)

Parameters identical to the known co-zero:
  TOP_TF=50, informative = unbound in 5-95% of cells, min_support=max(10, 2% of cells),
  MAX_SIZE=3 (Apriori).

NOTE: TF co-occurrence is TF-matrix-based (not region-annotated), so there is NO
gene-body / regulatory-only split. Because the percent-bound matrix already spans all
footprints, this run reproduces/validates the original TF_ZeroCooccur while being
delivered as the explicit joint (known+novel, all-region) co-occurrence.

Phases (checkpointed): 1 per-subtype mining -> novel_cooccur/<sub>_cozero_itemsets.csv.gz
                       2 cross-subtype prevalence + known-vs-novel comparison + figures
Env: `zeros`.  Base: /path/to/data
"""
from __future__ import annotations
import os, glob, json, time, argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd

BASE = "/path/to/data"
VOCAB = f"{BASE}/unbound_characetrize/training_data/tf_vocabulary.json"
SUMMARY = f"{BASE}/celltype_subtype_summary.csv"
KNOWN_COZERO = f"{BASE}/TF_ZeroCooccur/Cross/cozero_pair_prevalence.csv"
OUT = f"{BASE}/Joint_TF_ZeroCooccur"
PER = f"{OUT}/per_subtype"
CROSS = f"{OUT}/Cross"
CKPT = f"{OUT}/.checkpoints"
MONITOR = f"{OUT}/monitor.log"
# --- parameters (identical to 16_occur_cell_all_subtypes.py) ---
TOP_TF = 50
ZERO_LO, ZERO_HI = 0.05, 0.95
MIN_SUPPORT_FR, MIN_SUPPORT_N = 0.02, 10
MAX_SIZE = 3
TOP_PLOT = 30
N_WORKERS = 8       # gentle on shared node (TF-divergence run may still be active)
SHARD = 400
for d in (OUT, PER, CROSS, CKPT):
    os.makedirs(d, exist_ok=True)

_v = json.load(open(VOCAB))
TF_LIST = _v["tf_list"] if isinstance(_v, dict) and "tf_list" in _v else _v
N_TF = len(TF_LIST)
def _acc(m): i = m.find("_MA"); return m[i + 1:] if i >= 0 else m
ACC2IDX = {_acc(m): k for k, m in enumerate(TF_LIST)}
TF_NAME = [m.split("_")[0] for m in TF_LIST]


def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True); open(MONITOR, "a").write(line + "\n")


def resolve_matrices(sub):
    """All percent-bound matrix parts for a subtype (all footprints, no region limit)."""
    pat = f"{BASE}/*BD_results/{sub}/*_{sub}_aggregated_data_percent_values_converted.csv"
    return sorted(g for g in glob.glob(pat) if "CRC" not in g)


def build_zero_matrix(sub):
    """Return (tf_names, Z bool [n_tf x n_cell]); True = TF percent-bound==0 (unbound
    across ALL footprints of that cell). Concatenates matrix parts on the nucleus axis."""
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
    M = np.nan_to_num(np.concatenate(mats, axis=1), nan=0.0)   # [n_tf x n_cell]
    return list(ref), (M == 0)


def mine_subtype(sub):
    done = f"{CKPT}/cooccur_{sub}.done"
    outf = f"{PER}/{sub}_cozero_itemsets.csv.gz"
    if os.path.exists(done) and os.path.exists(outf):
        log(f"  [skip] {sub}"); return
    tf_names, Z = build_zero_matrix(sub)
    if Z is None:
        pd.DataFrame().to_csv(outf); json.dump({"subtype": sub, "n_items": 0}, open(done, "w")); return
    tf_short = [str(t).split("_MA")[0] for t in tf_names]
    n_tf, n_cell = Z.shape
    min_support = max(MIN_SUPPORT_N, int(round(MIN_SUPPORT_FR * n_cell)))
    zero_rate = Z.mean(axis=1)
    informative = np.where((zero_rate >= ZERO_LO) & (zero_rate <= ZERO_HI))[0]
    if len(informative) == 0:
        log(f"  {sub}: no informative TFs (n={n_cell})")
        pd.DataFrame().to_csv(outf); json.dump({"subtype": sub, "n_cell": int(n_cell), "n_items": 0}, open(done, "w")); return
    var = zero_rate[informative] * (1 - zero_rate[informative])
    keep = informative[np.argsort(-var)[:TOP_TF]]
    names = [tf_short[i] for i in keep]
    Zf = Z[keep, :]
    Zi = Zf.astype(np.int32)
    k = len(keep)
    rows = []
    s1 = Zf.sum(axis=1)
    freq1 = [a for a in range(k) if s1[a] >= min_support]
    for a in freq1:
        rows.append((1, (names[a],), int(s1[a])))
    freq_pairs = []
    if freq1:
        pc = Zi[freq1] @ Zi[freq1].T
        f1 = len(freq1)
        for ia in range(f1):
            for ib in range(ia + 1, f1):
                c = int(pc[ia, ib])
                if c >= min_support:
                    a, b = freq1[ia], freq1[ib]
                    freq_pairs.append((a, b))
                    rows.append((2, (names[a], names[b]), c))
    if MAX_SIZE >= 3 and freq_pairs:
        fps = set(freq_pairs); seen3 = set()
        for (a, b) in freq_pairs:
            for c in freq1:
                if c == a or c == b:
                    continue
                trip = tuple(sorted((a, b, c)))
                if trip in seen3:
                    continue
                if tuple(sorted((a, c))) not in fps or tuple(sorted((b, c))) not in fps:
                    continue
                cnt = int(np.count_nonzero(Zf[a] & Zf[b] & Zf[c]))
                if cnt >= min_support:
                    rows.append((3, tuple(names[i] for i in trip), cnt))
                seen3.add(trip)
    items = pd.DataFrame(rows, columns=["size", "TFs", "support_count"])
    if items.empty:
        pd.DataFrame().to_csv(outf); json.dump({"subtype": sub, "n_cell": int(n_cell), "n_items": 0}, open(done, "w")); return
    items["support_pct"] = items["support_count"] / n_cell * 100
    items["TFs_str"] = items["TFs"].apply(lambda t: " + ".join(t))
    items = items.sort_values("support_count", ascending=False).reset_index(drop=True)
    items.to_csv(outf, index=False, compression="gzip")
    json.dump({"subtype": sub, "n_cell": int(n_cell), "n_informative": int(len(informative)),
               "min_support": int(min_support), "n_items": int(len(items))}, open(done, "w"))
    log(f"  [done] {sub}: n={n_cell}, informative={len(informative)}, itemsets={len(items)}")


def phase1(subs, only):
    log("=== PHASE 1: novel-region co-zero mining ===")
    for s in subs:
        if only and s not in only:
            continue
        mine_subtype(s)


def phase2(subs):
    log("=== PHASE 2: cross-subtype prevalence + known-vs-novel ===")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    pair_prev = Counter()          # frozenset pair -> n subtypes
    pair_support = {}              # pair -> list of support_pct
    per_counts = []
    for s in subs:
        f = f"{PER}/{s}_cozero_itemsets.csv.gz"
        if not os.path.exists(f):
            continue
        try:
            d = pd.read_csv(f)
        except Exception:
            continue
        if d.empty or "size" not in d:
            per_counts.append({"subtype": s, "n_itemsets": 0}); continue
        per_counts.append({"subtype": s, "n_itemsets": len(d),
                           "n_pairs": int((d["size"] == 2).sum()),
                           "n_triples": int((d["size"] == 3).sum())})
        for _, r in d[d["size"] == 2].iterrows():
            key = tuple(sorted(str(r["TFs_str"]).split(" + ")))
            pair_prev[key] += 1
            pair_support.setdefault(key, []).append(r["support_pct"])
    prev = pd.DataFrame([{"TF_pair": " + ".join(k), "n_subtypes": v,
                          "mean_support_pct": round(np.mean(pair_support[k]), 3),
                          "max_support_pct": round(np.max(pair_support[k]), 3)}
                         for k, v in pair_prev.items()]).sort_values(
        ["n_subtypes", "mean_support_pct"], ascending=False)
    prev.to_csv(f"{CROSS}/cozero_pair_prevalence.csv", index=False)
    pd.DataFrame(per_counts).to_csv(f"{CROSS}/cozero_counts_per_subtype.csv", index=False)

    # known-vs-novel comparison
    cmp = [f"NOVEL-region co-zero: total distinct TF pairs = {len(prev)}",
           "Most widely shared NOVEL co-zero pairs (n_subtypes):",
           prev.head(15).to_string(index=False), ""]
    if os.path.exists(KNOWN_COZERO):
        kn = pd.read_csv(KNOWN_COZERO)
        kcol = [c for c in kn.columns if "pair" in c.lower()]
        kcol = kcol[0] if kcol else kn.columns[0]
        known_top = set(kn[kcol].head(15).apply(lambda x: frozenset(str(x).replace(" + ", ",").replace("+", ",").split(","))))
        novel_top = set(prev["TF_pair"].head(15).apply(lambda x: frozenset(x.split(" + "))))
        cmp += [f"KNOWN top-15 co-zero pairs recorded in {KNOWN_COZERO}",
                f"NOVEL top-15 pairs: {[' + '.join(sorted(p)) for p in list(novel_top)[:15]]}",
                f"SHARED (top-15 known ∩ novel): {[' + '.join(sorted(p)) for p in known_top & novel_top]}"]
    open(f"{OUT}/known_vs_novel_cozero_comparison.txt", "w").write("\n".join(cmp))

    if not prev.empty:
        fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
        top = prev.head(20)
        ax.barh(top["TF_pair"][::-1], top["n_subtypes"][::-1], color="#4C72B0")
        ax.set_xlabel("# subtypes sharing the NOVEL-region co-zero TF pair")
        ax.set_title("Convergent novel-region co-repressed (co-zero) TF pairs")
        for ext in ("png", "pdf"):
            fig.savefig(f"{OUT}/fig_cozero_shared_pairs.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig)
    log(f"  wrote cross-subtype prevalence ({len(prev)} pairs) + comparison")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all", choices=["1", "2", "all"])
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    only = set(args.only.split(",")) if args.only else None
    subs = list(pd.read_csv(SUMMARY)["Subtype"])
    log(f"start: {len(subs)} subtypes, {N_TF} TFs, only={only}")
    if args.phase in ("1", "all"):
        phase1(subs, only)
    if args.phase in ("2", "all"):
        phase2(subs)
    log("JOINT TF CO-OCCURRENCE COMPLETE")


if __name__ == "__main__":
    main()
