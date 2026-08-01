#!/usr/bin/env python3
"""
18_TF_permutation_test.py
=========================
Calibrated permutation test for BETWEEN-SUBTYPE TF divergence, per multi-subtype
celltype. Answers: "Does this TF's within-subtype rank-heterogeneity
(Divergence_STD) differ across the celltype's subtypes MORE than expected when
nuclei are randomly reassigned to subtypes?" — which, unlike a parametric
between-group test, is NOT inflated by the large nuclei counts (the null is
built at the same n).

Procedure (per celltype with >=2 subtypes)
  1. Load + concatenate the subtypes' TOBIAS percent matrices (fillna 0).
  2. Rank TFs within each nucleus (average ranks) -> R (TF x nucleus). Ranks are
     label-independent, so they are computed ONCE.
  3. Observed: per subtype, Divergence_STD[tf] = SD(R[tf, subtype cols], ddof=1);
     between-subtype statistic  T_obs[tf] = SD over subtypes (ddof=1).
  4. Null: shuffle the nucleus->subtype labels NPERM times (sizes preserved),
     recompute T. Empirical p = (#perm >= obs + 1)/(NPERM+1); z vs null;
     Benjamini-Hochberg FDR across the 879 TFs.

Output (per celltype): TF_Divergence/CrossSubtype/within_<GROUP>_permutation_TFdivergence.csv
        combined:        TF_Divergence/CrossSubtype/permutation_TFdivergence_ALL.csv

Run: $HOME/.conda/envs/zeros/bin/python -u 18_TF_permutation_test.py
"""
import os, glob, time
import numpy as np
import pandas as pd
from scipy.stats import rankdata

BASE = '/path/to/data'
CSUB = f'{BASE}/TF_Divergence/CrossSubtype'
os.makedirs(CSUB, exist_ok=True)
NPERM   = 2000
SIG_FDR = 0.05
rng = np.random.default_rng(0)

GROUP_SUBTYPES = {
    'ITL23':['ITL23_1','ITL23_2','ITL23_3','ITL23_4','ITL23_5','ITL23_6'],
    'ASCT':['ASCT_1','ASCT_2','ASCT_3'],'L6B':['L6B_1','L6B_2'],
    'ITL4':['ITL4_1','ITL4_2'],'ITL45':['ITL45_1','ITL45_2'],
    'Dopaminergic':['D12NAC','D1CaB','D1Pu','D2CaB','D2Pu'],
    'MSN':['MSN_1','MSN_2','MSN_3'],'FOXP2':['FOXP2_1','FOXP2_2','FOXP2_3','FOXP2_4'],
    'PVALB':['PVALB_1','PVALB_2','PVALB_3','PVALB_4'],
    'VIP':['VIP_1','VIP_2','VIP_3','VIP_4','VIP_5','VIP_6','VIP_7'],
    'CNGA':['CNGA_1','CNGA_2'],
}  # only multi-subtype celltypes


def resolve(st):
    return sorted(glob.glob(f'{BASE}/*BD_results/{st}/*_{st}_aggregated_data_percent_values_converted.csv'))


def load_celltype(subs):
    cols, labels, tf_ref = [], [], None
    used = []
    for si, st in enumerate(subs):
        files = resolve(st)
        if not files:
            continue
        for f in files:
            df = pd.read_csv(f, index_col=0)
            tf_ref = df.index if tf_ref is None else tf_ref
            if not df.index.equals(tf_ref):
                df = df.reindex(tf_ref)
            cols.append(np.nan_to_num(df.values.astype(np.float32), nan=0.0))
            labels.extend([len(used)] * df.shape[1])
        used.append(st)
    if not cols:
        return None, None, None, None
    return list(tf_ref), np.concatenate(cols, axis=1), np.asarray(labels), used


def divstd(R, R2, labels, k):
    """TF x k matrix of per-subtype SD(ddof=1) of ranks, via one-hot grouping."""
    n = R.shape[1]
    G = np.zeros((n, k), dtype=np.float32)
    G[np.arange(n), labels] = 1.0
    sizes = G.sum(0)                       # (k,)
    s  = R @ G                             # TF x k  (sum)
    s2 = R2 @ G                            # TF x k  (sum of squares)
    var = (s2 - s*s/sizes) / np.maximum(sizes - 1, 1)
    return np.sqrt(np.clip(var, 0, None))


def perm_block_T(R, R2, labels, k, B, rng):
    """Between-subtype statistic T for a BLOCK of B permutations at once.
    Build a (n x k*B) one-hot of B shuffled label assignments and do a single
    matmul — far faster than looping per permutation. Returns (n_TF x B)."""
    n = len(labels)
    ntf = R.shape[0]
    P = np.zeros((n, k*B), dtype=np.float32)
    ar = np.arange(n)
    for b in range(B):
        P[ar, b*k + rng.permutation(labels)] = 1.0
    S  = R @ P                              # ntf x kB
    S2 = R2 @ P
    sizes = P.sum(0)                        # kB
    var = (S2 - S*S/sizes) / np.maximum(sizes - 1, 1)
    sd = np.sqrt(np.clip(var, 0, None)).reshape(ntf, B, k)
    return sd.std(axis=2, ddof=1)           # ntf x B


def bh(p):
    p = np.asarray(p, float); n = len(p)
    o = np.argsort(p); adj = p[o]*n/np.arange(1, n+1)
    adj = np.minimum.accumulate(adj[::-1])[::-1].clip(0, 1)
    out = np.empty(n); out[o] = adj; return out


all_rows = []
for grp, subs in GROUP_SUBTYPES.items():
    t0 = time.time()
    tf, M, labels, used = load_celltype(subs)
    if M is None or len(used) < 2:
        print(f'SKIP {grp}: <2 subtypes with matrices'); continue
    k = len(used)
    R = rankdata(-M, method='average', axis=0).astype(np.float32)
    R2 = R * R
    T_obs = divstd(R, R2, labels, k).std(axis=1, ddof=1)

    ge = np.zeros(len(tf)); s1 = np.zeros(len(tf)); s2 = np.zeros(len(tf))
    B = 200
    done = 0
    while done < NPERM:
        b = min(B, NPERM - done)
        Tp = perm_block_T(R, R2, labels, k, b, rng)   # ntf x b
        ge += (Tp >= T_obs[:, None]).sum(axis=1)
        s1 += Tp.sum(axis=1); s2 += (Tp*Tp).sum(axis=1)
        done += b
    emp_p = (ge + 1) / (NPERM + 1)
    null_mean = s1 / NPERM
    null_sd = np.sqrt(np.clip(s2/NPERM - null_mean**2, 0, None))
    z = np.where(null_sd > 0, (T_obs - null_mean) / null_sd, 0.0)
    fdr = bh(emp_p)

    res = pd.DataFrame({
        'TF': [t.split('_MA')[0] for t in tf], 'TF_full': tf,
        'between_subtype_T_obs': T_obs, 'null_mean': null_mean, 'null_sd': null_sd,
        'z_score': z, 'perm_emp_p': emp_p, 'perm_FDR': fdr,
        'significant': fdr < SIG_FDR,
    }).sort_values('perm_FDR').reset_index(drop=True)
    res.to_csv(f'{CSUB}/within_{grp}_permutation_TFdivergence.csv', index=False)
    res2 = res.copy(); res2.insert(0, 'Group', grp)
    all_rows.append(res2)
    print(f'{grp}: k={k} nuclei={M.shape[1]} | significant(FDR<{SIG_FDR})={int(res["significant"].sum())} '
          f'| top={res.head(3)["TF"].tolist()} ({time.time()-t0:.0f}s)', flush=True)

pd.concat(all_rows, ignore_index=True).to_csv(f'{CSUB}/permutation_TFdivergence_ALL.csv', index=False)
print(f'\nDone. Per-celltype + combined permutation results in {CSUB}/')
