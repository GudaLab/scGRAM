#!/usr/bin/env python3
"""
16_occur_cell_all_subtypes.py
=============================
Redesign of occur_cell.py for the full 20-celltype / 49-subtype study.

Original intent (kept): find sets of transcription factors that are JOINTLY
UNBOUND ("zero") in the same nuclei — co-repressed TF modules.

Why a redesign was required
---------------------------
The original enumerated every size-1..3 combination of the zero-TFs in each cell.
With 879 TFs, cells can have hundreds (up to all 879) of zero TFs, so C(z,3)
reaches ~1.1e8 PER CELL and the merged dictionary approaches C(879,3)=1.1e8
keys → tens of GB RAM / hours / OOM. This rewrite instead:

  * restricts to an INFORMATIVE TF universe (the TOP_TF most variable on/off TFs,
    zero-rate in [LO,HI]) — bounded and biologically meaningful,
  * counts co-zero support with VECTORISED boolean algebra (no per-cell combos),
  * applies a MIN-SUPPORT threshold (frequent-itemset / Apriori pruning),
  * caps itemset size at MAX_SIZE.

Result: tractable and identical-in-spirit, runnable across all 49 subtypes.

Outputs (per subtype, under TF_ZeroCooccur/<subtype>/)
  - <st>_cozero_itemsets.csv         all frequent co-zero itemsets (size, TFs, support, %)
  - <st>_top_cozero_combinations.png  top-30 itemsets by support (no label overlap)
  - <st>_pairwise_cozero_heatmap.png  pairwise co-zero % among top TFs (clustermap)
Cross-subtype (TF_ZeroCooccur/Cross/)
  - cozero_pair_prevalence.csv        how many subtypes share each co-zero TF pair

Run: $HOME/.conda/envs/zeros/bin/python -u 16_occur_cell_all_subtypes.py
"""

import os, glob, itertools, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

BASE   = '/path/to/data'
OUTDIR = f'{BASE}/TF_ZeroCooccur'
CROSS  = f'{OUTDIR}/Cross'
os.makedirs(CROSS, exist_ok=True)

# ── mining parameters ──
TOP_TF        = 50      # informative TFs kept for itemset mining (bounds cost)
ZERO_LO       = 0.05    # keep TFs unbound in >=5% of cells ...
ZERO_HI       = 0.95    # ... and <=95% (exclude never/always-zero, uninformative)
MIN_SUPPORT_FR= 0.02    # itemset must be co-zero in >= 2% of cells ...
MIN_SUPPORT_N = 10      # ... and >= 10 cells (absolute floor)
MAX_SIZE      = 3       # max itemset size
TOP_PLOT      = 30      # itemsets shown in the bar plot

# 49-subtype roster (same as TF divergence)
GROUP_SUBTYPES = {
    'ITL23':['ITL23_1','ITL23_2','ITL23_3','ITL23_4','ITL23_5','ITL23_6'],
    'ASCT':['ASCT_1','ASCT_2','ASCT_3'], 'L6B':['L6B_1','L6B_2'],
    'ITL34':['ITL34'], 'ITL4':['ITL4_1','ITL4_2'], 'ITL45':['ITL45_1','ITL45_2'],
    'CHO':['CHO'], 'AMY':['AMY'],
    'Dopaminergic':['D12NAC','D1CaB','D1Pu','D2CaB','D2Pu'],
    'MSN':['MSN_1','MSN_2','MSN_3'],
    'FOXP2':['FOXP2_1','FOXP2_2','FOXP2_3','FOXP2_4'],
    'PVALB':['PVALB_1','PVALB_2','PVALB_3','PVALB_4'],
    'VIP':['VIP_1','VIP_2','VIP_3','VIP_4','VIP_5','VIP_6','VIP_7'],
    'CNGA':['CNGA_1','CNGA_2'],
    'BFEXA':['BFEXA'], 'CNMIX':['CNMIX'], 'BNGA':['BNGA'], 'PV_ChCs':['PV_ChCs'],
    'MGC_2':['MGC_2'], 'MGC_1':['MGC_1'],
}
SUBTYPES = [st for subs in GROUP_SUBTYPES.values() for st in subs]


def resolve_matrices(subtype):
    pat = f'{BASE}/*BD_results/{subtype}/*_{subtype}_aggregated_data_percent_values_converted.csv'
    return sorted(glob.glob(pat))


def load_binary_zero(subtype):
    """Return (tf_names, Z bool[n_tf x n_cell]) where True = TF unbound (==0)."""
    files = resolve_matrices(subtype)
    if not files:
        return None, None
    mats, ref = [], None
    for f in files:
        df = pd.read_csv(f, index_col=0)
        if ref is None: ref = df.index
        else: df = df.reindex(ref)
        mats.append(df.values.astype(np.float32))
    M = np.nan_to_num(np.concatenate(mats, axis=1), nan=0.0)
    return list(ref), (M == 0)


def short(tf):
    return tf.split('_MA')[0]


def mine_subtype(subtype):
    tf_names, Z = load_binary_zero(subtype)
    if Z is None:
        print(f'  SKIP (no matrices): {subtype}'); return None
    n_tf, n_cell = Z.shape
    min_support = max(MIN_SUPPORT_N, int(round(MIN_SUPPORT_FR * n_cell)))

    zero_rate = Z.mean(axis=1)
    informative = np.where((zero_rate >= ZERO_LO) & (zero_rate <= ZERO_HI))[0]
    if len(informative) == 0:
        print(f'  {subtype}: no informative TFs'); return None
    # rank informative TFs by binarised variance p(1-p) (most "switching"), keep TOP_TF
    var = zero_rate[informative] * (1 - zero_rate[informative])
    keep = informative[np.argsort(-var)[:TOP_TF]]
    names = [tf_names[i] for i in keep]
    Zf = Z[keep, :]                       # [k x n_cell] bool
    Zi = Zf.astype(np.int32)
    k = len(keep)

    rows = []
    # size 1
    s1 = Zf.sum(axis=1)
    freq1 = [a for a in range(k) if s1[a] >= min_support]
    for a in freq1:
        rows.append((1, (names[a],), int(s1[a])))
    # size 2 — vectorised co-zero counts among frequent-1 TFs
    pair_count = Zi[freq1] @ Zi[freq1].T   # [f1 x f1]
    freq_pairs = []
    f1 = len(freq1)
    for ia in range(f1):
        for ib in range(ia + 1, f1):
            c = int(pair_count[ia, ib])
            if c >= min_support:
                a, b = freq1[ia], freq1[ib]
                freq_pairs.append((a, b))
                rows.append((2, (names[a], names[b]), c))
    # size 3 — Apriori: extend frequent pairs by a third frequent-1 TF (all sub-pairs frequent)
    if MAX_SIZE >= 3 and freq_pairs:
        freq_pair_set = set(freq_pairs)
        seen3 = set()
        for (a, b) in freq_pairs:
            for c in freq1:
                if c == a or c == b:
                    continue
                trip = tuple(sorted((a, b, c)))
                if trip in seen3:
                    continue
                pa = tuple(sorted((a, c))); pb = tuple(sorted((b, c)))
                if pa not in freq_pair_set or pb not in freq_pair_set:
                    continue          # Apriori prune
                cnt = int(np.count_nonzero(Zf[a] & Zf[b] & Zf[c]))
                if cnt >= min_support:
                    rows.append((3, tuple(names[i] for i in trip), cnt))
                seen3.add(trip)

    items = pd.DataFrame(rows, columns=['size', 'TFs', 'support_count'])
    if items.empty:
        print(f'  {subtype}: no frequent co-zero itemsets at support>={min_support}'); return None
    items['support_pct'] = items['support_count'] / n_cell * 100
    items['TFs_short'] = items['TFs'].apply(lambda t: ' + '.join(short(x) for x in t))
    items = items.sort_values('support_count', ascending=False).reset_index(drop=True)

    return dict(subtype=subtype, n_cell=n_cell, n_inform=len(informative),
                min_support=min_support, names=names, Zf=Zf, items=items)


def plot_subtype(res, odir):
    st, items = res['subtype'], res['items']
    # 1. top itemsets bar (multi-TF combos preferred; show top by support)
    multi = items[items['size'] >= 2].head(TOP_PLOT)
    show = multi if len(multi) >= 5 else items.head(TOP_PLOT)
    labels = [f"{r.TFs_short} (n={r.support_count}, {r.support_pct:.1f}%)"
              for r in show.itertuples()]
    fig, ax = plt.subplots(figsize=(11, max(6, len(show) * 0.34)))
    ax.barh(range(len(show))[::-1], show['support_count'].values,
            color='#2C6FAF', edgecolor='white')
    ax.set_yticks(range(len(show))[::-1])
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel('Nuclei with all TFs jointly unbound (co-zero)', fontsize=10)
    ax.set_title(f'Top co-zero TF modules — {st}\n'
                 f'N={res["n_cell"]} nuclei | support ≥ {res["min_support"]} | size ≤ {MAX_SIZE}',
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(f'{odir}/{st}_top_cozero_combinations.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

    # 2. pairwise co-zero % heatmap among top TFs (clustermap)
    names, Zf = res['names'], res['Zf']
    Zi = Zf.astype(np.int32)
    co = Zi @ Zi.T / res['n_cell'] * 100.0
    co_df = pd.DataFrame(co, index=[short(n) for n in names], columns=[short(n) for n in names])
    k = len(names)
    try:
        g = sns.clustermap(co_df, cmap='magma_r', figsize=(max(10, k*0.22), max(10, k*0.22)),
                           xticklabels=True, yticklabels=True,
                           cbar_kws={'label': 'Co-zero (% of nuclei)'})
        g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(),
                                     rotation=90, fontsize=max(4, min(7, 360//k)))
        g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(),
                                     fontsize=max(4, min(7, 360//k)))
        g.fig.suptitle(f'Pairwise TF Co-zero (joint unbinding) — {st}', y=1.01, fontsize=11)
        g.fig.savefig(f'{odir}/{st}_pairwise_cozero_heatmap.png', dpi=200, bbox_inches='tight')
        plt.close(g.fig)
    except Exception as e:
        print(f'    (heatmap skipped for {st}: {e})')


if __name__ == '__main__':
    print(f'Zero co-occurrence mining across {len(SUBTYPES)} subtypes '
          f'(TOP_TF={TOP_TF}, max_size={MAX_SIZE})')
    pair_prevalence = {}   # frozenset(pair_short) -> n_subtypes
    for st in SUBTYPES:
        odir = f'{OUTDIR}/{st}'
        os.makedirs(odir, exist_ok=True)
        res = mine_subtype(st)
        if res is None:
            continue
        res['items'][['size', 'TFs_short', 'support_count', 'support_pct']].to_csv(
            f'{odir}/{st}_cozero_itemsets.csv', index=False)
        plot_subtype(res, odir)
        nfreq = len(res['items'])
        npair = int((res['items']['size'] == 2).sum())
        ntrip = int((res['items']['size'] == 3).sum())
        print(f'— {st}: N={res["n_cell"]} | informative={res["n_inform"]} | '
              f'itemsets={nfreq} (pairs={npair}, triples={ntrip}) | minsup={res["min_support"]}')
        for r in res['items'][res['items']['size'] == 2].itertuples():
            key = frozenset(r.TFs)
            pair_prevalence[key] = pair_prevalence.get(key, 0) + 1

    # cross-subtype: most widely-shared co-zero pairs
    if pair_prevalence:
        prev = pd.DataFrame(
            [{'TF_pair': ' + '.join(short(x) for x in sorted(k)), 'n_subtypes': v}
             for k, v in pair_prevalence.items()]
        ).sort_values('n_subtypes', ascending=False).reset_index(drop=True)
        prev.to_csv(f'{CROSS}/cozero_pair_prevalence.csv', index=False)
        print(f'\nSaved cross-subtype pair prevalence ({len(prev)} pairs) -> {CROSS}/cozero_pair_prevalence.csv')
    print('\nAll done. Outputs under', OUTDIR)
