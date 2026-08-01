#!/usr/bin/env python3
"""
16_TF_divergence_all_subtypes.py
================================
Adapted from TF_Divergence_AllCelltypes.py for the full 20-celltype / 49-subtype
study (CRC excluded). Computes per-TF Divergence_STD (variability of a TF's
within-nucleus rank across nuclei) from TOBIAS percent-bound matrices, then a
cross-celltype comparison.

Key adaptations vs the original
-------------------------------
* BASE -> /path/to/data
* Auto part-merge: ITL23 (4 parts), ASCT (8 cspmei parts), MGC_1 (10 parts;
  part 5 footprinting failed) — matrices are concatenated along nuclei by a
  glob resolver, so no part counts are hardcoded.
* Full 49-subtype roster with a node/colour scheme extended for the new
  celltypes (cortical excitatory, astrocyte, microglia).
* Rank ties handled with scipy.stats.rankdata(method='average') instead of
  argsort(argsort()) — correct handling of tied percent values (toggle via
  RANK_METHOD).
* seaborn 0.13-safe (hue= instead of bare palette=).
* No overlapping text: KDE coloured by node with a node-level legend (not 49
  entries), scaled bar/heatmap fonts/width, decluttered scatter labels.
* Per-subtype checkpoint (skips subtypes whose table already exists).

Run:  $HOME/.conda/envs/zeros/bin/python 16_TF_divergence_all_subtypes.py
"""

import os, glob, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import rankdata
from diptest import diptest
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
BASE   = '/path/to/data'
DIVDIR = f'{BASE}/TF_Divergence'
CROSS  = f'{DIVDIR}/Cross_Celltype_Comparison'
os.makedirs(CROSS, exist_ok=True)

RANK_METHOD = 'average'   # 'average' (correct tie handling) or 'ordinal' (original argsort behaviour)

# Node colours (extended for the new celltypes)
NODE_COLOR = {
    'Striatum':           '#7B2D8B',  # purple
    'Dopaminergic':       '#C0392B',  # red
    'Interneuron':        '#1A8A6E',  # teal
    'Extended_Amygdala':  '#D4AC0D',  # gold
    'Pallidum':           '#2471A3',  # blue
    'Midbrain':           '#E67E22',  # orange
    'Cortical_Excitatory':'#27AE60',  # green   (new: ITL*/L6B/CHO)
    'Astrocyte':          '#E91E63',  # pink    (new: ASCT)
    'Microglia':          '#6E4B3A',  # brown   (new: MGC_1/2)
}

# group -> node
GROUP_NODE = {
    'Dopaminergic':'Dopaminergic',
    'MSN':'Striatum', 'FOXP2':'Striatum', 'CNGA':'Striatum',
    'PVALB':'Interneuron', 'VIP':'Interneuron', 'PV_ChCs':'Interneuron',
    'BFEXA':'Extended_Amygdala', 'AMY':'Extended_Amygdala',
    'BNGA':'Pallidum', 'CNMIX':'Midbrain',
    'ITL23':'Cortical_Excitatory', 'ITL34':'Cortical_Excitatory',
    'ITL4':'Cortical_Excitatory', 'ITL45':'Cortical_Excitatory',
    'L6B':'Cortical_Excitatory', 'CHO':'Cortical_Excitatory',
    'ASCT':'Astrocyte', 'MGC_1':'Microglia', 'MGC_2':'Microglia',
}

# group -> list of subtypes (49 total)
GROUP_SUBTYPES = {
    'ITL23':['ITL23_1','ITL23_2','ITL23_3','ITL23_4','ITL23_5','ITL23_6'],
    'ASCT':['ASCT_1','ASCT_2','ASCT_3'],
    'L6B':['L6B_1','L6B_2'],
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

# Build the (cell_class, subtype, node) roster
SUBTYPES = []
for grp, subs in GROUP_SUBTYPES.items():
    for st in subs:
        SUBTYPES.append((grp, st, GROUP_NODE[grp]))

TOP_HEATMAP  = 25
TOP_VIOLIN   = 20
VIOLIN_NSAMP = 300
TOP_CROSS    = 50
N_TF_EXPECTED = 879

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def resolve_matrices(subtype):
    """All aggregated percent matrices for a subtype across any BD_results dir
    (auto-handles part splitting). Returns sorted list of file paths."""
    pat = f'{BASE}/*BD_results/{subtype}/*_{subtype}_aggregated_data_percent_values_converted.csv'
    return sorted(glob.glob(pat))


def load_subtype_matrix(subtype):
    """Load + concatenate (by nucleus) all parts for a subtype.
    Returns (tf_names, matrix float32 [n_tf x n_nuc], n_parts) or (None,None,0)."""
    files = resolve_matrices(subtype)
    if not files:
        return None, None, 0
    mats, ref_index = [], None
    for f in files:
        df = pd.read_csv(f, index_col=0)
        if ref_index is None:
            ref_index = df.index
        else:
            # align TFs to the reference order (identical across parts, but be safe)
            df = df.reindex(ref_index)
        mats.append(df.values.astype(np.float32))
    matrix = np.concatenate(mats, axis=1)   # nuclei stacked along columns
    # NaN = TF footprint not measured in that nucleus -> treat as 0% bound
    # (matches the fillna(0) convention; without this, rankdata(axis=0) would
    #  propagate NaN across whole nucleus columns and degenerate the result).
    matrix = np.nan_to_num(matrix, nan=0.0)
    return list(ref_index), matrix, len(files)


def out_dir(subtype_id):
    d = f'{DIVDIR}/TF_Divergence_{subtype_id}'
    os.makedirs(d, exist_ok=True)
    return d


def save_close(fig, path):
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'    Saved: {path}')


def declutter_labels(ax, xs, ys, labels, colors, min_dx, min_dy, fontsize=5.5):
    """Greedily place point labels, skipping any that would collide with an
    already-placed one (no adjustText dependency)."""
    placed = []
    for x, y, lab, col in zip(xs, ys, labels, colors):
        if any(abs(x-px) < min_dx and abs(y-py) < min_dy for px, py in placed):
            continue
        ax.annotate(lab, (x, y), fontsize=fontsize, color=col,
                    xytext=(2, 1), textcoords='offset points')
        placed.append((x, y))


gold_brown = matplotlib.colors.LinearSegmentedColormap.from_list(
    'gold_darkbrown', ['#FFD700', '#654321'])


# ══════════════════════════════════════════════════════════════════════════════
# CORE
# ══════════════════════════════════════════════════════════════════════════════
def compute_divergence(subtype_id):
    tf_names, matrix, n_parts = load_subtype_matrix(subtype_id)
    if matrix is None:
        return None, None, None, 0
    n_tfs, n_nuc = matrix.shape

    # Within-nucleus ranking; rank 1 = highest percent bound. Ties -> average
    # (vectorised over nuclei via axis=0).
    if RANK_METHOD == 'average':
        ranks = rankdata(-matrix, method='average', axis=0).astype(np.float32)
    else:  # 'ordinal' reproduces the original argsort(argsort()) behaviour
        ranks = (np.argsort(np.argsort(-matrix, axis=0), axis=0) + 1).astype(np.float32)

    div_std   = np.std(ranks, axis=1, ddof=1)
    mean_rank = np.mean(ranks, axis=1)
    cv        = np.where(mean_rank > 0, div_std / mean_rank, 0.0)

    rng = np.random.default_rng(42)
    samp_idx = rng.choice(n_nuc, size=min(n_nuc, 1000), replace=False)
    dip_pvals = np.array([diptest(ranks[i, samp_idx])[1] for i in range(n_tfs)])

    result = pd.DataFrame({
        'TF': tf_names, 'Divergence_STD': div_std, 'Mean_Rank': mean_rank,
        'CV': cv, 'Dip_pval': dip_pvals, 'Bimodal': dip_pvals < 0.05,
        'N_nuclei': n_nuc,
    }).sort_values('Divergence_STD', ascending=False).reset_index(drop=True)
    return result, ranks, tf_names, n_parts


# ══════════════════════════════════════════════════════════════════════════════
# PER-SUBTYPE PLOTS
# ══════════════════════════════════════════════════════════════════════════════
def plot_subtype(result_df, ranks_matrix, tf_names, subtype_id, node, odir):
    tf_index = {tf: i for i, tf in enumerate(tf_names)}
    n_nuc = ranks_matrix.shape[1]
    n_tf  = ranks_matrix.shape[0]

    # 1. Rank heatmap top-25
    top25 = result_df.head(TOP_HEATMAP)
    row_idx = [tf_index[tf] for tf in top25['TF']]
    data = ranks_matrix[row_idx, :]
    rng = np.random.default_rng(0)
    disp_cols = np.sort(rng.choice(n_nuc, size=min(n_nuc, 80), replace=False))
    data_disp = data[:, disp_cols]

    fig, ax = plt.subplots(figsize=(max(10, len(disp_cols)*0.14), 7))
    im = ax.imshow(data_disp, cmap=gold_brown, aspect='auto')
    plt.colorbar(im, ax=ax, label='Rank (1 = highest binding)')
    ax.set_yticks(np.arange(len(top25)))
    bimodal_set = set(top25.loc[top25['Bimodal'], 'TF'])
    ax.set_yticklabels([tf + (' *' if tf in bimodal_set else '') for tf in top25['TF']], fontsize=7)
    ax.set_xticks([])
    ax.set_xlabel(f'Individual nuclei  (showing {len(disp_cols)} of {n_nuc})', fontsize=8)
    ax.set_title(f'TF Rank Divergence — {subtype_id}  (top {TOP_HEATMAP})\n'
                 f'Node: {node}  |  * = bimodal (dip p<0.05)', fontsize=9, pad=6)
    fig.tight_layout()
    save_close(fig, f'{odir}/{subtype_id}_heatmap_top{TOP_HEATMAP}.png')

    # 2. Violin top-20 (seaborn-0.13 safe: hue=TF, legend off)
    top_v = result_df.head(TOP_VIOLIN)
    row_idx_v = [tf_index[tf] for tf in top_v['TF']]
    samp_nuc = rng.choice(n_nuc, size=min(n_nuc, VIOLIN_NSAMP), replace=False)
    long_rows = []
    for tf, ri, bm in zip(top_v['TF'], row_idx_v, top_v['Bimodal']):
        for rank_val in ranks_matrix[ri, samp_nuc]:
            long_rows.append({'TF': tf, 'Rank': float(rank_val)})
    long = pd.DataFrame(long_rows)
    tf_order = top_v['TF'].tolist()
    palette = {tf: ('#C0392B' if bm else NODE_COLOR[node])
               for tf, bm in zip(top_v['TF'], top_v['Bimodal'])}

    fig, ax = plt.subplots(figsize=(13, max(6, TOP_VIOLIN*0.42)))
    sns.violinplot(data=long, y='TF', x='Rank', orient='h', hue='TF',
                   palette=palette, inner=None, cut=0, linewidth=0.7,
                   order=tf_order, legend=False, ax=ax)
    if n_nuc <= 200:
        sns.stripplot(data=long, y='TF', x='Rank', orient='h', color='black',
                      size=3, alpha=0.6, jitter=False, order=tf_order, ax=ax)
    ax.invert_xaxis()
    ax.axvline(x=n_tf/2, color='grey', linestyle='--', lw=0.7, alpha=0.7)
    ax.set_xlabel('Rank  (1 = highest binding)', fontsize=9)
    ax.set_ylabel('TF', fontsize=9)
    ax.tick_params(axis='y', labelsize=7)
    ax.set_title(f'Within-{subtype_id} TF Rank Distribution  |  N={n_nuc} nuclei\n'
                 f'Top {TOP_VIOLIN} by Divergence_STD  |  red = bimodal  |  '
                 f'shown: {min(n_nuc, VIOLIN_NSAMP)} sampled', fontsize=9)
    fig.tight_layout()
    save_close(fig, f'{odir}/{subtype_id}_violin_top{TOP_VIOLIN}.png')

    # 3. Divergence vs Mean Rank scatter (decluttered labels)
    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(result_df['Mean_Rank'], result_df['Divergence_STD'],
                    c=result_df['CV'], cmap='RdYlGn_r', s=18, alpha=0.55, edgecolors='none')
    plt.colorbar(sc, ax=ax, label='CV (STD / Mean Rank)')
    head = result_df.head(25)
    dx = (result_df['Mean_Rank'].max()-result_df['Mean_Rank'].min())*0.04
    dy = (result_df['Divergence_STD'].max()-result_df['Divergence_STD'].min())*0.025
    declutter_labels(ax, head['Mean_Rank'].values, head['Divergence_STD'].values,
                     [t.split('_MA')[0] for t in head['TF']],
                     ['#8B0000' if b else '#222' for b in head['Bimodal']],
                     dx, dy, fontsize=6)
    ax.axvline(result_df['Mean_Rank'].median(), color='grey', lw=0.7, ls='--', alpha=0.6)
    ax.axhline(result_df['Divergence_STD'].median(), color='grey', lw=0.7, ls='--', alpha=0.6)
    ax.set_xlabel('Mean Rank (lower = higher mean binding)', fontsize=9)
    ax.set_ylabel('Divergence_STD', fontsize=9)
    ax.set_title(f'Divergence vs Mean Rank — {subtype_id}\n'
                 f'Dashed = medians  |  colour = CV  |  red = bimodal', fontsize=9)
    fig.tight_layout()
    save_close(fig, f'{odir}/{subtype_id}_scatter.png')


# ══════════════════════════════════════════════════════════════════════════════
# CROSS-CELL-TYPE COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
def cross_celltype_analysis(summary_rows, per_subtype_dfs):
    summary = pd.DataFrame(summary_rows).sort_values(['node', 'cell_class', 'subtype'])
    n = len(summary)
    colors = [NODE_COLOR[r['node']] for _, r in summary.iterrows()]

    # 1. Mean Top-50 Divergence_STD per subtype (bar)
    fig, ax = plt.subplots(figsize=(max(16, n*0.42), 6.5))
    bars = ax.bar(range(n), summary['mean_top50_std'], color=colors,
                  edgecolor='white', linewidth=0.5)
    ax.set_xticks(range(n))
    ax.set_xticklabels(summary['subtype'], rotation=70, ha='right',
                       fontsize=max(5, min(9, 420//n)))
    ax.set_ylabel('Mean Divergence_STD (top 50 TFs)', fontsize=10)
    ax.set_title('Overall TF Regulatory Divergence per Cell Type / Subtype\n'
                 'Higher = greater within-subtype heterogeneity of TF binding across nuclei',
                 fontsize=11)
    used_nodes = [nn for nn in NODE_COLOR if nn in set(summary['node'])]
    ax.legend(handles=[mpatches.Patch(color=NODE_COLOR[nn], label=nn) for nn in used_nodes],
              fontsize=8, loc='upper right', title='Node', title_fontsize=8, ncol=2)
    for bar, (_, r) in zip(bars, summary.iterrows()):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                f'{r["n_nuclei"]}', ha='center', va='bottom', fontsize=4.5,
                color='#444', rotation=90)
    fig.tight_layout()
    save_close(fig, f'{CROSS}/01_mean_divergence_per_subtype.png')

    # 2. Cross-cell-type heatmap (TFs in top-50 of >=2 subtypes)
    tf_counts = {}
    for sub, df in per_subtype_dfs.items():
        for tf in df.head(TOP_CROSS)['TF']:
            tf_counts[tf] = tf_counts.get(tf, 0) + 1
    shared_tfs = sorted([tf for tf, c in tf_counts.items() if c >= 2],
                        key=lambda tf: -tf_counts[tf])[:60]
    subtype_order = summary['subtype'].tolist()
    cross_mat = pd.DataFrame(index=shared_tfs, columns=subtype_order, dtype=float)
    for sub in subtype_order:
        df = per_subtype_dfs[sub].set_index('TF')
        for tf in shared_tfs:
            cross_mat.loc[tf, sub] = df.loc[tf, 'Divergence_STD'] if tf in df.index else np.nan
    cross_z = cross_mat.subtract(cross_mat.mean(axis=1), axis=0)\
                       .divide(cross_mat.std(axis=1).replace(0, 1), axis=0)

    g = sns.clustermap(cross_z.fillna(0), cmap='RdBu_r', center=0, vmin=-2.5, vmax=2.5,
                       col_cluster=False, figsize=(max(18, n*0.42), max(10, len(shared_tfs)*0.28)),
                       xticklabels=True, yticklabels=True,
                       cbar_kws={'label': 'Divergence_STD (z-score per TF)'},
                       dendrogram_ratio=(0.15, 0.0), colors_ratio=0.0)
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=70, ha='right',
                                 fontsize=max(5, min(8, 420//n)))
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), fontsize=7)
    node_map = dict(zip(summary['subtype'], summary['node']))
    for lbl in g.ax_heatmap.get_xticklabels():
        lbl.set_color(NODE_COLOR.get(node_map.get(lbl.get_text(), ''), 'black'))
    g.fig.suptitle(f'Cross-Cell-Type TF Divergence Heatmap\n'
                   f'TFs in top-{TOP_CROSS} of ≥2 subtypes  |  rows clustered  |  '
                   f'cols by node  |  colour = z-score', y=1.01, fontsize=11)
    g.fig.savefig(f'{CROSS}/02_cross_celltype_heatmap.png', dpi=200, bbox_inches='tight')
    plt.close(g.fig)
    print(f'    Saved: {CROSS}/02_cross_celltype_heatmap.png')

    # 3. Divergence_STD distribution — coloured by NODE with a NODE-level legend
    #    (49-curve spaghetti is avoided; one legend entry per node, not per subtype)
    fig, ax = plt.subplots(figsize=(12, 7))
    for _, r in summary.iterrows():
        sns.kdeplot(per_subtype_dfs[r['subtype']]['Divergence_STD'].values, ax=ax,
                    color=NODE_COLOR[r['node']], linewidth=1.0, alpha=0.5)
    ax.legend(handles=[mpatches.Patch(color=NODE_COLOR[nn], label=nn) for nn in used_nodes],
              fontsize=8, title='Node', title_fontsize=9, ncol=2)
    ax.set_xlabel('Divergence_STD', fontsize=10); ax.set_ylabel('Density', fontsize=10)
    ax.set_title('Distribution of TF Divergence_STD (all TFs) per Subtype\n'
                 'One thin line per subtype, coloured by node', fontsize=11)
    fig.tight_layout()
    save_close(fig, f'{CROSS}/03_divergence_distribution_kde.png')

    # 4. Per-node box contrast (all subtypes, grouped by node)
    long_rows = []
    for _, r in summary.iterrows():
        for val in per_subtype_dfs[r['subtype']]['Divergence_STD'].values:
            long_rows.append({'Subtype': r['subtype'], 'Node': r['node'], 'Divergence_STD': val})
    long_c = pd.DataFrame(long_rows)
    order = summary.sort_values(['node', 'subtype'])['subtype'].tolist()
    pal = {r['subtype']: NODE_COLOR[r['node']] for _, r in summary.iterrows()}
    fig, ax = plt.subplots(figsize=(max(14, n*0.4), 6.5))
    sns.boxplot(data=long_c, x='Subtype', y='Divergence_STD', order=order, hue='Subtype',
                palette=pal, width=0.7, fliersize=1, legend=False, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=70, ha='right',
                       fontsize=max(5, min(9, 420//n)))
    for lbl in ax.get_xticklabels():
        lbl.set_color(NODE_COLOR.get(dict(zip(summary['subtype'], summary['node'])).get(lbl.get_text(), ''), 'black'))
    ax.set_xlabel('Subtype (colour = node)', fontsize=10)
    ax.set_ylabel('Divergence_STD (all TFs)', fontsize=10)
    ax.set_title('TF Regulatory Divergence Across All Subtypes (grouped by node)', fontsize=11)
    ax.legend(handles=[mpatches.Patch(color=NODE_COLOR[nn], label=nn) for nn in used_nodes],
              fontsize=8, title='Node', ncol=2, loc='upper right')
    fig.tight_layout()
    save_close(fig, f'{CROSS}/04_divergence_boxplot_all_subtypes.png')

    # 5. Convergently divergent TFs (top-25 across most subtypes)
    conv_counts = {}
    for sub, df in per_subtype_dfs.items():
        for tf in df.head(25)['TF']:
            conv_counts[tf] = conv_counts.get(tf, 0) + 1
    conv_df = (pd.DataFrame.from_dict(conv_counts, orient='index', columns=['N_subtypes'])
               .sort_values('N_subtypes', ascending=False).head(35).reset_index()
               .rename(columns={'index': 'TF'}))
    conv_df['TF_short'] = conv_df['TF'].str.split('_MA').str[0]
    fig, ax = plt.subplots(figsize=(10, max(8, len(conv_df)*0.28)))
    ax.barh(conv_df['TF_short'][::-1], conv_df['N_subtypes'][::-1],
            color='#5D6D7E', edgecolor='white')
    ax.set_xlabel('Number of subtypes where TF is in top-25 Divergence_STD', fontsize=10)
    ax.set_title('Convergently Divergent TFs Across Cell Types', fontsize=11)
    thr = max(3, int(round(n*0.25)))
    ax.axvline(x=thr, color='#C0392B', ls='--', lw=1, label=f'≥{thr} subtypes')
    ax.legend(fontsize=8); ax.tick_params(axis='y', labelsize=8)
    fig.tight_layout()
    save_close(fig, f'{CROSS}/05_convergent_divergent_TFs.png')

    # CSV outputs
    summary.to_csv(f'{CROSS}/summary_divergence_per_subtype.csv', index=False)
    conv_df.to_csv(f'{CROSS}/convergent_divergent_TFs.csv', index=False)
    cross_mat.to_csv(f'{CROSS}/cross_celltype_divergence_matrix.csv')
    print(f'    Saved cross-comparison CSVs in {CROSS}/')


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    summary_rows, per_subtype_dfs = [], {}
    print(f'TF Divergence across {len(SUBTYPES)} subtypes | RANK_METHOD={RANK_METHOD}')

    for cell_class, subtype_id, node in SUBTYPES:
        odir = out_dir(subtype_id)
        table_all = f'{odir}/{subtype_id}_divergence_table_all.csv'

        files = resolve_matrices(subtype_id)
        if not files:
            print(f'  SKIP (no matrices): {subtype_id}')
            continue
        print(f'\n— {subtype_id} [{node}] — {len(files)} part-matrix(es)')
        result_df, ranks_matrix, tf_names, n_parts = compute_divergence(subtype_id)
        if result_df is None:
            print(f'  SKIP: {subtype_id} load failed')
            continue
        n_nuc = int(result_df['N_nuclei'].iloc[0]); n_bim = int(result_df['Bimodal'].sum())
        print(f'  N nuclei={n_nuc} | parts={n_parts} | bimodal={n_bim} | '
              f'top5={result_df.head(5)["TF"].str.split("_MA").str[0].tolist()}')
        result_df.to_csv(table_all, index=False)
        result_df.head(50).to_csv(f'{odir}/{subtype_id}_divergence_table_top50.csv', index=False)
        plot_subtype(result_df, ranks_matrix, tf_names, subtype_id, node, odir)
        del ranks_matrix

        top50std = result_df.head(50)['Divergence_STD'].mean()
        summary_rows.append({
            'subtype': subtype_id, 'cell_class': cell_class, 'node': node,
            'n_nuclei': int(result_df['N_nuclei'].iloc[0]),
            'mean_top50_std': float(top50std),
            'median_std': float(result_df['Divergence_STD'].median()),
            'n_bimodal': int(result_df['Bimodal'].sum()),
        })
        per_subtype_dfs[subtype_id] = result_df

    print('\n══ CROSS-CELL-TYPE ANALYSIS ══')
    cross_celltype_analysis(summary_rows, per_subtype_dfs)
    print(f'\nAll done. Per-subtype: {DIVDIR}/TF_Divergence_<SUBTYPE>/  |  Cross: {CROSS}/')
