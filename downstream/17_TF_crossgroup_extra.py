#!/usr/bin/env python3
"""
17_TF_crossgroup_extra.py
=========================
Adds the missing cross-subtype / cross-celltype TF views on top of the outputs
of 16_TF_divergence_all_subtypes.py and 16_occur_cell_all_subtypes.py, plus a
companion data table for every figure and a FIGURES_INDEX.

Reads only the small per-subtype tables (no re-reading of the big matrices):
  TF_Divergence/TF_Divergence_<ST>/<ST>_divergence_table_all.csv
  TF_ZeroCooccur/<ST>/<ST>_cozero_itemsets.csv

Produces
--------
B. Cross-subtype TF-divergence (all 49 subtypes)
   - crosssub_divergence_clustermap.png  (1 - Spearman of per-TF Divergence_STD profiles)
   - crosssub_divergence_pca.png
   + crosssub_divergence_matrix.csv, crosssub_pca_coordinates.csv
D. Global ranked between-subtype TF divergence (across all 49 subtypes)
   - global_ranked_TF_between_subtype_divergence.png  + .csv (all 879 TFs)
C. Within-celltype ranked TF divergence (each multi-subtype celltype)
   - within_<GROUP>_ranked_TF_divergence.png + heatmap  + combined CSV
UpSet comparisons (which significant / co-zero TFs are shared vs unique)
   - upset_significant_divergent_TFs_by_celltype.png / _by_node.png
   - upset_cozero_TFs_by_celltype.png / _by_node.png
   + membership + unique-per-celltype CSVs
Indexes
   - TF_Divergence/FIGURES_INDEX.csv, TF_ZeroCooccur/FIGURES_INDEX.csv

Run: $HOME/.conda/envs/zeros/bin/python -u 17_TF_crossgroup_extra.py
"""
import os, glob, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import spearmanr
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from upsetplot import UpSet, from_contents
warnings.filterwarnings('ignore')

BASE   = '/path/to/data'
DIVDIR = f'{BASE}/TF_Divergence'
CSUB   = f'{DIVDIR}/CrossSubtype'
ZDIR   = f'{BASE}/TF_ZeroCooccur'
ZCROSS = f'{ZDIR}/Cross'
for d in (CSUB, ZCROSS):
    os.makedirs(d, exist_ok=True)

NODE_COLOR = {
    'Striatum':'#7B2D8B','Dopaminergic':'#C0392B','Interneuron':'#1A8A6E',
    'Extended_Amygdala':'#D4AC0D','Pallidum':'#2471A3','Midbrain':'#E67E22',
    'Cortical_Excitatory':'#27AE60','Astrocyte':'#E91E63','Microglia':'#6E4B3A',
}
GROUP_NODE = {
    'Dopaminergic':'Dopaminergic','MSN':'Striatum','FOXP2':'Striatum','CNGA':'Striatum',
    'PVALB':'Interneuron','VIP':'Interneuron','PV_ChCs':'Interneuron',
    'BFEXA':'Extended_Amygdala','AMY':'Extended_Amygdala','BNGA':'Pallidum','CNMIX':'Midbrain',
    'ITL23':'Cortical_Excitatory','ITL34':'Cortical_Excitatory','ITL4':'Cortical_Excitatory',
    'ITL45':'Cortical_Excitatory','L6B':'Cortical_Excitatory','CHO':'Cortical_Excitatory',
    'ASCT':'Astrocyte','MGC_1':'Microglia','MGC_2':'Microglia',
}
GROUP_SUBTYPES = {
    'ITL23':['ITL23_1','ITL23_2','ITL23_3','ITL23_4','ITL23_5','ITL23_6'],
    'ASCT':['ASCT_1','ASCT_2','ASCT_3'],'L6B':['L6B_1','L6B_2'],
    'ITL34':['ITL34'],'ITL4':['ITL4_1','ITL4_2'],'ITL45':['ITL45_1','ITL45_2'],
    'CHO':['CHO'],'AMY':['AMY'],
    'Dopaminergic':['D12NAC','D1CaB','D1Pu','D2CaB','D2Pu'],
    'MSN':['MSN_1','MSN_2','MSN_3'],'FOXP2':['FOXP2_1','FOXP2_2','FOXP2_3','FOXP2_4'],
    'PVALB':['PVALB_1','PVALB_2','PVALB_3','PVALB_4'],
    'VIP':['VIP_1','VIP_2','VIP_3','VIP_4','VIP_5','VIP_6','VIP_7'],
    'CNGA':['CNGA_1','CNGA_2'],
    'BFEXA':['BFEXA'],'CNMIX':['CNMIX'],'BNGA':['BNGA'],'PV_ChCs':['PV_ChCs'],
    'MGC_2':['MGC_2'],'MGC_1':['MGC_1'],
}
SUB2GROUP = {st: g for g, subs in GROUP_SUBTYPES.items() for st in subs}
SUBTYPES  = [st for subs in GROUP_SUBTYPES.values() for st in subs]
TOP_SIG   = 50    # TFs in a celltype "divergent signature" (for UpSet)


def bh_fdr(p):
    p = np.asarray(p, float); n = len(p)
    o = np.argsort(p); adj = p[o]*n/np.arange(1, n+1)
    adj = np.minimum.accumulate(adj[::-1])[::-1].clip(0, 1)
    out = np.empty(n); out[o] = adj; return out


def short(tf): return tf.split('_MA')[0]

# short labels for UpSet category axis (long names overlap the totals bars)
ABBREV = {
    'Cortical_Excitatory':'CtxExc','Extended_Amygdala':'ExtAmy','Dopaminergic':'DA',
    'Interneuron':'IN','Striatum':'Str','Astrocyte':'Astro','Microglia':'MGC',
    'Pallidum':'Pal','Midbrain':'MidB',
}
def ab(name): return ABBREV.get(name, name)

def savefig_both(fig, png_path):
    """Save PNG and PDF."""
    fig.savefig(png_path, dpi=200, bbox_inches='tight')
    try: fig.savefig(png_path.replace('.png', '.pdf'), bbox_inches='tight')
    except Exception: pass

def label_leaders(ax, xs, ys, labels, fontsize=7, off=0.07):
    """Place labels off their points with thin leader lines + simple repulsion,
    so text never sits on the plotted circles (no adjustText dependency)."""
    import numpy as _np
    xs = _np.asarray(xs, float); ys = _np.asarray(ys, float)
    cx, cy = xs.mean(), ys.mean()
    xr = (xs.max()-xs.min()) or 1.0; yr = (ys.max()-ys.min()) or 1.0
    ang = _np.arctan2(ys-cy, xs-cx)
    lx = xs + _np.cos(ang)*xr*off; ly = ys + _np.sin(ang)*yr*off
    mindx, mindy = 0.085*xr, 0.05*yr
    for _ in range(400):
        moved = False
        for i in range(len(lx)):
            for j in range(i+1, len(lx)):
                if abs(lx[i]-lx[j]) < mindx and abs(ly[i]-ly[j]) < mindy:
                    sh = (mindy - abs(ly[i]-ly[j]))/2 + 1e-9
                    if ly[i] <= ly[j]: ly[i] -= sh; ly[j] += sh
                    else: ly[i] += sh; ly[j] -= sh
                    moved = True
        if not moved: break
    for x, y, xl, yl, lab in zip(xs, ys, lx, ly, labels):
        ax.annotate(lab, (x, y), xytext=(xl, yl), fontsize=fontsize, fontweight='bold',
                    ha='center', va='center', zorder=6,
                    bbox=dict(boxstyle='round,pad=0.12', fc='white', ec='#bbb', alpha=0.8),
                    arrowprops=dict(arrowstyle='-', color='#999', lw=0.5, shrinkA=0, shrinkB=3))
    # expand axes so labels (esp. left/right-pushed ones) stay INSIDE the plot,
    # never spilling over the y-axis tick labels
    allx = _np.concatenate([xs, lx]); ally = _np.concatenate([ys, ly])
    mx = (allx.max()-allx.min())*0.08 or 1.0; my = (ally.max()-ally.min())*0.06 or 1.0
    ax.set_xlim(min(allx.min()-mx, ax.get_xlim()[0]), max(allx.max()+mx, ax.get_xlim()[1]))
    ax.set_ylim(min(ally.min()-my, ax.get_ylim()[0]), max(ally.max()+my, ax.get_ylim()[1]))

# ── load per-subtype divergence tables ──
div = {}
for st in SUBTYPES:
    f = f'{DIVDIR}/TF_Divergence_{st}/{st}_divergence_table_all.csv'
    if os.path.exists(f):
        d = pd.read_csv(f)
        d['FDR'] = bh_fdr(d['Dip_pval'].values)
        div[st] = d
present = [st for st in SUBTYPES if st in div]
print(f'loaded {len(present)} subtype divergence tables')

# DIV matrix: TF x subtype of Divergence_STD (TFs identical across subtypes)
tf_ref = div[present[0]].set_index('TF')['Divergence_STD'].index
DIV = pd.DataFrame({st: div[st].set_index('TF').reindex(tf_ref)['Divergence_STD'] for st in present})
DIV.index = tf_ref
node_of = {st: GROUP_NODE[SUB2GROUP[st]] for st in present}

# ── Statistical significance of TF divergence ──────────────────────────────
# Test: Hartigan's DIP TEST for unimodality on each TF's within-nucleus rank
# distribution (per subtype), Benjamini–Hochberg FDR across the 879 TFs.
# A TF is "significantly divergent (bimodal)" in a subtype if FDR < SIG_FDR.
SIG_FDR = 0.05
from collections import Counter
subtype_sig = {st: set(div[st].loc[div[st]['FDR'] < SIG_FDR, 'TF']) for st in present}

def perm_sig_set(grp):
    """Full-name TF set with significant BETWEEN-SUBTYPE divergence by the
    permutation test (18_TF_permutation_test.py), BH-FDR<0.05. Returns None if
    the permutation result is not available for this celltype."""
    f = f'{CSUB}/within_{grp}_permutation_TFdivergence.csv'
    if not os.path.exists(f):
        return None
    p = pd.read_csv(f)
    return set(p.loc[p['perm_FDR'] < SIG_FDR, 'TF_full'])

def celltype_sig_set(grp):
    """Full-name TF set significant (dip FDR<0.05) in a MAJORITY of the
    celltype's subtypes (≥1 for single-subtype celltypes)."""
    subs = [s for s in GROUP_SUBTYPES[grp] if s in present]
    if not subs:
        return set()
    c = Counter()
    for s in subs:
        c.update(subtype_sig[s])
    need = 1 if len(subs) == 1 else (len(subs) + 1) // 2
    return {tf for tf, k in c.items() if k >= need}

index_rows = []   # FIGURES_INDEX for TF_Divergence
def reg(fig, table, desc): index_rows.append({'Figure': fig, 'Companion_Table': table, 'Description': desc})

# ══════════════════════════════════════════════════════════════════
# B. Cross-subtype TF-divergence (49 x 49)
# ══════════════════════════════════════════════════════════════════
print('B: cross-subtype TF-divergence matrix')
n = len(present)
dist = np.zeros((n, n))
for i in range(n):
    for j in range(i+1, n):
        rho, _ = spearmanr(DIV.iloc[:, i], DIV.iloc[:, j])
        d = 1 - (0 if np.isnan(rho) else rho)
        dist[i, j] = dist[j, i] = d
dist_df = pd.DataFrame(dist, index=present, columns=present)
dist_df.to_csv(f'{CSUB}/crosssub_divergence_matrix.csv')

Z = linkage(squareform(np.clip(dist, 0, None)), method='average')
row_colors = pd.Series([NODE_COLOR[node_of[s]] for s in present], index=present, name='Node')
g = sns.clustermap(dist_df, row_linkage=Z, col_linkage=Z, cmap='RdYlBu_r',
                   row_colors=row_colors, col_colors=row_colors,
                   figsize=(max(16, n*0.32), max(15, n*0.30)),
                   xticklabels=True, yticklabels=True,
                   cbar_kws={'label': 'TF-divergence distance (1 - Spearman)'})
fs = max(4, min(8, 420//n))
g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=90, fontsize=fs)
g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=fs)
used_nodes = [nn for nn in NODE_COLOR if nn in set(node_of.values())]
g.fig.legend(handles=[mpatches.Patch(color=NODE_COLOR[nn], label=nn) for nn in used_nodes],
             title='Node', loc='upper right', bbox_to_anchor=(1.0, 1.0), fontsize=8, ncol=1)
g.fig.suptitle('Cross-Subtype TF Regulatory Divergence (all 49 subtypes)\n'
               'distance = 1 - Spearman of per-TF Divergence_STD profiles', y=1.02, fontsize=13)
g.savefig(f'{CSUB}/crosssub_divergence_clustermap.png', dpi=200, bbox_inches='tight')
plt.close(g.fig)
reg('CrossSubtype/crosssub_divergence_clustermap.png', 'CrossSubtype/crosssub_divergence_matrix.csv',
    '49x49 subtype divergence in TF space (1 - Spearman of Divergence_STD profiles)')

# PCA of subtypes in TF-divergence space (numpy SVD; no sklearn dependency)
X = DIV.T.values.astype(float)
Xc = X - X.mean(axis=0, keepdims=True)
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
coords = U * S
var_ratio = (S**2) / (S**2).sum()
class _P:  # tiny shim so downstream code reads explained_variance_ratio_
    explained_variance_ratio_ = var_ratio
pca = _P()
pco = pd.DataFrame({'Subtype': present, 'Group': [SUB2GROUP[s] for s in present],
                    'Node': [node_of[s] for s in present]})
for k in range(coords.shape[1]):
    pco[f'PC{k+1}'] = coords[:, k]
pco.to_csv(f'{CSUB}/crosssub_pca_coordinates.csv', index=False)
fig, ax = plt.subplots(figsize=(12, 9))
for nn in used_nodes:
    idx = [i for i, s in enumerate(present) if node_of[s] == nn]
    ax.scatter(coords[idx, 0], coords[idx, 1], c=NODE_COLOR[nn], s=60, alpha=0.85,
               edgecolors='k', linewidths=0.4, label=nn)
# group-centroid labels placed OFF the points with leader lines (no overlap on circles)
gx, gy, gl = [], [], []
for g_ in GROUP_SUBTYPES:
    idx = [i for i, s in enumerate(present) if SUB2GROUP[s] == g_]
    if idx:
        gx.append(coords[idx, 0].mean()); gy.append(coords[idx, 1].mean()); gl.append(g_)
label_leaders(ax, gx, gy, gl, fontsize=7.5, off=0.10)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=11)
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=11)
ax.set_title('Subtypes in TF-Divergence Space (PCA)\nlabels = group centroids (leader lines)', fontsize=13)
ax.legend(title='Node', fontsize=8, ncol=2, loc='upper left', bbox_to_anchor=(1.01, 1.0))
fig.tight_layout(); savefig_both(fig, f'{CSUB}/crosssub_divergence_pca.png')
plt.close(fig)
reg('CrossSubtype/crosssub_divergence_pca.png', 'CrossSubtype/crosssub_pca_coordinates.csv',
    'PCA of subtypes by per-TF Divergence_STD profile')

# ══════════════════════════════════════════════════════════════════
# D. Global ranked between-subtype TF divergence
# ══════════════════════════════════════════════════════════════════
print('D: global ranked between-subtype TF divergence')
glob_rank = pd.DataFrame({
    'TF': [short(t) for t in DIV.index], 'TF_full': DIV.index,
    'mean_Divergence_STD': DIV.mean(axis=1).values,
    'between_subtype_STD': DIV.std(axis=1, ddof=1).values,
    'between_subtype_range': (DIV.max(axis=1) - DIV.min(axis=1)).values,
    'max_subtype': DIV.idxmax(axis=1).values, 'min_subtype': DIV.idxmin(axis=1).values,
}).sort_values('between_subtype_STD', ascending=False).reset_index(drop=True)
glob_rank.to_csv(f'{CSUB}/global_ranked_TF_between_subtype_divergence.csv', index=False)
topg = glob_rank.head(30)
fig, ax = plt.subplots(figsize=(10, max(8, len(topg)*0.3)))
ax.barh(topg['TF'][::-1], topg['between_subtype_STD'][::-1], color='#34495E', edgecolor='white')
ax.set_xlabel('Between-subtype STD of Divergence_STD (across all 49 subtypes)', fontsize=10)
ax.set_title('TFs whose regulatory divergence varies most BETWEEN subtypes\n'
             '(globally, across all celltypes)', fontsize=11)
ax.tick_params(axis='y', labelsize=8)
fig.tight_layout(); fig.savefig(f'{CSUB}/global_ranked_TF_between_subtype_divergence.png', dpi=200, bbox_inches='tight')
plt.close(fig)
reg('CrossSubtype/global_ranked_TF_between_subtype_divergence.png',
    'CrossSubtype/global_ranked_TF_between_subtype_divergence.csv',
    'TFs ranked by between-subtype variability of Divergence_STD (all 49 subtypes)')

# ══════════════════════════════════════════════════════════════════
# C. Within-celltype ranked TF divergence (multi-subtype groups)
# ══════════════════════════════════════════════════════════════════
print('C: within-celltype ranked TF divergence')
within_rows = []
for grp, subs in GROUP_SUBTYPES.items():
    subs = [s for s in subs if s in present]
    if len(subs) < 2:
        continue
    sub_div = DIV[subs]
    spread = (sub_div.std(axis=1, ddof=1)).sort_values(ascending=False)
    rk = pd.DataFrame({'TF': [short(t) for t in spread.index], 'TF_full': spread.index,
                       'within_celltype_STD': spread.values,
                       'within_celltype_range': (sub_div.max(axis=1)-sub_div.min(axis=1)).reindex(spread.index).values})
    for _, r in rk.iterrows():
        within_rows.append({'Group': grp, **r.to_dict()})
    # barh top-25
    top = rk.head(25)
    fig, ax = plt.subplots(figsize=(9, max(7, len(top)*0.3)))
    ax.barh(top['TF'][::-1], top['within_celltype_STD'][::-1],
            color=NODE_COLOR[GROUP_NODE[grp]], edgecolor='white')
    ax.set_xlabel(f'Between-subtype STD of Divergence_STD (within {grp})', fontsize=10)
    ax.set_title(f'Top TFs differing across {grp} subtypes (n={len(subs)})', fontsize=11)
    ax.tick_params(axis='y', labelsize=8)
    fig.tight_layout(); fig.savefig(f'{CSUB}/within_{grp}_ranked_TF_divergence.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    # heatmap top-25 TF x subtype Divergence_STD — VALUES annotated in each cell;
    # TF labels get a '*' when significantly divergent (dip-test FDR<0.05) in the celltype
    hm = sub_div.reindex(rk.head(25)['TF_full'])
    # Star = significant BETWEEN-SUBTYPE divergence by the permutation test
    # (falls back to the dip test only if the permutation result is missing).
    psig = perm_sig_set(grp)
    if psig is not None:
        sig_full = psig
        star_note = f"* = significant between-subtype divergence (permutation test, BH-FDR<{SIG_FDR})"
    else:
        sig_full = celltype_sig_set(grp)
        star_note = f"* = significantly bimodal (Hartigan's dip test, BH-FDR<{SIG_FDR})"
    ylabels = [short(t) + (' *' if t in sig_full else '') for t in hm.index]
    hm.index = ylabels
    fig, ax = plt.subplots(figsize=(max(6.5, len(subs)*1.05), 9))
    sns.heatmap(hm, cmap='magma_r', ax=ax, cbar_kws={'label': 'Divergence_STD'},
                annot=True, fmt='.0f', annot_kws={'fontsize': max(5, min(8, 90//len(subs)))},
                linewidths=0.4, linecolor='white', xticklabels=True, yticklabels=True)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=60, ha='right', fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=7)
    ax.set_title(f'{grp}: top variable TFs across subtypes (cell = Divergence_STD)\n{star_note}', fontsize=10)
    fig.tight_layout(); fig.savefig(f'{CSUB}/within_{grp}_TF_divergence_heatmap.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
within_df = pd.DataFrame(within_rows)
within_df.to_csv(f'{CSUB}/within_celltype_ranked_TF_divergence_ALL.csv', index=False)
reg('CrossSubtype/within_<GROUP>_ranked_TF_divergence.png',
    'CrossSubtype/within_celltype_ranked_TF_divergence_ALL.csv',
    'Per multi-subtype celltype: TFs ranked by between-subtype divergence variability')
reg('CrossSubtype/within_<GROUP>_TF_divergence_heatmap.png',
    'CrossSubtype/permutation_TFdivergence_ALL.csv',
    'Heatmap of Divergence_STD (annotated); * = permutation-test significant between-subtype divergence (BH-FDR<0.05)')

# ══════════════════════════════════════════════════════════════════
# UpSet — significant divergent TF signatures shared vs unique
# ══════════════════════════════════════════════════════════════════
print('UpSet: significant / co-zero TF sets')

def celltype_sig_sets():
    """celltype -> set of TF (short) STATISTICALLY significantly divergent:
    bimodal by Hartigan's dip test (BH-FDR<0.05) in a majority of its subtypes."""
    out = {}
    for grp, subs in GROUP_SUBTYPES.items():
        subs = [s for s in subs if s in present]
        if not subs: continue
        sset = celltype_sig_set(grp)
        if sset:
            out[grp] = {short(t) for t in sset}
    return out

def celltype_cozero_sets():
    """celltype -> set of TF (short) appearing in any frequent co-zero itemset."""
    out = {}
    for grp, subs in GROUP_SUBTYPES.items():
        tfs = set()
        for st in subs:
            f = f'{ZDIR}/{st}/{st}_cozero_itemsets.csv'
            if not os.path.exists(f): continue
            it = pd.read_csv(f)
            for s in it['TFs_short']:
                tfs.update(x.strip() for x in str(s).split('+'))
        if tfs: out[grp] = tfs
    return out

def node_aggregate(celltype_sets):
    out = {}
    for grp, s in celltype_sets.items():
        nn = GROUP_NODE[grp]; out.setdefault(nn, set()).update(s)
    return out

def membership_csv(sets, path, label):
    allt = sorted(set().union(*sets.values())) if sets else []
    df = pd.DataFrame({label: allt})
    for k, s in sets.items():
        df[k] = df[label].isin(s).astype(int)
    df['n_sets'] = df.drop(columns=[label]).sum(axis=1)
    df = df.sort_values('n_sets', ascending=False)
    df.to_csv(path, index=False)
    # uniques: present in exactly one set
    uniq = df[df['n_sets'] == 1].copy()
    rows = []
    for _, r in uniq.iterrows():
        for k in sets:
            if r[k] == 1:
                rows.append({label: r[label], 'unique_to': k})
    pd.DataFrame(rows).to_csv(path.replace('.csv', '_UNIQUE.csv'), index=False)
    return df

def upset_plot(sets, path, title, max_rank=30, label_thresh=14):
    """UpSet with: abbreviated category labels (no overlap on the totals bars),
    smaller matrix dots, and the actual TF names printed above each small
    intersection bar (size <= label_thresh)."""
    if len(sets) < 2:
        return
    # intersections (member-set -> TFs), sorted by size desc, top max_rank
    membership = {}
    for k, vals in sets.items():
        for el in vals:
            membership.setdefault(el, set()).add(k)
    groups = {}
    for el, mem in membership.items():
        groups.setdefault(frozenset(mem), []).append(el)
    inter = sorted(groups.items(), key=lambda kv: -len(kv[1]))[:max_rank]

    short_sets = {ab(k): list(v) for k, v in sets.items()}   # abbreviated category names
    data = from_contents(short_sets)
    fig = plt.figure(figsize=(max(13, len(sets)*0.95), 9))
    up = UpSet(data, sort_by='cardinality', show_counts=True, min_subset_size=1,
               max_subset_rank=max_rank, element_size=14, facecolor='#2c3e50')
    axd = up.plot(fig=fig)
    ax_int = axd['intersections']
    bars = sorted([p for p in ax_int.patches], key=lambda p: p.get_x())
    ymax = max((p.get_height() for p in bars), default=1)
    # bars are descending in size (sort_by cardinality) == order of `inter`.
    # NUMBER each small intersection bar and list its TFs in a compact key below —
    # avoids cramming many TF names onto narrow, closely-spaced bars.
    key_lines, num = [], 0
    for p, (mem, tfs) in zip(bars, inter):
        if abs(p.get_height() - len(tfs)) > 0.5:    # safety: skip if mapping mismatches
            continue
        if 0 < len(tfs) <= label_thresh:
            num += 1
            ax_int.text(p.get_x()+p.get_width()/2, p.get_height()+ymax*0.02, str(num),
                        ha='center', va='bottom', fontsize=7, fontweight='bold', color='#b00020')
            key_lines.append(f"{num}: " + ", ".join(sorted(tfs)))
    ax_int.set_ylim(0, ymax*1.12)
    fig.suptitle(title, fontsize=12, y=1.0)
    if key_lines:
        fig.text(0.01, -0.02, "Numbered intersections (TF members):\n" + "\n".join(key_lines),
                 ha='left', va='top', fontsize=6.5, family='monospace', color='#222')
    savefig_both(fig, path)
    plt.close(fig)

def write_intersections_csv(sets, path, label):
    """Enumerate every observed intersection -> the actual element (TF) names.
    UpSet shows only counts; this is the companion that names the TFs."""
    keys = list(sets)
    membership = {}
    for k in keys:
        for el in sets[k]:
            membership.setdefault(el, set()).add(k)
    groups = {}
    for el, mem in membership.items():
        groups.setdefault(frozenset(mem), []).append(el)
    rows = []
    for mem, els in groups.items():
        rows.append({'degree': len(mem),
                     'n_TFs': len(els),
                     label + 's': ' & '.join(sorted(mem)),
                     'TFs': ', '.join(sorted(els))})
    pd.DataFrame(rows).sort_values(['degree', 'n_TFs'], ascending=[True, False]).to_csv(path, index=False)

def plot_unique_tfs(sets, path, title):
    """Labeled figure naming the TFs UNIQUE to each set. ALL celltypes in `sets`
    are shown (even those with 0 unique TFs), so it is clear that a missing-from-
    the-old-plot celltype simply shared all its significant TFs with others."""
    keys = list(sets)
    membership = {}
    for k in keys:
        for el in sets[k]:
            membership.setdefault(el, set()).add(k)
    uniq = {k: sorted(el for el, mem in membership.items() if mem == {k}) for k in keys}
    order = sorted(keys, key=lambda k: len(uniq[k]))           # ascending; largest on top
    counts = [len(uniq[k]) for k in order]
    fig, ax = plt.subplots(figsize=(14, max(4, len(order)*0.5)))
    ax.barh(range(len(order)), counts,
            color=[NODE_COLOR.get(GROUP_NODE.get(k, ''), '#888') for k in order], edgecolor='white')
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=9)
    ax.set_xlabel('# TFs significant ONLY in this celltype (unique)', fontsize=10)
    ax.set_title(title + '\n("unique" = significant in this celltype and in NO other; '
                 '0 = all its significant TFs are shared)', fontsize=10)
    xmax = max(counts) if max(counts) > 0 else 1
    for i, k in enumerate(order):
        names = uniq[k]
        txt = ('none' if not names else
               ', '.join(names[:16]) + (f'  …(+{len(names)-16})' if len(names) > 16 else ''))
        ax.text(counts[i] + xmax*0.02, i, txt, va='center', fontsize=6.5,
                color='#333' if names else '#999')
    ax.set_xlim(0, xmax * 2.3 + 1)
    fig.tight_layout(); savefig_both(fig, path)
    plt.close(fig)

zc_index = []  # FIGURES_INDEX for TF_ZeroCooccur
def zreg(fig, table, desc): zc_index.append({'Figure': fig, 'Companion_Table': table, 'Description': desc})

# Significant divergent TFs = bimodal by Hartigan's dip test, BH-FDR<0.05 (majority of subtypes)
SIG_TITLE = "Statistically significant divergent TFs (Hartigan's dip test, BH-FDR<0.05)"
sig_ct = celltype_sig_sets()
membership_csv(sig_ct, f'{CSUB}/significant_TF_membership_by_celltype.csv', 'TF')
write_intersections_csv(sig_ct, f'{CSUB}/significant_TF_intersections_TFs.csv', 'celltype')
upset_plot(sig_ct, f'{CSUB}/upset_significant_divergent_TFs_by_celltype.png',
           SIG_TITLE + ' — shared vs unique across celltypes (top 30 intersections)')
upset_plot(node_aggregate(sig_ct), f'{CSUB}/upset_significant_divergent_TFs_by_node.png',
           SIG_TITLE + ' — shared vs unique across nodes')
plot_unique_tfs(sig_ct, f'{CSUB}/unique_significant_TFs_per_celltype.png',
                'TFs significantly divergent UNIQUELY in one celltype (dip test, BH-FDR<0.05)')
reg('CrossSubtype/upset_significant_divergent_TFs_by_celltype.png',
    'CrossSubtype/significant_TF_intersections_TFs.csv',
    "UpSet of dip-test-significant (BH-FDR<0.05) divergent TFs per celltype; intersections_TFs.csv names the TFs in each intersection")
reg('CrossSubtype/unique_significant_TFs_per_celltype.png',
    'CrossSubtype/significant_TF_membership_by_celltype_UNIQUE.csv',
    'TF names significantly divergent in exactly one celltype (dip test BH-FDR<0.05)')

# Permutation-based significance (BETWEEN-SUBTYPE) — restricted to the 11
# multi-subtype celltypes (single-subtype celltypes have no between-subtype test).
MULTI = [g for g, subs in GROUP_SUBTYPES.items() if len([s for s in subs if s in present]) >= 2]
perm_ct = {}
for grp in MULTI:
    ps = perm_sig_set(grp)            # None if no permutation file
    if ps:                           # also drops celltypes with 0 significant (CNGA, ITL45)
        perm_ct[grp] = {short(t) for t in ps}
if len(perm_ct) >= 2:
    PERM_TITLE = 'Between-subtype significant divergent TFs (permutation test, BH-FDR<0.05)'
    membership_csv(perm_ct, f'{CSUB}/perm_significant_TF_membership_by_celltype.csv', 'TF')
    write_intersections_csv(perm_ct, f'{CSUB}/perm_significant_TF_intersections_TFs.csv', 'celltype')
    upset_plot(perm_ct, f'{CSUB}/upset_perm_significant_TFs_by_celltype.png',
               PERM_TITLE + f' — shared vs unique across {len(perm_ct)} multi-subtype celltypes (top 30 intersections)')
    upset_plot(node_aggregate(perm_ct), f'{CSUB}/upset_perm_significant_TFs_by_node.png',
               PERM_TITLE + ' — shared vs unique across nodes')
    plot_unique_tfs(perm_ct, f'{CSUB}/unique_perm_significant_TFs_per_celltype.png',
                    'TFs with between-subtype-significant divergence UNIQUE to one celltype (permutation, BH-FDR<0.05)')
    reg('CrossSubtype/upset_perm_significant_TFs_by_celltype.png',
        'CrossSubtype/perm_significant_TF_intersections_TFs.csv',
        'UpSet of permutation-significant between-subtype divergent TFs across the multi-subtype celltypes; intersections_TFs.csv names the TFs')
    reg('CrossSubtype/unique_perm_significant_TFs_per_celltype.png',
        'CrossSubtype/perm_significant_TF_membership_by_celltype_UNIQUE.csv',
        'TFs with permutation-significant between-subtype divergence in exactly one celltype')
    print(f'Permutation UpSet over {len(perm_ct)} celltypes: {sorted(perm_ct)}')

# Co-zero ("missing"/unbound) TF sets — frequency-based (min-support), not a hypothesis test
coz_ct = celltype_cozero_sets()
if coz_ct:
    membership_csv(coz_ct, f'{ZCROSS}/cozero_TF_membership_by_celltype.csv', 'TF')
    write_intersections_csv(coz_ct, f'{ZCROSS}/cozero_TF_intersections_TFs.csv', 'celltype')
    upset_plot(coz_ct, f'{ZCROSS}/upset_cozero_TFs_by_celltype.png',
               'Frequently co-zero (jointly unbound) TFs — shared vs unique across celltypes (min-support frequent itemsets)')
    upset_plot(node_aggregate(coz_ct), f'{ZCROSS}/upset_cozero_TFs_by_node.png',
               'Frequently co-zero TFs — shared vs unique across nodes')
    plot_unique_tfs(coz_ct, f'{ZCROSS}/unique_cozero_TFs_per_celltype.png',
                    'Frequently-unbound TFs UNIQUE to one celltype (co-zero frequent itemsets)')
    zreg('Cross/upset_cozero_TFs_by_celltype.png', 'Cross/cozero_TF_intersections_TFs.csv',
         'UpSet of frequently co-zero TF sets per celltype; intersections_TFs.csv names the TFs')
    zreg('Cross/cozero_pair_prevalence.png', 'Cross/cozero_pair_prevalence.csv',
         'Co-zero TF pairs ranked by how many subtypes share them')

# ══════════════════════════════════════════════════════════════════
# FIGURES_INDEX — map every TF figure to a companion table
# ══════════════════════════════════════════════════════════════════
# per-subtype TF_Divergence figures (companion = divergence tables)
for st in present:
    base = f'TF_Divergence_{st}'
    reg(f'{base}/{st}_heatmap_top25.png', f'{base}/{st}_divergence_table_top50.csv',
        'Per-nucleus rank heatmap of top-25 divergent TFs')
    reg(f'{base}/{st}_violin_top20.png', f'{base}/{st}_divergence_table_top50.csv',
        'Within-subtype rank distribution, top-20 divergent TFs')
    reg(f'{base}/{st}_scatter.png', f'{base}/{st}_divergence_table_all.csv',
        'Divergence_STD vs Mean_Rank, all TFs')
# cross-celltype panels
for fig, tab, desc in [
    ('Cross_Celltype_Comparison/01_mean_divergence_per_subtype.png',
     'Cross_Celltype_Comparison/summary_divergence_per_subtype.csv', 'Mean top-50 Divergence_STD per subtype'),
    ('Cross_Celltype_Comparison/02_cross_celltype_heatmap.png',
     'Cross_Celltype_Comparison/cross_celltype_divergence_matrix.csv', 'Shared-TF divergence z-score heatmap'),
    ('Cross_Celltype_Comparison/03_divergence_distribution_kde.png',
     'Cross_Celltype_Comparison/summary_divergence_per_subtype.csv', 'Per-subtype Divergence_STD KDE'),
    ('Cross_Celltype_Comparison/04_divergence_boxplot_all_subtypes.png',
     'Cross_Celltype_Comparison/summary_divergence_per_subtype.csv', 'Per-subtype Divergence_STD boxplots'),
    ('Cross_Celltype_Comparison/05_convergent_divergent_TFs.png',
     'Cross_Celltype_Comparison/convergent_divergent_TFs.csv', 'TFs divergent across most subtypes'),
]:
    reg(fig, tab, desc)
pd.DataFrame(index_rows).to_csv(f'{DIVDIR}/FIGURES_INDEX.csv', index=False)

# TF_ZeroCooccur per-subtype companions
for st in present:
    if os.path.exists(f'{ZDIR}/{st}/{st}_cozero_itemsets.csv'):
        zreg(f'{st}/{st}_top_cozero_combinations.png', f'{st}/{st}_cozero_itemsets.csv',
             'Top co-zero TF modules by support')
        zreg(f'{st}/{st}_pairwise_cozero_heatmap.png', f'{st}/{st}_cozero_itemsets.csv',
             'Pairwise TF co-zero % among top informative TFs')
pd.DataFrame(zc_index).to_csv(f'{ZDIR}/FIGURES_INDEX.csv', index=False)

# ══════════════════════════════════════════════════════════════════
# STATISTICAL METHODS — which test backs which figure
# ══════════════════════════════════════════════════════════════════
methods = [
    ('Divergence_STD (per-subtype heatmap/violin/scatter, cross panels, within/global rankings, '
     'cross-subtype clustermap)',
     'DESCRIPTIVE statistic — NOT a hypothesis test. Divergence_STD = standard deviation of a TF\'s '
     'within-nucleus rank (average-rank ties) across nuclei. No p-value. At 10^3–10^5 nuclei a naive '
     'mean-difference test is over-powered, so effect size / ranking is used instead of a p-value.'),
    ('Bimodality flag, "* significant", and the significant-divergent-TF UpSet / unique-TF figures',
     "Hartigan's DIP TEST for unimodality on each TF's within-nucleus rank distribution "
     '(up to 1000 nuclei sampled), Benjamini–Hochberg FDR across the 879 TFs. '
     'Significant divergent (bimodal) = BH-FDR < 0.05. Celltype-level = significant in a majority of subtypes.'),
    ('Within-celltype heatmap "*" significance + permutation_TFdivergence CSVs',
     'PERMUTATION TEST (n-calibrated): between-subtype statistic T = SD across subtypes of each '
     "TF's Divergence_STD. Null built by shuffling nucleus->subtype labels (sizes preserved) 2000x; "
     'empirical p=(#perm>=obs+1)/2001, Benjamini-Hochberg FDR over 879 TFs. Significant=FDR<0.05. '
     'Unlike a parametric between-group test this is NOT inflated by large nuclei counts (null at same n). '
     'Run by 18_TF_permutation_test.py.'),
    ('Cross-subtype divergence distance (clustermap, PCA)',
     '1 − Spearman rank correlation between subtypes\' per-TF Divergence_STD profiles. '
     'Used as a DISTANCE for clustering, not as a hypothesis test.'),
    ('Co-zero TF modules and co-zero UpSet / unique figures',
     'FREQUENCY / FREQUENT-ITEMSET analysis (Apriori): a TF set is reported if jointly unbound (footprint%==0) '
     'in >= max(10 cells, 2% of nuclei). Support threshold, NOT a hypothesis test.'),
]
with open(f'{DIVDIR}/STATISTICAL_METHODS.txt', 'w') as f:
    f.write('STATISTICAL METHODS — TF footprint divergence / co-occurrence analyses\n')
    f.write('=====================================================================\n\n')
    for what, how in methods:
        f.write(f'• {what}\n    {how}\n\n')
print('Wrote', f'{DIVDIR}/STATISTICAL_METHODS.txt')

print(f'\nDone. New outputs in {CSUB}/ and {ZCROSS}/')
print(f'Indexes: {DIVDIR}/FIGURES_INDEX.csv  |  {ZDIR}/FIGURES_INDEX.csv')
