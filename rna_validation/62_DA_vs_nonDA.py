#!/usr/bin/env python3
"""
DA-vs-non-DA regulatory contrast (Track B). Finds regulatory regions that are
UNIQUELY PRESENT in the dopaminergic (DA) subtypes and those MISSING in DA but
present in the rest of CSPMEI. Two reference sets (both requested):
  - GABAergic non-DA  (fair within-inhibitory contrast)
  - ALL non-DA        (whole-atlas contrast: GABA + glutamatergic + non-neuronal)

Source: Joint_Differential/regulatory/<group>_joint_master.csv.gz (known+novel,
per-subtype Pct). Region key = region_id (basket|class|chr:start-end). Regions
absent from a group's master are treated as 0% (not accessible) — standard for
these presence-based cross-group panels.

Outputs -> DA_vs_nonDA/<contrast>/:
  DA_specific_regions.csv     present in >=50% DA subtypes, <=10% reference
  DA_absent_regions.csv       present in >=50% reference, <=10% DA subtypes
  DA_vs_nonDA_scatter.png     DA_mean vs ref_mean presence, hits highlighted
  DA_specific_heatmap.png     top DA-specific regions x subtypes (DA | ref)
  DA_absent_heatmap.png       top DA-absent  regions x subtypes (DA | ref)
  summary.txt
Writes NEW dirs only; nothing existing is overwritten. Env: zeros.
"""
import os, re
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.colors as mcolors
from scipy.stats import mannwhitneyu
import da_config as C

BASE = "/path/to/data"
DIFF = f"{BASE}/Joint_Differential/regulatory"
OUT = f"{BASE}/DA_vs_nonDA"
ALL_GROUPS = ["MSN", "Dopaminergic", "MDGA", "ICGA_1", "ICGA_2", "SEPGA", "SIGA", "CTXMIX",
              "FOXP2", "PVALB", "VIP", "CNGA", "BFEXA", "BNGA", "CNMIX", "PV_ChCs",
              "ITL23", "ITL34", "ITL4", "ITL45", "L6B", "CHO", "AMY",
              "ASCT", "MGC_1", "MGC_2"]
PRES = 20.0        # presence threshold (% cells) to call a region "active" in a subtype
FRAC_IN = 0.50     # present in >= this fraction of the in-group
FRAC_OUT = 0.10    # present in <= this fraction of the out-group
TOPN = 40          # regions per heatmap
CLASS_COL = {"DA": "#e6550d", "GABA_nonDA": "#3182bd", "GLUT": "#31a354", "NonN": "#756bb1"}
RT_COL = {"enhancer": "#3498db", "silencer": "#2ecc71", "promoter": "#f39c12",
          "dual": "#8e44ad", "other_regulatory": "#95a5a6", "insulator": "#1abc9c"}
BASK_COL = {"known": "#b0bec5", "novel": "#e74c3c"}


def log(m): print(m, flush=True)


def build_matrix():
    """Return (M: region_id x subtype Pct DataFrame, meta: per-region basket/class/coord)."""
    need = set(C.DA_SUBTYPES) | set(C.ALL_NONDA_SUBTYPES)
    series = {}
    for g in ALL_GROUPS:
        f = f"{DIFF}/{g}_joint_master.csv.gz"
        if not os.path.exists(f):
            log(f"  WARN missing master {g}"); continue
        d = pd.read_csv(f, usecols=lambda c: c == "region_id" or c.startswith("Pct_"),
                        dtype={"region_id": str}, low_memory=False)
        d = d.set_index("region_id")
        for col in d.columns:
            sub = col[4:]
            if sub in need:
                series[sub] = d[col].astype("float32")
        log(f"  loaded {g}: {d.shape[0]:,} regions, {d.shape[1]} subtypes")
    missing = need - set(series)
    if missing:
        log(f"  WARN subtypes with no data: {sorted(missing)}")
    M = pd.concat(series, axis=1).fillna(0.0)     # region_id x subtype, 0 where absent
    # trim uninformative rows (never active anywhere)
    M = M[M.max(axis=1) >= PRES]
    # region meta from region_id "basket|class|chr:start-end"
    parts = M.index.to_series().str.split("|", n=2, expand=True)
    meta = pd.DataFrame({"basket": parts[0].values, "region_class": parts[1].values,
                         "coord": parts[2].values}, index=M.index)
    log(f"  matrix: {M.shape[0]:,} informative regions x {M.shape[1]} subtypes")
    return M, meta


def contrast(M, meta, ref_subs, name):
    outdir = f"{OUT}/{name}"; os.makedirs(outdir, exist_ok=True)
    da_cols = [s for s in C.DA_SUBTYPES if s in M.columns]
    ref_cols = [s for s in ref_subs if s in M.columns]
    DA = M[da_cols].values; REF = M[ref_cols].values
    da_mean = DA.mean(1); ref_mean = REF.mean(1)
    da_frac = (DA >= PRES).mean(1); ref_frac = (REF >= PRES).mean(1)
    delta = da_mean - ref_mean
    base = pd.DataFrame({
        "region_id": M.index, "basket": meta["basket"].values,
        "region_class": meta["region_class"].values, "coord": meta["coord"].values,
        "DA_mean_pct": da_mean.round(2), "ref_mean_pct": ref_mean.round(2),
        "delta_pct": delta.round(2),
        "DA_frac_present": da_frac.round(3), "ref_frac_present": ref_frac.round(3),
    })
    spec_mask = (da_frac >= FRAC_IN) & (ref_frac <= FRAC_OUT)
    absent_mask = (ref_frac >= FRAC_IN) & (da_frac <= FRAC_OUT)

    def finish(mask, sort_desc, fname):
        idx = np.where(mask)[0]
        if len(idx) == 0:
            pd.DataFrame(columns=list(base.columns) + ["mwu_p", "mwu_FDR"]).to_csv(f"{outdir}/{fname}", index=False)
            return base.iloc[[]]
        rows = base.iloc[idx].copy()
        ps = []
        for i in idx:
            try:
                _, p = mannwhitneyu(DA[i], REF[i], alternative="two-sided")
            except ValueError:
                p = 1.0
            ps.append(p)
        ps = np.array(ps)
        order = np.argsort(ps); ranked = np.empty_like(order); ranked[order] = np.arange(1, len(ps) + 1)
        fdr = np.minimum.accumulate((ps[order] * len(ps) / np.arange(1, len(ps) + 1))[::-1])[::-1]
        fdr_full = np.empty_like(ps); fdr_full[order] = np.clip(fdr, 0, 1)
        rows["mwu_p"] = ps; rows["mwu_FDR"] = fdr_full.round(5)
        rows = rows.sort_values("delta_pct", ascending=not sort_desc)
        rows.to_csv(f"{outdir}/{fname}", index=False)
        return rows

    spec = finish(spec_mask, True, "DA_specific_regions.csv")
    absent = finish(absent_mask, False, "DA_absent_regions.csv")

    scatter(da_mean, ref_mean, spec_mask, absent_mask, name, outdir, len(ref_cols))
    if len(spec):
        heatmap(M, meta, spec.head(TOPN)["region_id"].tolist(), da_cols, ref_cols,
                f"DA-SPECIFIC regulatory regions — unique to dopaminergic (vs {name})",
                f"{outdir}/DA_specific_heatmap.png")
    if len(absent):
        heatmap(M, meta, absent.head(TOPN)["region_id"].tolist(), da_cols, ref_cols,
                f"DA-ABSENT regulatory regions — missing in dopaminergic (present in {name})",
                f"{outdir}/DA_absent_heatmap.png")
    with open(f"{outdir}/summary.txt", "w") as fh:
        fh.write(f"DA vs {name}\n"
                 f"DA subtypes ({len(da_cols)}): {da_cols}\n"
                 f"reference ({len(ref_cols)}): {ref_cols}\n"
                 f"thresholds: present>= {PRES}% ; in-frac>= {FRAC_IN} ; out-frac<= {FRAC_OUT}\n"
                 f"informative regions tested: {M.shape[0]}\n"
                 f"DA-specific (unique to DA): {int(spec_mask.sum())}\n"
                 f"DA-absent  (missing in DA): {int(absent_mask.sum())}\n")
        for lab, tb in [("DA-specific", spec), ("DA-absent", absent)]:
            if len(tb):
                fh.write(f"\ntop 10 {lab} (basket / class / coord / DA% / ref% / delta):\n")
                for _, r in tb.head(10).iterrows():
                    fh.write(f"  {r.basket:5s} {r.region_class:16s} {r.coord:28s} "
                             f"{r.DA_mean_pct:6.1f} {r.ref_mean_pct:6.1f} {r.delta_pct:+7.1f}\n")
    log(f"[{name}] DA-specific={int(spec_mask.sum())}  DA-absent={int(absent_mask.sum())}  -> {outdir}")
    return {"contrast": name, "DA_specific": int(spec_mask.sum()), "DA_absent": int(absent_mask.sum())}


def scatter(da_mean, ref_mean, spec_mask, absent_mask, name, outdir, n_ref):
    fig, ax = plt.subplots(figsize=(6.5, 6.2))
    other = ~(spec_mask | absent_mask)
    ax.scatter(ref_mean[other], da_mean[other], s=3, c="#cccccc", alpha=0.4, label="other", rasterized=True)
    ax.scatter(ref_mean[spec_mask], da_mean[spec_mask], s=10, c="#e6550d", label=f"DA-specific ({spec_mask.sum()})")
    ax.scatter(ref_mean[absent_mask], da_mean[absent_mask], s=10, c="#3182bd", label=f"DA-absent ({absent_mask.sum()})")
    lim = max(da_mean.max(), ref_mean.max()) * 1.02
    ax.plot([0, lim], [0, lim], "--", c="k", lw=0.7)
    ax.set_xlabel(f"mean presence in reference ({name}, n={n_ref}) [%]")
    ax.set_ylabel("mean presence in DA subtypes [%]")
    ax.set_title(f"DA vs {name}: per-region presence")
    ax.legend(fontsize=8, frameon=False); ax.set_xlim(-1, lim); ax.set_ylim(-1, lim)
    fig.tight_layout(); fig.savefig(f"{outdir}/DA_vs_nonDA_scatter.png", dpi=170); plt.close(fig)


def heatmap(M, meta, region_ids, da_cols, ref_cols, title, path):
    cols = da_cols + ref_cols
    H = M.loc[region_ids, cols].values
    rmeta = meta.loc[region_ids]
    labels = [f"[{'N' if b=='novel' else 'K'}] {c}" for b, c in zip(rmeta.basket, rmeta.coord)]
    n_r, n_c = H.shape
    n_da = len(da_cols)
    row_h, col_w = 0.19, 0.24
    left_in, strips_in, cbar_in, top_in, bottom_in = 3.0, 0.5, 1.4, 2.2, 2.0
    fig_w = left_in + n_c*col_w + strips_in + cbar_in
    fig_h = top_in + 0.5 + n_r*row_h + bottom_in
    fig = plt.figure(figsize=(fig_w, fig_h))
    L, R = left_in/fig_w, (left_in+n_c*col_w+strips_in)/fig_w
    T, Bo = 1-top_in/fig_h, bottom_in/fig_h
    gs = fig.add_gridspec(2, 3, width_ratios=[n_c*col_w, strips_in/2, strips_in/2],
                          height_ratios=[0.5, n_r*row_h], wspace=0.03, hspace=0.02,
                          left=L, right=R, top=T, bottom=Bo)
    ax = fig.add_subplot(gs[1, 0]); ax_top = fig.add_subplot(gs[0, 0])
    ax_rt = fig.add_subplot(gs[1, 1]); ax_bk = fig.add_subplot(gs[1, 2])
    ax_top.set_xlim(-0.5, n_c-0.5)
    vmax = np.percentile(H[H > 0], 98) if (H > 0).any() else 100
    im = ax.imshow(H, cmap="YlOrRd", aspect="auto", vmin=0, vmax=vmax)
    # top class strip
    strip = np.array([[mcolors.to_rgb(CLASS_COL[C.SUBTYPE_CLASS[c]]) for c in cols]])
    ax_top.imshow(strip, aspect="auto"); ax_top.set_yticks([]); ax_top.set_xticks([])
    # right strips
    ax_rt.imshow([[mcolors.to_rgb(RT_COL.get(t, "#ccc"))] for t in rmeta.region_class], aspect="auto")
    ax_bk.imshow([[mcolors.to_rgb(BASK_COL.get(b, "#ccc"))] for b in rmeta.basket], aspect="auto")
    for a in (ax_rt, ax_bk):
        a.set_xticks([]); a.set_yticks([])
    ax_rt.set_xlabel("type", fontsize=7); ax_bk.set_xlabel("K/N", fontsize=7)
    # region labels (left)
    ax.set_yticks(range(n_r)); ax.set_yticklabels(labels, fontsize=6)
    for tk, b in zip(ax.get_yticklabels(), rmeta.basket):
        tk.set_color(BASK_COL.get(b, "#000"))
    # subtype x labels
    ax.set_xticks(range(n_c)); ax.set_xticklabels(cols, rotation=90, fontsize=7)
    for tk, c in zip(ax.get_xticklabels(), cols):
        tk.set_color(CLASS_COL[C.SUBTYPE_CLASS[c]])
    ax.axvline(n_da-0.5, color="black", lw=2)          # DA | reference divider
    ax_top.axvline(n_da-0.5, color="black", lw=2)
    ax_top.text((n_da-1)/2, -0.9, "DA", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax_top.text((n_da+n_c-1)/2, -0.9, "reference", ha="center", va="bottom", fontsize=9, fontweight="bold")
    cax = fig.add_axes([R + 0.4/fig_w, Bo + 0.3*(T-Bo), 0.16/fig_w, 0.34*(T-Bo)])
    cb = fig.colorbar(im, cax=cax); cb.set_label("Presence (%)", fontsize=9)
    fig.suptitle(title + f"\n{n_r} regions | left [K]=known [N]=novel | x-labels coloured by class | black line = DA|reference",
                 fontsize=12, fontweight="bold", y=1 - 0.4/fig_h)
    classes = sorted({C.SUBTYPE_CLASS[c] for c in cols}, key=["DA", "GABA_nonDA", "GLUT", "NonN"].index)
    handles = [Patch(color=CLASS_COL[k], label=k) for k in classes] + \
              [Patch(color=BASK_COL[b], label=f"{b} region") for b in ("known", "novel")] + \
              [Patch(color=RT_COL[t], label=t) for t in sorted(set(rmeta.region_class)) if t in RT_COL]
    fig.legend(handles=handles, loc="lower center", ncol=min(8, len(handles)), fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, 0.4/fig_h))
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.3); plt.close(fig)
    log(f"  wrote {path}")


def main():
    os.makedirs(OUT, exist_ok=True)
    M, meta = build_matrix()
    M.to_parquet(f"{OUT}/presence_matrix_DA_plus_nonDA.parquet") if False else None
    results = []
    results.append(contrast(M, meta, C.GABA_NONDA_SUBTYPES, "GABAergic_nonDA"))
    results.append(contrast(M, meta, C.ALL_NONDA_SUBTYPES, "ALL_nonDA"))
    pd.DataFrame(results).to_csv(f"{OUT}/contrast_counts.csv", index=False)
    log("DA_vs_nonDA COMPLETE")


if __name__ == "__main__":
    main()
