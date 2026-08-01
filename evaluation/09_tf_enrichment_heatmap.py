#!/usr/bin/env python3
"""
09_tf_enrichment_heatmap.py
===========================
TF enrichment per regulatory category. For each TF and each category
(enhancer-only / silencer-only / dual / neither) computes the log2 fold
enrichment of that TF's presence relative to its genome-wide frequency:

    enrichment[tf, cat] = log2( P(TF present | cat) / P(TF present overall) )

Categories are derived from the ground-truth multi-label tensors
(lab2_*.npy: is_enhancer, is_silencer). TF presence comes from the
bit-packed TF features (tfb_*.npy) unpacked with the tf_vocabulary.

Output: figures/15_tf_enrichment_heatmap.png / .pdf

Usage:
    python 09_tf_enrichment_heatmap.py [--chunks 200] [--top 30]
"""

import argparse
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/path/to/scgram"
MMAP = os.path.join(BASE, "training_data", "mmap")
FIG_DIR = os.path.join(BASE, "figures")
CATS = ["enhancer", "silencer", "enhancer_silencer", "neither"]


def category_of(is_enh, is_sil):
    """Vectorized: (n,) is_enh, is_sil uint8 → category index 0..3."""
    cat = np.full(len(is_enh), 3, dtype=np.int8)            # neither
    cat[(is_enh == 1) & (is_sil == 0)] = 0                  # enhancer
    cat[(is_enh == 0) & (is_sil == 1)] = 1                  # silencer
    cat[(is_enh == 1) & (is_sil == 1)] = 2                  # dual
    return cat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=int, default=200,
                    help="number of chunks to sample (50k peaks each)")
    ap.add_argument("--top", type=int, default=30,
                    help="number of top TFs to display")
    args = ap.parse_args()

    os.makedirs(FIG_DIR, exist_ok=True)
    with open(os.path.join(BASE, "training_data", "tf_vocabulary.json")) as f:
        tf_list = json.load(f)["tf_list"]
    n_tfs = len(tf_list)

    info = json.load(open(os.path.join(MMAP, "dataset_info_mmap.json")))
    n_chunks = info["n_chunks"]
    packed_dim = info.get("tf_bits_packed_dim", (n_tfs + 7) // 8)

    rng = np.random.default_rng(0)
    chunk_ids = sorted(rng.choice(n_chunks, min(args.chunks, n_chunks), replace=False))

    # accumulators
    cat_tf = np.zeros((4, n_tfs), dtype=np.int64)   # TF-present count per category
    cat_tot = np.zeros(4, dtype=np.int64)           # peaks per category

    used = 0
    for ci in chunk_ids:
        lab_p = os.path.join(MMAP, f"lab2_{ci:04d}.npy")
        tfb_p = os.path.join(MMAP, f"tfb_{ci:04d}.npy")
        if not (os.path.isfile(lab_p) and os.path.isfile(tfb_p)):
            continue
        lab = np.load(lab_p)                        # (n, 2) uint8
        tfb = np.load(tfb_p)                        # (n, packed_dim) uint8
        tf = np.unpackbits(tfb, axis=1)[:, :n_tfs]  # (n, n_tfs) 0/1
        cat = category_of(lab[:, 0], lab[:, 1])
        for c in range(4):
            mask = cat == c
            if mask.any():
                cat_tot[c] += int(mask.sum())
                cat_tf[c] += tf[mask].sum(axis=0).astype(np.int64)
        used += 1

    total = cat_tot.sum()
    overall_freq = cat_tf.sum(axis=0) / max(total, 1)        # (n_tfs,)
    # per-category TF frequency
    cat_freq = cat_tf / np.maximum(cat_tot[:, None], 1)      # (4, n_tfs)
    # log2 fold enrichment vs overall, with small pseudocount
    eps = 1e-6
    log2fc = np.log2((cat_freq + eps) / (overall_freq[None, :] + eps))  # (4, n_tfs)

    # pick top TFs by max |log2FC| across categories, restricted to TFs with
    # non-trivial overall frequency (avoid ultra-rare noise)
    keep = overall_freq > 0.001
    score = np.where(keep, np.abs(log2fc).max(axis=0), -1)
    top_idx = np.argsort(score)[::-1][:args.top]
    top_idx = top_idx[score[top_idx] > 0]

    M = log2fc[:, top_idx].T   # (top, 4)
    names = [tf_list[i].split("_")[0] for i in top_idx]  # strip motif ID

    print(f"Sampled {used} chunks, {total:,} peaks. "
          f"Category totals: {dict(zip(CATS, cat_tot.tolist()))}")

    fig, ax = plt.subplots(figsize=(8.5, max(8, 0.42 * len(names))))
    vmax = np.nanpercentile(np.abs(M), 98)
    im = ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["enhancer", "silencer", "dual", "neither"],
                       fontsize=12, rotation=20, ha="right")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_title(f"TF enrichment per category  (log2 fold vs genome-wide)\n"
                 f"top {len(names)} TFs by max |enrichment|, {total:,} peaks",
                 fontsize=13, fontweight="bold")
    # annotate cells
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center",
                    fontsize=7, color="black" if abs(M[i, j]) < vmax * 0.6 else "white")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("log2 fold enrichment", fontsize=11)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG_DIR, f"15_tf_enrichment_heatmap.{ext}"),
                    dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {FIG_DIR}/15_tf_enrichment_heatmap.png and .pdf")


if __name__ == "__main__":
    main()
