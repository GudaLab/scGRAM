#!/usr/bin/env python3
"""
12_eval_reg_vs_nonreg.py
========================
Re-bucket the 4-head multi-label predictions into reg vs. non-reg per the
reviewer's second comment:

  "Predict the labels in the data (Enh, Prom, Genic), then treat them as
   reg / non-reg as per our need."

For a peak to be regulatory we OR over the regulatory-flavor heads:

  is_reg  =  is_enhancer  ∨  is_promoter  ∨  is_silencer
            (i.e. annotated as a cis-regulatory element of any flavor)

We exclude is_genic from the OR because gene-body annotations are not
regulatory in the cis-regulatory-element sense, but you can flip the
--include-genic flag if your downstream needs the wider definition.

Combined probability (probabilistic OR under independence):

  P(is_reg) = 1 − ∏(1 − P(label_i))   over the regulatory labels

Outputs (under model_output_v2/<sub>/):
  reg_eval.json                 AUROC, AUPRC, F1 etc.
  figures/18_reg_vs_nonreg.png  bar plot per model
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (average_precision_score, f1_score, roc_auc_score,
                             precision_score, recall_score)

BASE = "/path/to/scgram"
FIG_DIR = os.path.join(BASE, "figures")
LABELS_FULL = ["is_enhancer", "is_promoter", "is_genic", "is_silencer"]
DEFAULT_REG = {"is_enhancer", "is_promoter", "is_silencer"}


def find_prediction_files():
    """Locate test_predictions.npz for every trained model variant."""
    paths = []
    for root in (
        os.path.join(BASE, "model_output_v2"),
        os.path.join(BASE, "model_output_v2_hybrid_v2"),
        os.path.join(BASE, "model_output"),
        os.path.join(BASE, "benchmarks", "results"),
    ):
        for p in sorted(glob.glob(os.path.join(root, "*", "test_predictions.npz"))):
            paths.append(p)
    return paths


def combine_reg(probs, labels, reg_idxs):
    """Take (n, k) probs/labels → (n,) reg-positive prob + (n,) reg truth."""
    reg_probs = 1.0 - np.prod(1.0 - probs[:, reg_idxs], axis=1)
    reg_truth = (labels[:, reg_idxs].sum(axis=1) > 0).astype(np.uint8)
    return reg_probs, reg_truth


def evaluate(probs, labels, reg_set):
    """probs (n,k), labels (n,k) → reg-vs-nonreg metrics dict."""
    # Determine which columns count as "reg" given the available label list
    # (some benchmarks predict only 2 heads; some 4)
    n_cols = probs.shape[1]
    cols_available = LABELS_FULL[:n_cols] if n_cols <= 4 \
                     else LABELS_FULL  # safe fallback
    reg_idxs = [i for i, name in enumerate(cols_available) if name in reg_set]
    if not reg_idxs:
        return None
    rp, rt = combine_reg(probs, labels, reg_idxs)
    if rt.sum() == 0 or (1 - rt).sum() == 0:
        return None
    rpred = (rp >= 0.5).astype(np.uint8)
    return {
        "labels_combined": [cols_available[i] for i in reg_idxs],
        "n_test": int(len(rt)),
        "n_reg_positive": int(rt.sum()),
        "n_reg_negative": int((1 - rt).sum()),
        "auroc": float(roc_auc_score(rt, rp)),
        "auprc": float(average_precision_score(rt, rp)),
        "f1_at_0p5": float(f1_score(rt, rpred, zero_division=0)),
        "precision_at_0p5": float(precision_score(rt, rpred, zero_division=0)),
        "recall_at_0p5": float(recall_score(rt, rpred, zero_division=0)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-genic", action="store_true",
                    help="treat is_genic as regulatory too (wider def.)")
    args = ap.parse_args()
    reg_set = set(DEFAULT_REG)
    if args.include_genic:
        reg_set.add("is_genic")
    print(f"Combining as reg: {sorted(reg_set)}")

    rows = []
    for p in find_prediction_files():
        try:
            z = np.load(p)
            # 04_train uses "probabilities"; benchmark scripts use "probs"
            probs_key = "probabilities" if "probabilities" in z.files else "probs"
            probs = z[probs_key].astype(np.float32)
            labs = z["labels"].astype(np.uint8)
        except Exception as e:
            print(f"  skip {p}: {e}"); continue
        if probs.ndim != 2 or labs.ndim != 2:
            continue
        m = evaluate(probs, labs, reg_set)
        if m is None:
            continue
        # Model name = parent dir relative to the search roots
        parts = p.split(os.sep)
        name = f"{parts[-3]}/{parts[-2]}" if len(parts) >= 3 else parts[-2]
        m["model"] = name
        rows.append(m)
        print(f"  {name:40s}  AUROC={m['auroc']:.4f}  AUPRC={m['auprc']:.4f}  "
              f"F1@0.5={m['f1_at_0p5']:.4f}")

    if not rows:
        print("No 4-head predictions found yet — run v2 retrain first.")
        return

    out_json = os.path.join(BASE, "reg_vs_nonreg_summary.json")
    json.dump(rows, open(out_json, "w"), indent=2)
    print(f"\nWrote {out_json}")

    # Bar plot — AUROC, AUPRC, F1 per model
    os.makedirs(FIG_DIR, exist_ok=True)
    rows.sort(key=lambda r: r["auroc"], reverse=True)
    names = [r["model"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(15, max(4, 0.35 * len(rows))),
                             sharey=True)
    for ax, key, title in zip(axes, ["auroc", "auprc", "f1_at_0p5"],
                              ["Reg AUROC", "Reg AUPRC", "Reg F1@0.5"]):
        vals = [r[key] for r in rows]
        ax.barh(range(len(rows)), vals, color="#2e7d32",
                edgecolor="black", linewidth=0.5)
        ax.set_yticks(range(len(rows))); ax.set_yticklabels(names, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.0); ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(alpha=0.3, axis="x")
        for i, v in enumerate(vals):
            ax.text(min(v + 0.01, 0.98), i, f"{v:.3f}",
                    va="center", fontsize=9)
    fig.suptitle(f"Regulatory-vs-Non-regulatory  "
                 f"(combined: {sorted(reg_set)})",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG_DIR, f"18_reg_vs_nonreg.{ext}"),
                    dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {FIG_DIR}/18_reg_vs_nonreg.png")


if __name__ == "__main__":
    main()
