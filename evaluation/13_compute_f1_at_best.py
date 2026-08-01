#!/usr/bin/env python3
"""
13_compute_f1_at_best.py
========================
For every model's test_predictions.npz, sweep thresholds per label and write
a small summary JSON next to it (`f1_at_best.json`). aggregate.py then reads
those so the comparison table can emit F1@0.5 and F1@best side-by-side
(addressing the label-smoothing F1@0.5 artifact).

Reports:
  per-label best threshold + F1@best
  Macro F1@best (4-way, raw)
  Macro F1@best (3-way, excluding is_genic which is dominated by a
                 prevalence artifact: at low thresholds the trivial
                 "call all positive" gives F1 ≈ 0.94 at 89% prevalence)
  Macro F1@best (like-for-like: enh + sil only — matches v1 head set)
"""

import glob
import json
import os
import sys

import numpy as np
from sklearn.metrics import f1_score

BASE = "/path/to/scgram"
LABELS = ["is_enhancer", "is_promoter", "is_genic", "is_silencer"]
SHARED_WITH_V1 = ["is_enhancer", "is_silencer"]
THRESHOLDS = np.linspace(0.05, 0.95, 19)


def sweep_one(z, label_list):
    p = z["probabilities" if "probabilities" in z.files else "probs"]
    y = z["labels"].astype(np.uint8)
    out = {}
    for c, name in enumerate(label_list):
        if c >= y.shape[1]:
            continue
        y1 = y[:, c]; p1 = p[:, c]
        if y1.sum() == 0 or (1 - y1).sum() == 0:
            continue
        f1_05 = f1_score(y1, (p1 >= 0.5).astype(np.uint8), zero_division=0)
        f1s = [f1_score(y1, (p1 >= t).astype(np.uint8), zero_division=0)
               for t in THRESHOLDS]
        idx = int(np.argmax(f1s))
        out[name] = {"f1_at_0p5": float(f1_05),
                     "best_threshold": float(THRESHOLDS[idx]),
                     "f1_at_best": float(f1s[idx])}
    return out


def main():
    paths = []
    for root in (os.path.join(BASE, "model_output_v2"),
                 os.path.join(BASE, "model_output_v2_hybrid_v2"),
                 os.path.join(BASE, "model_output"),
                 os.path.join(BASE, "benchmarks", "results")):
        paths.extend(sorted(glob.glob(os.path.join(root, "*", "test_predictions.npz"))))
    for p in paths:
        try:
            z = np.load(p)
        except Exception as e:
            print(f"  skip {p}: {e}"); continue
        labels_present = LABELS if z["labels"].shape[1] >= 4 else SHARED_WITH_V1
        per = sweep_one(z, labels_present)
        # Macro F1@best (all available labels)
        f1b = [per[n]["f1_at_best"] for n in per]
        f1_05 = [per[n]["f1_at_0p5"] for n in per]
        summary = {
            "n_labels_swept": len(per),
            "per_label": per,
            "macro_f1_at_0p5_all": float(np.mean(f1_05)) if f1_05 else None,
            "macro_f1_at_best_all": float(np.mean(f1b)) if f1b else None,
        }
        # Macro F1@best excluding is_genic (high-prevalence artifact)
        no_genic = [per[n]["f1_at_best"] for n in per if n != "is_genic"]
        if no_genic:
            summary["macro_f1_at_best_excl_genic"] = float(np.mean(no_genic))
        # Like-for-like macro F1 (shared labels with v1)
        shared = [per[n]["f1_at_best"] for n in per if n in SHARED_WITH_V1]
        if len(shared) == 2:
            summary["macro_f1_at_best_shared"] = float(np.mean(shared))
        out_path = os.path.join(os.path.dirname(p), "f1_at_best.json")
        json.dump(summary, open(out_path, "w"), indent=2)
        tag = os.path.relpath(os.path.dirname(p), BASE)
        print(f"  {tag:50s}  F1@0.5_all={summary['macro_f1_at_0p5_all']:.4f}  "
              f"F1@best_all={summary['macro_f1_at_best_all']:.4f}  "
              f"F1@best_excl_genic={summary.get('macro_f1_at_best_excl_genic', float('nan')):.4f}")


if __name__ == "__main__":
    main()
