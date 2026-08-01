#!/usr/bin/env python3
"""
11_aggregate_loco.py
====================
Aggregate the 10-fold chromosome-rotation CV results: collect each fold's
held-out test metrics, compute mean ± std (per-chromosome-group error bars),
write a summary JSON, and plot AUROC/AUPRC/F1 across folds with error bars.

Reads:  model_output_loco/fold_<k>/cnn/training_results.json
Writes: model_output_loco/loco_cv_summary.json
        figures/16_loco_cv_auroc.png / .pdf
"""

import json
import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/path/to/scgram"
LOCO = os.path.join(BASE, "model_output_loco")
FIG_DIR = os.path.join(BASE, "figures")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    folds = sorted(glob.glob(os.path.join(LOCO, "fold_*")))
    rows = []
    for fd in folds:
        rp = os.path.join(fd, "cnn", "training_results.json")
        if not os.path.isfile(rp):
            continue
        r = json.load(open(rp))
        pl = r.get("test_per_label", {})
        rows.append({
            "fold": os.path.basename(fd),
            "macro_auroc": r.get("test_macro_auroc", float("nan")),
            "macro_auprc": r.get("test_macro_auprc", float("nan")),
            "macro_f1": r.get("test_macro_f1", float("nan")),
            "enh_auroc": pl.get("is_enhancer", {}).get("auroc", float("nan")),
            "sil_auroc": pl.get("is_silencer", {}).get("auroc", float("nan")),
        })
    if not rows:
        print("No completed folds found.")
        return

    def ms(key):
        v = np.array([r[key] for r in rows], dtype=float)
        v = v[~np.isnan(v)]
        return float(v.mean()), float(v.std(ddof=1) if len(v) > 1 else 0.0), v

    summary = {"n_folds_completed": len(rows), "per_fold": rows, "aggregate": {}}
    for key in ["macro_auroc", "macro_auprc", "macro_f1", "enh_auroc", "sil_auroc"]:
        m, s, _ = ms(key)
        summary["aggregate"][key] = {"mean": m, "std": s}

    out = os.path.join(LOCO, "loco_cv_summary.json")
    json.dump(summary, open(out, "w"), indent=2)
    print(f"Wrote {out}")
    print(f"\n{len(rows)}-fold LOCO-CV (mean ± std):")
    for key in ["macro_auroc", "enh_auroc", "sil_auroc", "macro_auprc", "macro_f1"]:
        m, s, _ = ms(key)
        print(f"  {key:<12} {m:.4f} ± {s:.4f}")

    # Plot: per-metric mean ± std bar with per-fold scatter
    metrics = [("macro_auroc", "Macro AUROC"), ("enh_auroc", "Enhancer AUROC"),
               ("sil_auroc", "Silencer AUROC"), ("macro_auprc", "Macro AUPRC"),
               ("macro_f1", "Macro F1@0.5")]
    fig, ax = plt.subplots(figsize=(10, 6))
    xs = np.arange(len(metrics))
    means = [ms(k)[0] for k, _ in metrics]
    stds = [ms(k)[1] for k, _ in metrics]
    bars = ax.bar(xs, means, yerr=stds, capsize=8, color="#2e7d32",
                  alpha=0.75, edgecolor="#1b5e20", linewidth=1.5, zorder=2)
    # overlay per-fold points
    for i, (k, _) in enumerate(metrics):
        _, _, v = ms(k)
        ax.scatter(np.full_like(v, i) + np.random.uniform(-0.12, 0.12, len(v)),
                   v, color="#0f172a", s=22, zorder=3, alpha=0.7)
    for b, m, s in zip(bars, means, stds):
        ax.text(b.get_x() + b.get_width() / 2, m + s + 0.012,
                f"{m:.3f}\n±{s:.3f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(xs); ax.set_xticklabels([t for _, t in metrics], fontsize=11)
    ax.set_ylim(0, 1.0); ax.set_ylabel("Score", fontsize=12)
    ax.set_title(f"{len(rows)}-fold chromosome-rotation cross-validation\n"
                 f"(held-out chromosome groups; points = per-fold values)",
                 fontsize=14, fontweight="bold")
    ax.grid(alpha=0.3, axis="y")
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG_DIR, f"16_loco_cv_auroc.{ext}"),
                    dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {FIG_DIR}/16_loco_cv_auroc.png and .pdf")


if __name__ == "__main__":
    main()
