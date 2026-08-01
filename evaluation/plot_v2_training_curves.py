#!/usr/bin/env python3
"""Regenerate training-curve figures from the v2 training_results.json files.
Produces:
  figures/01_training_curves_cnn_v2.{png,pdf}
  figures/01_training_curves_hybrid_v2.{png,pdf}         (original v2 Hybrid)
  figures/01_training_curves_hybrid_planA.{png,pdf}      (Plan A retrain, LR warmup + grad clip)
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/path/to/scgram"
FIGDIR = os.path.join(BASE, "figures")

RUNS = [
    ("cnn_v2",         "v2 / CNN",
     f"{BASE}/model_output_v2/cnn/training_results.json"),
    ("hybrid_v2",      "v2 / Hybrid (original)",
     f"{BASE}/model_output_v2/hybrid/training_results.json"),
    ("hybrid_planA",   "v2 / Hybrid — Plan A (LR warmup + grad clip)",
     f"{BASE}/model_output_v2_hybrid_v2/hybrid/training_results.json"),
]

for slug, title, path in RUNS:
    d = json.load(open(path))
    h = d["history"]
    epochs = list(range(1, len(h["train_loss"]) + 1))
    best = d["best_epoch"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # panel 1: loss
    ax = axes[0]
    ax.plot(epochs, h["train_loss"], marker="o", label="train_loss", color="#1f77b4")
    ax.plot(epochs, h["val_loss"],   marker="s", label="val_loss",   color="#d62728")
    ax.axvline(best, color="grey", linestyle="--", alpha=0.7,
               label=f"best epoch = {best}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Loss")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    # panel 2: val macro metrics
    ax2 = axes[1]
    ax2.plot(epochs, h["val_macro_auroc"], marker="o", label="val_macro_AUROC", color="#2ca02c")
    ax2.plot(epochs, h["val_macro_auprc"], marker="s", label="val_macro_AUPRC", color="#ff7f0e")
    ax2.plot(epochs, h["val_macro_f1"],    marker="^", label="val_macro_F1",    color="#9467bd")
    ax2.axvline(best, color="grey", linestyle="--", alpha=0.7)
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("value")
    ax2.set_title("Validation metrics")
    ax2.legend(loc="best", fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Annotate headline test AUROC
    test_line = (f"best_val_macro_AUROC = {d['best_val_macro_auroc']:.4f}   "
                 f"test_macro_AUROC = {d['test_macro_auroc']:.4f}   "
                 f"test_macro_AUPRC = {d['test_macro_auprc']:.4f}   "
                 f"test_macro_F1 = {d['test_macro_f1']:.4f}")
    fig.suptitle(f"{title}\n{test_line}", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    for ext in ("png", "pdf"):
        out = os.path.join(FIGDIR, f"01_training_curves_{slug}.{ext}")
        plt.savefig(out, dpi=150 if ext == "png" else None, bbox_inches="tight")
        print(f"wrote {out}")
    plt.close(fig)

print("done")
