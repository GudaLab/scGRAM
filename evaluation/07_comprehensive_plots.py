#!/usr/bin/env python3
"""
07_comprehensive_plots.py
=========================
Comprehensive, multilabel-aware visualization suite for the enhancer/silencer
characterization study. Reads only existing pipeline outputs (no GPU, no
re-inference):

  model_output/<model>/training_results.json   (history + test metrics)
  model_output/<model>/test_predictions.npz     (labels, probabilities)
  model_output/model_comparison.json
  predictions/prediction_summary.json
  predictions/<sample>/<celltype>/*_predictions.csv

Generates (into figures/):
  Training / model
    01_training_curves_<model>.png     loss + AUROC + AUPRC + F1 vs epoch
    02_model_comparison.png            CNN vs Hybrid vs Ensemble test metrics
  Test-set evaluation (per label, models overlaid)
    03_roc_curves.png
    04_pr_curves.png
    05_calibration.png                 reliability diagrams
    06_prob_histograms.png             P(label) split by true class
    07_threshold_sweep.png             F1/precision/recall vs threshold
    08_joint_prob_hexbin.png           P(enh) vs P(sil) density (test set)
    09_per_label_confusion.png         2x2 confusion @ 0.5 per label
  Uncharacterized predictions
    10_category_bar.png                global category counts
    11_category_donut.png
    12_per_sample_stacked.png          category composition per sample group
    13_ntfs_by_category.png            n_tfs distribution per category
    14_uncharacterized_joint.png       P(enh) vs P(sil) on uncharacterized

Usage:
    python 07_comprehensive_plots.py [--eval-model ensemble]
                                     [--subsample 500000]
                                     [--pred-sample-cells 3000]
"""

import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    f1_score, precision_score, recall_score, confusion_matrix,
)

BASE_DIR = "/path/to/scgram"
MODEL_DIR = os.path.join(BASE_DIR, "model_output")
PRED_DIR = os.path.join(BASE_DIR, "predictions")
FIG_DIR = os.path.join(BASE_DIR, "figures")
LABELS = ["is_enhancer", "is_silencer"]
MODELS = ["cnn", "hybrid", "ensemble"]
CAT_ORDER = ["enhancer", "silencer", "enhancer_silencer", "neither"]
CAT_COLORS = {"enhancer": "#2ca02c", "silencer": "#d62728",
              "enhancer_silencer": "#9467bd", "neither": "#7f7f7f"}


def _save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG_DIR, f"{name}.{ext}"),
                    dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}")


def _safe(name, fn, *a, **k):
    try:
        fn(*a, **k)
    except Exception as e:
        print(f"  [skip] {name}: {e}")


# ---------- Training / model ----------

def plot_training_curves(model):
    rp = os.path.join(MODEL_DIR, model, "training_results.json")
    if not os.path.isfile(rp):
        return
    h = json.load(open(rp)).get("history", {})
    if not h.get("train_loss"):
        return
    epochs = range(1, len(h["train_loss"]) + 1)
    panels = [("train_loss", "Train Loss"), ("val_loss", "Val Loss"),
              ("val_macro_auroc", "Val macro AUROC"),
              ("val_macro_auprc", "Val macro AUPRC"),
              ("val_macro_f1", "Val macro F1@0.5")]
    avail = [(k, t) for k, t in panels if k in h and h[k]]
    fig, axes = plt.subplots(1, len(avail), figsize=(4.2 * len(avail), 4))
    if len(avail) == 1:
        axes = [axes]
    for ax, (k, t) in zip(axes, avail):
        ax.plot(epochs, h[k], marker="o", ms=3)
        ax.set_xlabel("Epoch"); ax.set_title(t); ax.grid(alpha=0.3)
        if "auroc" in k or "auprc" in k or "f1" in k:
            best = max(h[k]); be = int(np.argmax(h[k])) + 1
            ax.axvline(be, color="gray", ls="--", alpha=0.5)
            ax.set_title(f"{t}\n(best {best:.4f} @ ep {be})")
    fig.suptitle(f"{model.upper()} training curves", y=1.02, fontsize=15)
    _save(fig, f"01_training_curves_{model}")


def plot_model_comparison():
    cmp_path = os.path.join(MODEL_DIR, "model_comparison.json")
    rows = {}
    for m in MODELS:
        rp = os.path.join(MODEL_DIR, m, "training_results.json")
        if os.path.isfile(rp):
            r = json.load(open(rp))
            rows[m] = {
                "AUROC": r.get("test_macro_auroc", 0),
                "AUPRC": r.get("test_macro_auprc", 0),
                "F1@0.5": r.get("test_macro_f1", 0),
            }
    if not rows:
        return
    metrics = ["AUROC", "AUPRC", "F1@0.5"]
    x = np.arange(len(metrics)); w = 0.8 / len(rows)
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (m, vals) in enumerate(rows.items()):
        bars = ax.bar(x + i * w, [vals[k] for k in metrics], w, label=m)
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                    f"{b.get_height():.3f}", ha="center", fontsize=8)
    ax.set_xticks(x + w * (len(rows) - 1) / 2); ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1); ax.set_ylabel("Test macro score")
    ax.set_title("Model comparison (test set)"); ax.legend(); ax.grid(alpha=0.3, axis="y")
    _save(fig, "02_model_comparison")


# ---------- Test-set evaluation ----------

def _load_test(model):
    p = os.path.join(MODEL_DIR, model, "test_predictions.npz")
    if not os.path.isfile(p):
        return None
    d = np.load(p)
    return d["labels"], d["probabilities"]


def _subsample(labels, probs, n):
    if len(labels) <= n:
        return labels, probs
    idx = np.random.default_rng(0).choice(len(labels), n, replace=False)
    return labels[idx], probs[idx]


def plot_roc(eval_models, subN):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for c, lab in enumerate(LABELS):
        for m in eval_models:
            t = _load_test(m)
            if t is None:
                continue
            y, p = _subsample(t[0], t[1], subN)
            fpr, tpr, _ = roc_curve(y[:, c], p[:, c])
            axes[c].plot(fpr, tpr, label=f"{m} (AUC={auc(fpr,tpr):.3f})")
        axes[c].plot([0, 1], [0, 1], "k--", alpha=0.4)
        axes[c].set_title(f"ROC — {lab}"); axes[c].set_xlabel("FPR")
        axes[c].set_ylabel("TPR"); axes[c].legend(); axes[c].grid(alpha=0.3)
    _save(fig, "03_roc_curves")


def plot_pr(eval_models, subN):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for c, lab in enumerate(LABELS):
        for m in eval_models:
            t = _load_test(m)
            if t is None:
                continue
            y, p = _subsample(t[0], t[1], subN)
            pr, rc, _ = precision_recall_curve(y[:, c], p[:, c])
            ap = average_precision_score(y[:, c], p[:, c])
            axes[c].plot(rc, pr, label=f"{m} (AP={ap:.3f})")
        prev = (_load_test(eval_models[0])[0][:, c].mean())
        axes[c].axhline(prev, color="gray", ls="--", alpha=0.5,
                        label=f"prevalence={prev:.3f}")
        axes[c].set_title(f"PR — {lab}"); axes[c].set_xlabel("Recall")
        axes[c].set_ylabel("Precision"); axes[c].legend(); axes[c].grid(alpha=0.3)
    _save(fig, "04_pr_curves")


def plot_calibration(model, subN):
    t = _load_test(model)
    if t is None:
        return
    y, p = _subsample(t[0], t[1], subN)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for c, lab in enumerate(LABELS):
        bins = np.linspace(0, 1, 11)
        idx = np.digitize(p[:, c], bins) - 1
        xs, ys = [], []
        for b in range(10):
            m = idx == b
            if m.sum() > 0:
                xs.append(p[m, c].mean()); ys.append(y[m, c].mean())
        axes[c].plot([0, 1], [0, 1], "k--", alpha=0.4, label="perfect")
        axes[c].plot(xs, ys, marker="o", label=model)
        axes[c].set_title(f"Calibration — {lab}")
        axes[c].set_xlabel("Mean predicted prob"); axes[c].set_ylabel("Observed freq")
        axes[c].legend(); axes[c].grid(alpha=0.3)
    _save(fig, "05_calibration")


def plot_prob_histograms(model, subN):
    t = _load_test(model)
    if t is None:
        return
    y, p = _subsample(t[0], t[1], subN)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for c, lab in enumerate(LABELS):
        axes[c].hist(p[y[:, c] == 1, c], bins=50, alpha=0.6,
                     density=True, label=f"true {lab}=1")
        axes[c].hist(p[y[:, c] == 0, c], bins=50, alpha=0.6,
                     density=True, label=f"true {lab}=0")
        axes[c].axvline(0.5, color="k", ls="--", alpha=0.5)
        axes[c].set_title(f"P({lab}) distribution"); axes[c].set_xlabel("Predicted prob")
        axes[c].set_ylabel("Density"); axes[c].legend(); axes[c].grid(alpha=0.3)
    _save(fig, "06_prob_histograms")


def plot_threshold_sweep(model, subN):
    t = _load_test(model)
    if t is None:
        return
    y, p = _subsample(t[0], t[1], subN)
    ths = np.linspace(0.05, 0.95, 19)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for c, lab in enumerate(LABELS):
        f1s, prs, rcs = [], [], []
        for th in ths:
            pred = (p[:, c] >= th).astype(int)
            f1s.append(f1_score(y[:, c], pred, zero_division=0))
            prs.append(precision_score(y[:, c], pred, zero_division=0))
            rcs.append(recall_score(y[:, c], pred, zero_division=0))
        best = ths[int(np.argmax(f1s))]
        axes[c].plot(ths, f1s, marker="o", ms=3, label="F1")
        axes[c].plot(ths, prs, marker="s", ms=3, label="Precision")
        axes[c].plot(ths, rcs, marker="^", ms=3, label="Recall")
        axes[c].axvline(best, color="green", ls="--", alpha=0.6,
                        label=f"best-F1 th={best:.2f}")
        axes[c].set_title(f"Threshold sweep — {lab}")
        axes[c].set_xlabel("Threshold"); axes[c].legend(); axes[c].grid(alpha=0.3)
    _save(fig, "07_threshold_sweep")


def plot_joint_prob(model, subN):
    t = _load_test(model)
    if t is None:
        return
    _, p = _subsample(t[0], t[1], min(subN, 300000))
    fig, ax = plt.subplots(figsize=(7, 6))
    hb = ax.hexbin(p[:, 0], p[:, 1], gridsize=50, bins="log", cmap="viridis")
    ax.axvline(0.5, color="w", ls="--", alpha=0.6)
    ax.axhline(0.5, color="w", ls="--", alpha=0.6)
    ax.set_xlabel("P(enhancer)"); ax.set_ylabel("P(silencer)")
    ax.set_title(f"Joint probability density — {model} (test)")
    fig.colorbar(hb, label="log10(count)")
    _save(fig, "08_joint_prob_hexbin")


def plot_per_label_confusion(model, subN):
    t = _load_test(model)
    if t is None:
        return
    y, p = _subsample(t[0], t[1], subN)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for c, lab in enumerate(LABELS):
        cm = confusion_matrix(y[:, c], (p[:, c] >= 0.5).astype(int))
        im = axes[c].imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            axes[c].text(j, i, f"{v:,}", ha="center",
                         color="white" if v > cm.max() / 2 else "black")
        axes[c].set_title(f"Confusion @0.5 — {lab}")
        axes[c].set_xticks([0, 1]); axes[c].set_yticks([0, 1])
        axes[c].set_xticklabels(["neg", "pos"]); axes[c].set_yticklabels(["neg", "pos"])
        axes[c].set_xlabel("Predicted"); axes[c].set_ylabel("True")
    _save(fig, "09_per_label_confusion")


# ---------- Uncharacterized predictions ----------

def _load_summary():
    p = os.path.join(PRED_DIR, "prediction_summary.json")
    return json.load(open(p)) if os.path.isfile(p) else {}


def plot_category_bar():
    s = _load_summary()
    if not s:
        return
    counts = [s.get(c, 0) for c in CAT_ORDER]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(CAT_ORDER, counts, color=[CAT_COLORS[c] for c in CAT_ORDER])
    total = sum(counts)
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"{c:,}\n({100*c/total:.2f}%)", ha="center", va="bottom", fontsize=9)
    ax.set_yscale("log"); ax.set_ylabel("Peak count (log)")
    ax.set_title(f"Uncharacterized region categories (n={total:,} peaks)")
    ax.grid(alpha=0.3, axis="y")
    _save(fig, "10_category_bar")


def plot_category_donut():
    s = _load_summary()
    if not s:
        return
    counts = [s.get(c, 0) for c in CAT_ORDER]
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(counts, labels=CAT_ORDER, colors=[CAT_COLORS[c] for c in CAT_ORDER],
           autopct=lambda pct: f"{pct:.2f}%", pctdistance=0.8,
           wedgeprops=dict(width=0.4))
    ax.set_title("Category composition (uncharacterized peaks)")
    _save(fig, "11_category_donut")


def _scan_predictions(max_cells, collect_rows):
    """Scan per-cell CSVs. Returns (per_sample_counts, sampled_rows).
    per_sample_counts[sample][category] = count.
    sampled_rows: list of (category, p_enh, p_sil, n_tfs) up to collect_rows."""
    per_sample = defaultdict(lambda: defaultdict(int))
    sampled = []
    rng = np.random.default_rng(0)
    csvs = glob.glob(os.path.join(PRED_DIR, "*", "*", "*_predictions.csv"))
    if max_cells and len(csvs) > max_cells:
        csvs = list(rng.choice(csvs, max_cells, replace=False))
    for path in csvs:
        sample = path.split(os.sep)[-3]
        try:
            with open(path) as f:
                rd = csv.DictReader(f)
                for row in rd:
                    cat = row.get("category", "")
                    per_sample[sample][cat] += 1
                    if collect_rows and len(sampled) < collect_rows and rng.random() < 0.02:
                        sampled.append((cat, float(row["p_enhancer"]),
                                        float(row["p_silencer"]), int(row["n_tfs"])))
        except Exception:
            continue
    return per_sample, sampled


def plot_per_sample_stacked(per_sample):
    if not per_sample:
        return
    samples = sorted(per_sample.keys())
    fig, ax = plt.subplots(figsize=(max(10, len(samples) * 0.5), 6))
    bottoms = np.zeros(len(samples))
    for cat in CAT_ORDER:
        vals = np.array([per_sample[s].get(cat, 0) for s in samples], dtype=float)
        # normalize to fraction per sample
        totals = np.array([sum(per_sample[s].values()) for s in samples], dtype=float)
        frac = np.divide(vals, totals, out=np.zeros_like(vals), where=totals > 0)
        ax.bar(samples, frac, bottom=bottoms, label=cat, color=CAT_COLORS[cat])
        bottoms += frac
    ax.set_ylabel("Category fraction"); ax.set_ylim(0, 1)
    ax.set_title("Per-sample category composition")
    ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1))
    plt.xticks(rotation=90)
    _save(fig, "12_per_sample_stacked")


def plot_ntfs_by_category(sampled):
    if not sampled:
        return
    by_cat = defaultdict(list)
    for cat, _, _, ntf in sampled:
        by_cat[cat].append(ntf)
    cats = [c for c in CAT_ORDER if by_cat.get(c)]
    fig, ax = plt.subplots(figsize=(8, 5))
    data = [by_cat[c] for c in cats]
    bp = ax.boxplot(data, labels=cats, showfliers=False, patch_artist=True)
    for patch, c in zip(bp["boxes"], cats):
        patch.set_facecolor(CAT_COLORS[c])
    ax.set_ylabel("n_tfs per peak"); ax.set_title("TF count by predicted category (sampled)")
    ax.grid(alpha=0.3, axis="y")
    _save(fig, "13_ntfs_by_category")


def plot_uncharacterized_joint(sampled):
    if not sampled:
        return
    pe = np.array([r[1] for r in sampled]); ps = np.array([r[2] for r in sampled])
    fig, ax = plt.subplots(figsize=(7, 6))
    hb = ax.hexbin(pe, ps, gridsize=45, bins="log", cmap="magma")
    ax.axvline(0.5, color="w", ls="--", alpha=0.6); ax.axhline(0.5, color="w", ls="--", alpha=0.6)
    ax.set_xlabel("P(enhancer)"); ax.set_ylabel("P(silencer)")
    ax.set_title("Uncharacterized predictions: P(enh) vs P(sil) (sampled)")
    fig.colorbar(hb, label="log10(count)")
    _save(fig, "14_uncharacterized_joint")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-model", default="ensemble",
                    help="model for single-model eval plots (calibration, "
                         "histograms, threshold, joint, confusion)")
    ap.add_argument("--subsample", type=int, default=500000,
                    help="rows subsampled from the test set for heavy plots")
    ap.add_argument("--pred-sample-cells", type=int, default=4000,
                    help="number of prediction CSVs to scan for per-sample / "
                         "n_tfs / joint plots (0 = all 68k, slow)")
    args = ap.parse_args()

    os.makedirs(FIG_DIR, exist_ok=True)
    eval_models = [m for m in MODELS
                   if os.path.isfile(os.path.join(MODEL_DIR, m, "test_predictions.npz"))]
    em = args.eval_model if args.eval_model in eval_models else (eval_models[-1] if eval_models else None)
    print(f"Eval models available: {eval_models}; single-model plots use: {em}")

    # Training / model
    for m in MODELS:
        _safe(f"01 training curves [{m}]", plot_training_curves, m)
    _safe("02 model comparison", plot_model_comparison)

    # Test-set evaluation
    if eval_models:
        _safe("03 ROC", plot_roc, eval_models, args.subsample)
        _safe("04 PR", plot_pr, eval_models, args.subsample)
        _safe("05 calibration", plot_calibration, em, args.subsample)
        _safe("06 prob histograms", plot_prob_histograms, em, args.subsample)
        _safe("07 threshold sweep", plot_threshold_sweep, em, args.subsample)
        _safe("08 joint prob", plot_joint_prob, em, args.subsample)
        _safe("09 per-label confusion", plot_per_label_confusion, em, args.subsample)

    # Uncharacterized predictions
    _safe("10 category bar", plot_category_bar)
    _safe("11 category donut", plot_category_donut)
    print(f"Scanning prediction CSVs (sample={args.pred_sample_cells})...")
    per_sample, sampled = _scan_predictions(args.pred_sample_cells, collect_rows=200000)
    _safe("12 per-sample stacked", plot_per_sample_stacked, per_sample)
    _safe("13 n_tfs by category", plot_ntfs_by_category, sampled)
    _safe("14 uncharacterized joint", plot_uncharacterized_joint, sampled)

    print(f"\nAll figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
