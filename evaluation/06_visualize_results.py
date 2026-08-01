#!/usr/bin/env python3
"""
06_visualize_results.py
========================
Generate evaluation visualizations:
  1. Training curves (loss, F1, AUROC over epochs)
  2. Confusion matrix heatmap
  3. Per-class ROC curves
  4. Per-class precision-recall curves
  5. UMAP of learned embeddings colored by class
  6. Prediction confidence distribution
  7. Class distribution comparison (training vs. predictions)

Usage:
    python 06_visualize_results.py
"""

import importlib
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.preprocessing import label_binarize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE_DIR = "/path/to/scgram"
MODEL_DIR = os.path.join(BASE_DIR, "model_output")
PRED_DIR = os.path.join(BASE_DIR, "predictions")
FIG_DIR = os.path.join(BASE_DIR, "figures")


def plot_training_curves(history, out_dir):
    """Plot training loss, validation loss, F1, AUROC over epochs."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    axes[0].plot(epochs, history["train_loss"], "b-", label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], "r-", label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Macro F1
    axes[1].plot(epochs, history["val_macro_f1"], "g-", marker="o", markersize=3)
    best_epoch = np.argmax(history["val_macro_f1"]) + 1
    best_f1 = max(history["val_macro_f1"])
    axes[1].axvline(x=best_epoch, color="gray", linestyle="--", alpha=0.5)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro F1")
    axes[1].set_title(f"Validation Macro F1 (best={best_f1:.4f} @ epoch {best_epoch})")
    axes[1].grid(True, alpha=0.3)

    # AUROC
    axes[2].plot(epochs, history["val_auroc"], "m-", marker="o", markersize=3)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("AUROC")
    axes[2].set_title("Validation AUROC")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_curves.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(out_dir, "training_curves.pdf"), dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved training_curves.png/pdf")


def plot_confusion_matrix(cm, label_list, out_dir):
    """Plot confusion matrix heatmap."""
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Raw counts
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_list, yticklabels=label_list, ax=axes[0])
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    axes[0].set_title("Confusion Matrix (counts)")

    # Normalized (row-wise = recall)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=label_list, yticklabels=label_list, ax=axes[1])
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    axes[1].set_title("Confusion Matrix (normalized by true class)")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(out_dir, "confusion_matrix.pdf"), dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved confusion_matrix.png/pdf")


def plot_roc_curves(labels, probs, label_list, out_dir):
    """Plot per-class ROC curves."""
    n_classes = len(label_list)
    labels_bin = label_binarize(labels, classes=range(n_classes))

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))

    for i, (label, color) in enumerate(zip(label_list, colors)):
        if labels_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(labels_bin[:, i], probs[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{label} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Per-Class ROC Curves")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "roc_curves.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(out_dir, "roc_curves.pdf"), dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved roc_curves.png/pdf")


def plot_pr_curves(labels, probs, label_list, out_dir):
    """Plot per-class Precision-Recall curves."""
    n_classes = len(label_list)
    labels_bin = label_binarize(labels, classes=range(n_classes))

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))

    for i, (label, color) in enumerate(zip(label_list, colors)):
        if labels_bin[:, i].sum() == 0:
            continue
        precision, recall, _ = precision_recall_curve(labels_bin[:, i], probs[:, i])
        ap = average_precision_score(labels_bin[:, i], probs[:, i])
        ax.plot(recall, precision, color=color, lw=2,
                label=f"{label} (AP={ap:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Per-Class Precision-Recall Curves")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pr_curves.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(out_dir, "pr_curves.pdf"), dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved pr_curves.png/pdf")


def plot_embeddings_umap(model_dir, data_dir, label_list, out_dir, max_samples=10000):
    """Generate UMAP of learned embeddings (requires umap-learn)."""
    try:
        import umap
    except ImportError:
        print("  umap-learn not installed, skipping embedding visualization.")
        return

    import torch

    _model_mod = importlib.import_module("03_model")
    RegulatoryClassifier = _model_mod.RegulatoryClassifier
    RegulomeDataset = _model_mod.RegulomeDataset

    # Load model
    ckpt = torch.load(os.path.join(model_dir, "best_model.pt"), map_location="cpu")
    model = RegulatoryClassifier(
        n_classes=ckpt["n_classes"],
        n_tfs=ckpt["n_tfs"],
        seq_length=ckpt["seq_length"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Load test data
    test_chroms = {"chr21", "chr22", "chrX"}
    dataset = RegulomeDataset(data_dir, chromosomes=test_chroms, augment=False)

    n = min(len(dataset), max_samples)
    indices = np.random.choice(len(dataset), n, replace=False)

    embeddings = []
    labels = []

    with torch.no_grad():
        for idx in indices:
            seq, tf_feat, tab_feat, label = dataset[idx]
            emb = model.get_embeddings(
                seq.unsqueeze(0), tf_feat.unsqueeze(0), tab_feat.unsqueeze(0)
            )
            embeddings.append(emb.squeeze(0).numpy())
            labels.append(label.item())

    embeddings = np.array(embeddings)
    labels = np.array(labels)

    # UMAP
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, metric="cosine", random_state=42)
    embedding_2d = reducer.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(12, 10))

    colors = plt.cm.tab10(np.linspace(0, 1, len(label_list)))
    for i, (label, color) in enumerate(zip(label_list, colors)):
        mask = labels == i
        if mask.sum() == 0:
            continue
        ax.scatter(embedding_2d[mask, 0], embedding_2d[mask, 1],
                   c=[color], label=label, alpha=0.5, s=10)

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("UMAP of Learned Embeddings (Test Set)")
    ax.legend(markerscale=3, fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "umap_embeddings.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(out_dir, "umap_embeddings.pdf"), dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved umap_embeddings.png/pdf")


def plot_prediction_confidence(pred_dir, label_list, out_dir):
    """Plot confidence distribution of predictions on uncharacterized regions."""
    import pandas as pd

    all_dfs = []
    for sample_dir in sorted(os.listdir(pred_dir)):
        sample_path = os.path.join(pred_dir, sample_dir)
        if not os.path.isdir(sample_path) or sample_dir == "figures":
            continue
        for celltype in os.listdir(sample_path):
            ct_path = os.path.join(sample_path, celltype)
            if not os.path.isdir(ct_path):
                continue
            for f in os.listdir(ct_path):
                if f.endswith("_predictions.csv"):
                    try:
                        df = pd.read_csv(os.path.join(ct_path, f))
                        all_dfs.append(df)
                    except Exception:
                        continue

    if not all_dfs:
        print("  No prediction files found, skipping confidence plot.")
        return

    combined = pd.concat(all_dfs, ignore_index=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Overall confidence distribution
    axes[0].hist(combined["confidence"], bins=50, edgecolor="black", alpha=0.7)
    axes[0].axvline(x=0.8, color="red", linestyle="--", label="Threshold (0.8)")
    axes[0].set_xlabel("Prediction Confidence")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Confidence Distribution (All Predictions)")
    axes[0].legend()

    # Per-class confidence
    for label in label_list:
        subset = combined[combined["predicted_class"] == label]["confidence"]
        if len(subset) > 0:
            axes[1].hist(subset, bins=30, alpha=0.5, label=label)
    axes[1].set_xlabel("Prediction Confidence")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Confidence Distribution by Predicted Class")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "prediction_confidence.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(out_dir, "prediction_confidence.pdf"), dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved prediction_confidence.png/pdf")

    # Class distribution bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    class_counts = combined["predicted_class"].value_counts()
    high_conf = combined[combined["high_confidence"] == "YES"]["predicted_class"].value_counts()

    x = np.arange(len(class_counts))
    width = 0.35
    bars1 = ax.bar(x - width/2, [class_counts.get(l, 0) for l in label_list],
                   width, label="All predictions", color="steelblue")
    bars2 = ax.bar(x + width/2, [high_conf.get(l, 0) for l in label_list],
                   width, label="High confidence (>=0.8)", color="darkorange")

    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Count")
    ax.set_title("Predicted Class Distribution for Uncharacterized Regions")
    ax.set_xticks(x)
    ax.set_xticklabels(label_list, rotation=45, ha="right")
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "predicted_class_distribution.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(out_dir, "predicted_class_distribution.pdf"), dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved predicted_class_distribution.png/pdf")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
    })

    # Resolve the results JSON. Prefer a per-model file with full history
    # (hybrid > cnn), then ensemble. The legacy top-level path may not exist
    # under the multi-model + multilabel layout.
    candidate_results = [
        os.path.join(MODEL_DIR, "training_results.json"),
        os.path.join(MODEL_DIR, "hybrid", "training_results.json"),
        os.path.join(MODEL_DIR, "cnn", "training_results.json"),
        os.path.join(MODEL_DIR, "ensemble", "training_results.json"),
    ]
    results_path = next((p for p in candidate_results if os.path.isfile(p)), None)
    results = {}
    if results_path:
        with open(results_path) as f:
            results = json.load(f)
        print(f"Using results from {results_path}")
    else:
        print("  No training_results.json found; skipping training-curve plots.")

    # Label list from dataset_info_mmap (multilabel) or dataset_info (multiclass)
    label_list = None
    for ip in (os.path.join(BASE_DIR, "training_data", "mmap", "dataset_info_mmap.json"),
               os.path.join(BASE_DIR, "training_data", "dataset_info.json")):
        if os.path.isfile(ip):
            with open(ip) as f:
                info = json.load(f)
            label_list = info.get("multilabel_classes") or info.get("label_list")
            break
    if not label_list:
        label_list = ["is_enhancer", "is_silencer"]

    is_multilabel = (results.get("task") == "multilabel"
                     or "test_per_label" in results
                     or set(label_list) == {"is_enhancer", "is_silencer"})

    print("Generating visualizations...\n")

    def _safe(step_name, fn, *a, **k):
        """Run a plotting step; never let one failure abort the rest."""
        try:
            print(step_name)
            fn(*a, **k)
        except Exception as e:
            print(f"  [skip] {step_name} failed: {e}")

    # 1. Training curves (history present in per-model results)
    if "history" in results:
        _safe("1. Training curves", plot_training_curves, results["history"], FIG_DIR)

    # 2/3/4. Confusion matrix + ROC/PR — multiclass-only helpers; for
    # multilabel we rely on the per-label metrics already in the JSON and
    # the ensemble test_predictions. Attempt ROC/PR per label.
    test_pred_path = next(
        (p for p in (os.path.join(MODEL_DIR, "ensemble", "test_predictions.npz"),
                     os.path.join(MODEL_DIR, "hybrid", "test_predictions.npz"),
                     os.path.join(MODEL_DIR, "cnn", "test_predictions.npz"))
         if os.path.isfile(p)), None)
    if not is_multilabel and "confusion_matrix" in results:
        _safe("2. Confusion matrix", plot_confusion_matrix,
              np.array(results["confusion_matrix"]), label_list, FIG_DIR)
    if test_pred_path:
        td = np.load(test_pred_path)
        _safe("3. ROC curves", plot_roc_curves,
              td["labels"], td["probabilities"], label_list, FIG_DIR)
        _safe("4. Precision-Recall curves", plot_pr_curves,
              td["labels"], td["probabilities"], label_list, FIG_DIR)
    else:
        print("  Test predictions not found, skipping ROC/PR curves.")

    # 5. UMAP embeddings — gated OFF by default. At this dataset scale
    # (hundreds of millions of peaks) embedding extraction + UMAP fitting is
    # impractical and not part of the core deliverable. Enable with
    # PLOT_UMAP=1 if desired (will be slow).
    if os.environ.get("PLOT_UMAP", "0") == "1":
        _safe("5. UMAP embeddings", plot_embeddings_umap,
              MODEL_DIR, os.path.join(BASE_DIR, "training_data"), label_list, FIG_DIR)
    else:
        print("5. UMAP embeddings — skipped (set PLOT_UMAP=1 to enable)")

    # 6. Prediction confidence / category distribution over uncharacterized
    _safe("6. Prediction confidence distributions",
          plot_prediction_confidence, PRED_DIR, label_list, FIG_DIR)

    print(f"\nAll figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
