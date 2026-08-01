#!/usr/bin/env python3
"""
07_calibrate_predictions.py
============================
Post-hoc calibration of trained model predictions. Operates on existing
test_predictions.npz files; does not retrain or touch training data.

Three calibration approaches, applied independently and in combination:

  1. RAW — argmax(probs) (baseline, the model's uncalibrated decision).

  2. PRIOR-CORRECTED — Bayes correction for the train/test prior shift
     introduced by class-balanced sampling. The sampler trained the model
     on a uniform class prior P_train(y) = 1/n_classes; at inference, the
     correct prior is the empirical class distribution P_test(y). Apply:

         P_corrected(y|x)  ∝  P_model(y|x) × P_test(y) / P_train(y)

     Renormalize, argmax. No val data required, no leakage.

  3. THRESHOLD-TUNED-ON-TEST — for each class, find the probability
     threshold (in raw probs and in prior-corrected probs) that maximizes
     one-vs-rest F1 on the test set. Uses test labels to fit, so this is
     OVERFIT to test and overestimates real-world F1. Reported as a
     diagnostic upper bound — NOT something to deploy.
     Proper threshold tuning requires saved val probs, which the current
     04_train.py does not write; future work.

Outputs (per model dir):
  calibrated_predictions.npz  — corrected probs + argmax preds for each method
  calibrated_results.json     — metrics for each method side-by-side

Usage:
  python 07_calibrate_predictions.py
  python 07_calibrate_predictions.py --models cnn transformer hybrid ensemble
  python 07_calibrate_predictions.py --model-output-dir /path/to/model_output
"""

import argparse
import json
import os
import sys

import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)


def metrics_from(labels, preds, probs, label_list):
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(labels, preds, average="weighted", zero_division=0)
    try:
        auroc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
    except ValueError:
        auroc = 0.0
    report = classification_report(
        labels, preds, target_names=label_list, zero_division=0, output_dict=True,
    )
    cm = confusion_matrix(labels, preds, labels=list(range(len(label_list))))
    return {
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "auroc": float(auroc),
        "report": report,
        "confusion_matrix": cm.tolist(),
    }


def prior_corrected_probs(probs, labels, n_classes):
    """Apply Bayes correction for the class-balanced-train -> natural-eval
    prior shift. Returns (n, n_classes) corrected probabilities.

    Math: under a balanced sampler, the model implicitly learns
        P_train(y|x) ∝ P(x|y) × (1/n_classes)
    The correct posterior on natural data is
        P_test(y|x) ∝ P(x|y) × P_test(y)
    so multiply by P_test(y) / P_train(y) and renormalize.
    """
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    p_test = counts / counts.sum()
    p_train = np.full(n_classes, 1.0 / n_classes, dtype=np.float64)
    factor = (p_test / p_train).reshape(1, -1)  # (1, n_classes)
    corrected = probs.astype(np.float64) * factor
    # Avoid divide-by-zero on rows that drop to 0 across the board
    row_sums = corrected.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums <= 0, 1.0, row_sums)
    corrected = corrected / row_sums
    return corrected.astype(np.float32)


def fit_per_class_thresholds(probs, labels, n_classes):
    """For each class c, return the probability threshold that maximizes
    the one-vs-rest F1 on (probs, labels). Default 0.5 if a class has no
    positive samples or no usable thresholds.

    NOTE: when called on TEST data this overfits — only use as a
    diagnostic upper bound, not for deployment.
    """
    thresholds = np.full(n_classes, 0.5, dtype=np.float64)
    for c in range(n_classes):
        y_true = (labels == c).astype(np.int64)
        if y_true.sum() == 0:
            continue
        y_score = probs[:, c]
        precs, recs, thrs = precision_recall_curve(y_true, y_score)
        if len(thrs) == 0:
            continue
        # precs/recs have len(thrs)+1 elements; the last pair is (1, 0)
        # with no associated threshold. F1 across the first N pairs maps
        # 1:1 to thrs[0..N-1].
        f1s = 2 * precs * recs / np.maximum(precs + recs, 1e-12)
        best_idx = int(np.argmax(f1s[:-1])) if len(f1s) > 1 else 0
        if 0 <= best_idx < len(thrs):
            thresholds[c] = float(thrs[best_idx])
    return thresholds


def predict_with_thresholds(probs, thresholds):
    """Multi-class prediction from per-class thresholds: pick class c that
    maximizes (probs[c] - thresholds[c]). Handles the case where no class
    is "above its threshold" by picking the closest one (max margin)."""
    margins = probs.astype(np.float64) - thresholds.reshape(1, -1)
    return np.argmax(margins, axis=1).astype(np.int64)


def calibrate_one_model(model_dir, label_list):
    pred_path = os.path.join(model_dir, "test_predictions.npz")
    if not os.path.isfile(pred_path):
        print(f"  [skip] no test_predictions.npz in {model_dir}")
        return None

    print(f"\n{'='*70}")
    print(f"  CALIBRATING: {model_dir}")
    print(f"{'='*70}")

    data = np.load(pred_path)
    raw_preds = data["predictions"].astype(np.int64)
    labels = data["labels"].astype(np.int64)
    probs = data["probabilities"].astype(np.float32)
    n_classes = probs.shape[1]
    if len(label_list) != n_classes:
        print(f"  [warn] label_list len {len(label_list)} != n_classes {n_classes}; "
              f"falling back to generic names")
        label_list = [f"c{i}" for i in range(n_classes)]

    print(f"  test set: {len(labels):,} samples, {n_classes} classes")
    print(f"  natural prior on test:")
    counts = np.bincount(labels, minlength=n_classes)
    for i, name in enumerate(label_list):
        pct = 100 * counts[i] / max(len(labels), 1)
        print(f"    {name:>20}: {counts[i]:>10,} ({pct:>7.4f}%)")

    # 1. RAW
    raw = metrics_from(labels, raw_preds, probs, label_list)

    # 2. PRIOR-CORRECTED
    pc_probs = prior_corrected_probs(probs, labels, n_classes)
    pc_preds = pc_probs.argmax(axis=1).astype(np.int64)
    pc = metrics_from(labels, pc_preds, pc_probs, label_list)

    # 3. THRESHOLD-TUNED-ON-TEST (overfit diagnostic)
    thresholds = fit_per_class_thresholds(probs, labels, n_classes)
    tt_preds = predict_with_thresholds(probs, thresholds)
    tt = metrics_from(labels, tt_preds, probs, label_list)

    # 3a. PRIOR + THRESHOLD-ON-CORRECTED-PROBS (also overfit)
    pc_thresholds = fit_per_class_thresholds(pc_probs, labels, n_classes)
    pct_preds = predict_with_thresholds(pc_probs, pc_thresholds)
    pct = metrics_from(labels, pct_preds, pc_probs, label_list)

    print(f"\n  Macro-F1 comparison:")
    print(f"    Raw (uncalibrated):                 {raw['macro_f1']:.4f}")
    print(f"    Prior-corrected:                    {pc['macro_f1']:.4f}")
    print(f"    Threshold-tuned-on-test (OVERFIT):  {tt['macro_f1']:.4f}")
    print(f"    Prior + threshold (OVERFIT):        {pct['macro_f1']:.4f}")
    print(f"  Weighted-F1 (prior-corrected):        {pc['weighted_f1']:.4f}")
    print(f"  AUROC (invariant to argmax):          {raw['auroc']:.4f}")

    print(f"\n  Per-class F1 (prior-corrected):")
    for name in label_list:
        cls = pc["report"].get(name, {})
        print(f"    {name:>20}: F1={cls.get('f1-score', 0.0):.4f}  "
              f"P={cls.get('precision', 0.0):.4f}  "
              f"R={cls.get('recall', 0.0):.4f}  "
              f"support={int(cls.get('support', 0)):,}")

    # ---- Persist ----
    out_results = {
        "model_dir": model_dir,
        "n_test": int(len(labels)),
        "n_classes": int(n_classes),
        "label_list": list(label_list),
        "test_natural_prior": {
            label_list[i]: float(counts[i] / max(len(labels), 1))
            for i in range(n_classes)
        },
        "raw": raw,
        "prior_corrected": pc,
        "threshold_tuned_on_test_OVERFIT": tt,
        "prior_corrected_plus_threshold_OVERFIT": pct,
        "tuned_thresholds_overfit": thresholds.tolist(),
        "tuned_thresholds_after_prior_overfit": pc_thresholds.tolist(),
        "_notes": (
            "RAW and PRIOR-CORRECTED are honest. THRESHOLD-TUNED metrics use "
            "test labels to fit thresholds and therefore overfit to test; "
            "they are reported only as an upper bound on what proper "
            "val-fit thresholds could achieve. To get principled threshold "
            "tuning, save val probabilities and fit thresholds on val "
            "before applying to test."
        ),
    }
    out_path = os.path.join(model_dir, "calibrated_results.json")
    with open(out_path + ".tmp", "w") as f:
        json.dump(out_results, f, indent=2)
    os.replace(out_path + ".tmp", out_path)
    print(f"\n  Wrote {out_path}")

    np.savez_compressed(
        os.path.join(model_dir, "calibrated_predictions.npz"),
        labels=labels,
        raw_predictions=raw_preds,
        raw_probabilities=probs,
        prior_corrected_predictions=pc_preds,
        prior_corrected_probabilities=pc_probs,
        threshold_tuned_predictions=tt_preds,
        prior_plus_threshold_predictions=pct_preds,
        thresholds=thresholds,
        thresholds_after_prior=pc_thresholds,
    )
    print(f"  Wrote {os.path.join(model_dir, 'calibrated_predictions.npz')}")
    return out_results


def discover_label_list(model_output_dir, models):
    """Pull the class-name ordering from the first model's training_results.json."""
    for m in models:
        tr_path = os.path.join(model_output_dir, m, "training_results.json")
        if os.path.isfile(tr_path):
            with open(tr_path) as f:
                tr = json.load(f)
            keys = list(tr.get("test_report", {}).keys())
            cleaned = [k for k in keys
                       if k not in ("accuracy", "macro avg", "weighted avg")]
            if cleaned:
                return cleaned
    return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-output-dir",
        default="/path/to/scgram/model_output",
    )
    parser.add_argument(
        "--models", nargs="*", default=None,
        help="model subdir names; default = auto-discover any with test_predictions.npz",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.model_output_dir):
        sys.exit(f"model output dir not found: {args.model_output_dir}")

    if args.models is None:
        discovered = []
        for name in sorted(os.listdir(args.model_output_dir)):
            d = os.path.join(args.model_output_dir, name)
            if os.path.isfile(os.path.join(d, "test_predictions.npz")):
                discovered.append(name)
        models = discovered
    else:
        models = args.models

    if not models:
        sys.exit(f"No models with test_predictions.npz found under "
                 f"{args.model_output_dir}")

    label_list = discover_label_list(args.model_output_dir, models)

    print(f"Models to calibrate: {models}")
    print(f"Label list: {label_list}")

    summary = {}
    for m in models:
        d = os.path.join(args.model_output_dir, m)
        result = calibrate_one_model(d, label_list)
        if result is not None:
            summary[m] = {
                "raw": result["raw"]["macro_f1"],
                "prior": result["prior_corrected"]["macro_f1"],
                "thresh_overfit": result["threshold_tuned_on_test_OVERFIT"]["macro_f1"],
                "prior_thresh_overfit": result["prior_corrected_plus_threshold_OVERFIT"]["macro_f1"],
                "auroc": result["raw"]["auroc"],
            }

    print(f"\n{'='*70}")
    print(f"  CALIBRATION SUMMARY (macro F1)")
    print(f"{'='*70}")
    print(f"  {'model':<13} {'raw':>8} {'+prior':>8} {'+thresh*':>10} {'+both*':>10} {'AUROC':>8}")
    print(f"  {'-'*13} {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*8}")
    for m, s in summary.items():
        print(f"  {m:<13} {s['raw']:>8.4f} {s['prior']:>8.4f} "
              f"{s['thresh_overfit']:>10.4f} {s['prior_thresh_overfit']:>10.4f} "
              f"{s['auroc']:>8.4f}")
    print(f"  * threshold-tuned columns use test labels to fit thresholds — "
          f"OVERFIT, diagnostic only.")


if __name__ == "__main__":
    main()
