#!/usr/bin/env python3
"""
benchmarks/run_classical.py
===========================
Classical (non-deep) baselines on the SAME chromosome-disjoint split used
by the main pipeline. STREAMING loaders — no full-set materialization,
no subsampling. Three baselines:

  logistic-tf      SGDClassifier(log_loss) on 879-d TF presence vector
  logistic-kmer    SGDClassifier(log_loss) on 4^6=4096-d 6-mer counts
  xgboost          xgb.QuantileDMatrix(streamed iterator) on TF + 6-mer + tab

Why streaming: at 462 M training peaks, even uint8 TF features alone are
~407 GB; float32 is 1.6 TB. SGD's partial_fit and XGBoost's external-memory
DataIter consume one chunk at a time, so the only resident state is the
model parameters.

Each baseline trains independently per label (4 sigmoid outputs in
--label-set full). Test predictions written to results.json + .npz.

Usage:
    python benchmarks/run_classical.py --baseline logistic-tf
"""

import argparse
import json
import os
import time

import numpy as np
from scipy.special import expit  # numerically stable sigmoid
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (average_precision_score, f1_score, roc_auc_score)

BASE = "/path/to/scgram"
MMAP = os.path.join(BASE, "training_data", "mmap")
INFO_PATH = os.path.join(MMAP, "dataset_info_mmap.json")
OUT_ROOT = os.path.join(BASE, "benchmarks", "results")

TRAIN_CHROMS = {f"chr{i}" for i in range(1, 18)}
TEST_CHROMS = {"chr21", "chr22", "chrX"}
LABELS_FULL = ["is_enhancer", "is_promoter", "is_genic", "is_silencer"]

K = 6
N_KMER = 4 ** K
POW4 = np.array([4 ** (K - 1 - i) for i in range(K)], dtype=np.int64)


def load_info():
    with open(INFO_PATH) as f:
        return json.load(f)


def chunk_ids_for(info, chroms):
    code_map = info["chrom_code_map"]
    allowed = {int(code_map[c]) for c in chroms if c in code_map}
    n_chunks = info["n_chunks"]
    out = []
    for ci in range(n_chunks):
        chrom_p = os.path.join(MMAP, f"chrom_{ci:04d}.npy")
        if not os.path.isfile(chrom_p):
            continue
        chrom = np.load(chrom_p, mmap_mode="r")
        mask = np.isin(chrom, list(allowed))
        if mask.any():
            out.append((ci, np.nonzero(mask)[0].astype(np.int64)))
    return out


def kmer_counts(seq8_block):
    """(n, L) uint8 → (n, 4^K) float32 counts (l1-normalized)."""
    n, L = seq8_block.shape
    seq = seq8_block.astype(np.int32, copy=True)
    seq[seq > 3] = 0
    out = np.zeros((n, N_KMER), dtype=np.float32)
    rows = np.arange(n)
    for off in range(L - K + 1):
        ids = (seq[:, off:off + K] * POW4).sum(axis=1)
        np.add.at(out, (rows, ids), 1.0)
    out /= max(L - K + 1, 1)
    return out


def features_for_chunk(ci, idxs, info, baseline, label_set="full"):
    """Load the feature matrix + labels for one chunk slice."""
    n_tfs = info["n_tfs"]
    lab_prefix = "lab4" if label_set == "full" else "lab2"
    tab = np.asarray(np.load(os.path.join(MMAP, f"tab_{ci:04d}.npy"),
                             mmap_mode="r")[idxs], dtype=np.float32)
    lab = np.asarray(np.load(os.path.join(MMAP, f"{lab_prefix}_{ci:04d}.npy"),
                             mmap_mode="r")[idxs], dtype=np.uint8)
    feats = []
    if baseline in ("logistic-tf", "xgboost"):
        tfb_p = os.path.join(MMAP, f"tfb_{ci:04d}.npy")
        if os.path.isfile(tfb_p):
            tfb = np.asarray(np.load(tfb_p, mmap_mode="r")[idxs], dtype=np.uint8)
            tf = np.unpackbits(tfb, axis=1)[:, :n_tfs].astype(np.float32)
        else:
            tf = (np.asarray(np.load(os.path.join(MMAP, f"tf_{ci:04d}.npy"),
                                     mmap_mode="r")[idxs]) > 0).astype(np.float32)
        feats.append(tf)
    # k-mer is logistic-kmer-only. Tried xgboost on (TF + k-mer + tab),
    # but k-mer compute (np.add.at) dominates: ~24 s/chunk × 10563 chunks ×
    # 4 labels (xgboost trains per-label, not amortized) → ~14 days. Dropped
    # k-mer here so xgboost runs on TF + tabular only — complements the
    # logistic-tf row as the non-linear classifier on the same feature set.
    if baseline == "logistic-kmer":
        seq8 = np.asarray(np.load(os.path.join(MMAP, f"seq8_{ci:04d}.npy"),
                                  mmap_mode="r")[idxs])
        feats.append(kmer_counts(seq8))
    if baseline == "xgboost":
        feats.append(tab)
    return np.concatenate(feats, axis=1), lab


# --------------------- SGDClassifier streaming train ---------------------

def train_sgd_all(info, baseline, n_labels, chunks, label_set="full", epochs=3):
    """Train ALL labels in one pass over the data — features for each chunk
    are decoded once and shared across the n_labels binary classifiers.
    Cuts I/O and k-mer compute by n_labels× vs. one-at-a-time."""
    # sklearn's SGDClassifier rejects class_weight='balanced' with partial_fit;
    # precompute explicit per-class weights from the full label tensor (cheap,
    # only reads lab*_*.npy files).
    lab_prefix = "lab4" if label_set == "full" else "lab2"
    counts = np.zeros((n_labels, 2), dtype=np.int64)
    for ci, idxs in chunks:
        lab = np.asarray(np.load(os.path.join(MMAP, f"{lab_prefix}_{ci:04d}.npy"),
                                 mmap_mode="r")[idxs], dtype=np.uint8)
        for c in range(n_labels):
            counts[c, 1] += int(lab[:, c].sum())
            counts[c, 0] += int(len(lab) - lab[:, c].sum())
    cw_list = []
    for c in range(n_labels):
        n_neg, n_pos = int(counts[c, 0]), int(counts[c, 1])
        N = max(n_neg + n_pos, 1)
        cw_list.append({0: N / (2 * max(n_neg, 1)),
                        1: N / (2 * max(n_pos, 1))})
    print(f"    class weights per label: {cw_list}")
    clfs = [SGDClassifier(loss="log_loss", alpha=1e-6, max_iter=1, tol=None,
                          learning_rate="optimal", class_weight=cw_list[c],
                          average=True, random_state=int(c))
            for c in range(n_labels)]
    classes = np.array([0, 1], dtype=np.uint8)
    rng = np.random.default_rng(0)
    for ep in range(epochs):
        order = rng.permutation(len(chunks))
        t0 = time.time()
        for n_seen, k in enumerate(order, 1):
            ci, idxs = chunks[k]
            X, y = features_for_chunk(ci, idxs, info, baseline, label_set)
            for c in range(n_labels):
                clfs[c].partial_fit(X, y[:, c], classes=classes)
            if n_seen % 500 == 0:
                print(f"    ep{ep+1} chunk {n_seen}/{len(order)} "
                      f"({time.time()-t0:.0f}s)", flush=True)
    return clfs


def predict_test_all(clfs, info, baseline, chunks, label_set="full"):
    """All labels at once — single feature decode per chunk."""
    n_labels = len(clfs)
    probs = [[] for _ in range(n_labels)]
    labels = [[] for _ in range(n_labels)]
    for ci, idxs in chunks:
        X, y = features_for_chunk(ci, idxs, info, baseline, label_set)
        for c in range(n_labels):
            s = clfs[c].decision_function(X)
            probs[c].append(expit(s))
            labels[c].append(y[:, c])
    return ([np.concatenate(p) for p in probs],
            [np.concatenate(l) for l in labels])


# --------------------- XGBoost external-memory iterator ---------------------

class XGBChunkIter:
    """xgb DataIter: streams chunks for QuantileDMatrix."""
    def __init__(self, info, baseline, label_idx, chunks):
        self.info = info; self.baseline = baseline
        self.label_idx = label_idx; self.chunks = chunks
        self._i = 0
        try:
            import xgboost as xgb
            self._DataIter = xgb.DataIter
        except Exception:
            self._DataIter = None
    def reset(self): self._i = 0
    def next(self, input_data):
        if self._i >= len(self.chunks): return 0
        ci, idxs = self.chunks[self._i]; self._i += 1
        X, y = features_for_chunk(ci, idxs, self.info, self.baseline)
        input_data(data=X, label=y[:, self.label_idx])
        return 1


def train_xgb(info, baseline, label_idx, chunks, label_set="full"):
    import xgboost as xgb
    # Empirical pos_weight from the WHOLE train set (cheap — labels only)
    lab_prefix = "lab4" if label_set == "full" else "lab2"
    n_pos = 0; n_tot = 0
    for ci, idxs in chunks:
        lab = np.load(os.path.join(MMAP, f"{lab_prefix}_{ci:04d}.npy"),
                      mmap_mode="r")[idxs]
        n_pos += int(lab[:, label_idx].sum()); n_tot += len(lab)
    spw = max((n_tot - n_pos) / max(n_pos, 1), 1.0)
    print(f"    pos_weight={spw:.1f}  (n_pos={n_pos:,}/{n_tot:,})")

    it = _XgbDataIter(info, baseline, label_idx, chunks, label_set)
    dtrain = xgb.QuantileDMatrix(it, max_bin=256)
    params = dict(
        # device="cpu" because GPU OOMs at ~530M rows × ~1100 features even
        # after QuantileDMatrix quantization (xgboost wants the compressed
        # matrix resident on the GPU; 381 GB requested vs 40 GB free).
        # CPU hist with our 1.5 TB system RAM handles this fine; slower per
        # round but viable.
        objective="binary:logistic", eval_metric="aucpr",
        tree_method="hist", device="cpu",
        max_depth=8, learning_rate=0.08,
        subsample=0.85, colsample_bytree=0.85,
        scale_pos_weight=spw, nthread=16,
    )
    booster = xgb.train(params, dtrain, num_boost_round=200)
    return booster


def _XgbDataIter(info, baseline, label_idx, chunks, label_set):
    """Late binding: xgb is imported only inside train_xgb."""
    import xgboost as xgb
    class _Iter(xgb.DataIter):
        def __init__(s):
            # QuantileDMatrix doesn't cache data — passing cache_prefix here
            # raises ValueError in xgboost >=2. Quantization (256 bins/feature)
            # keeps memory bounded without a disk cache.
            super().__init__()
            s.i = 0
        def reset(s): s.i = 0
        def next(s, input_data):
            if s.i >= len(chunks): return 0
            ci, idxs = chunks[s.i]; s.i += 1
            X, y = features_for_chunk(ci, idxs, info, baseline, label_set)
            input_data(data=X, label=y[:, label_idx])
            return 1
    return _Iter()


def predict_test_xgb(booster, info, baseline, label_idx, chunks, label_set="full"):
    import xgboost as xgb
    probs = []; labels = []
    for ci, idxs in chunks:
        X, y = features_for_chunk(ci, idxs, info, baseline, label_set)
        dtest = xgb.DMatrix(X)
        probs.append(booster.predict(dtest))
        labels.append(y[:, label_idx])
    return np.concatenate(probs), np.concatenate(labels)


# --------------------- driver ---------------------

def eval_multilabel(probs_per_label, labels_per_label, label_list):
    per = {}; macro = {"auroc": [], "auprc": [], "f1": []}
    for c, name in enumerate(label_list):
        y = labels_per_label[c]; p = probs_per_label[c]
        pred = (p >= 0.5).astype(np.uint8)
        per[name] = {
            "auroc": float(roc_auc_score(y, p)),
            "auprc": float(average_precision_score(y, p)),
            "f1_at_0p5": float(f1_score(y, pred, zero_division=0)),
            "n_pos": int(y.sum()), "n_total": int(len(y)),
        }
        macro["auroc"].append(per[name]["auroc"])
        macro["auprc"].append(per[name]["auprc"])
        macro["f1"].append(per[name]["f1_at_0p5"])
    return {
        "per_label": per,
        "test_per_label": per,  # alias for aggregate.py
        "macro_auroc": float(np.mean(macro["auroc"])),
        "macro_auprc": float(np.mean(macro["auprc"])),
        "macro_f1": float(np.mean(macro["f1"])),
        "test_macro_auroc": float(np.mean(macro["auroc"])),
        "test_macro_auprc": float(np.mean(macro["auprc"])),
        "test_macro_f1": float(np.mean(macro["f1"])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True,
                    choices=["logistic-tf", "logistic-kmer", "xgboost"])
    ap.add_argument("--label-set", default="full", choices=["basic", "full"])
    ap.add_argument("--epochs", type=int, default=3,
                    help="passes over the data for SGDClassifier")
    args = ap.parse_args()

    info = load_info()
    labels = LABELS_FULL if args.label_set == "full" else ["is_enhancer", "is_silencer"]
    out_dir = os.path.join(OUT_ROOT, args.baseline)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[{args.baseline}] indexing chunks...")
    train_chunks = chunk_ids_for(info, TRAIN_CHROMS)
    test_chunks = chunk_ids_for(info, TEST_CHROMS)
    print(f"  {len(train_chunks)} train chunks, {len(test_chunks)} test chunks")

    if args.baseline.startswith("logistic"):
        print(f"\n[{args.baseline}] training all {len(labels)} labels jointly")
        clfs = train_sgd_all(info, args.baseline, len(labels),
                             train_chunks, args.label_set, epochs=args.epochs)
        test_probs, test_labels = predict_test_all(
            clfs, info, args.baseline, test_chunks, args.label_set)
    else:
        # XGBoost: per-label booster (each needs its own DataIter); 4× I/O
        # cost is acceptable since the QuantileDMatrix bin is built once
        # then training is on the compressed matrix.
        test_probs = []; test_labels = []
        for c, name in enumerate(labels):
            print(f"\n[{args.baseline}] training XGBoost for {name}")
            booster = train_xgb(info, args.baseline, c, train_chunks,
                                args.label_set)
            p, y = predict_test_xgb(booster, info, args.baseline, c,
                                    test_chunks, args.label_set)
            test_probs.append(p); test_labels.append(y)
            # Clean up xgb external-memory cache (can be tens of GB)
            import glob, shutil
            for d in glob.glob(os.path.join(out_dir,
                                            f"_xgb_cache_{c}*")):
                try: shutil.rmtree(d) if os.path.isdir(d) else os.remove(d)
                except Exception: pass

    metrics = eval_multilabel(test_probs, test_labels, labels)
    metrics["baseline"] = args.baseline
    metrics["label_set"] = args.label_set
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    np.savez(os.path.join(out_dir, "test_predictions.npz"),
             probs=np.stack(test_probs, axis=1).astype(np.float32),
             labels=np.stack(test_labels, axis=1).astype(np.uint8))
    print(f"\n[{args.baseline}] Macro AUROC={metrics['macro_auroc']:.4f}  "
          f"AUPRC={metrics['macro_auprc']:.4f}  F1={metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
