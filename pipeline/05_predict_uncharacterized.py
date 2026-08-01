#!/usr/bin/env python3
"""
05_predict_uncharacterized.py
==============================
Apply the trained model to uncharacterized regions (derived by 01_derive)
and predict their regulatory class with confidence scores.

For each sample group, reads the uncharacterized BED files, extracts sequences
and TF features, runs inference, and writes predictions.

Usage:
    python 05_predict_uncharacterized.py [--sample SAMPLE_NAME] [--threshold 0.8]
"""

import argparse
import csv
import glob
import importlib
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pysam
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_model_mod = importlib.import_module("03_model")
RegulatoryClassifier = _model_mod.RegulatoryClassifier
RegulomeDataset = _model_mod.RegulomeDataset
MODEL_REGISTRY = _model_mod.MODEL_REGISTRY

BASE_DIR = "/path/to/data"
GENOME_FASTA = "/path/to/resources/refdata-cellranger-arc-GRCh38-2024-A/fasta/genome.fa"
MODEL_DIR = os.path.join(BASE_DIR, "unbound_characetrize", "model_output")
TRAINING_DATA_DIR = os.path.join(BASE_DIR, "unbound_characetrize", "training_data")
OUTPUT_BASE = os.path.join(BASE_DIR, "unbound_characetrize", "predictions")
SEQ_LENGTH = 512


def one_hot_encode(seq, length=SEQ_LENGTH):
    mapping = {"A": [1,0,0,0], "C": [0,1,0,0], "G": [0,0,1,0], "T": [0,0,0,1]}
    encoded = np.zeros((length, 4), dtype=np.float32)
    for i, base in enumerate(seq[:length]):
        encoded[i] = mapping.get(base.upper(), [0.25, 0.25, 0.25, 0.25])
    return encoded


def extract_sequence(fasta, chrom, start, end, target_len=SEQ_LENGTH):
    region_len = end - start
    center = (start + end) // 2
    new_start = center - target_len // 2
    new_end = new_start + target_len
    chrom_len = fasta.get_reference_length(chrom)
    new_start = max(0, new_start)
    new_end = min(chrom_len, new_end)
    seq = fasta.fetch(chrom, new_start, new_end).upper()
    if len(seq) < target_len:
        seq = seq + "N" * (target_len - len(seq))
    return seq[:target_len]


def parse_uncharacterized_bed(bed_path):
    """Parse uncharacterized BED (same 17-col format as bound merged BED).

    Returns list of dicts with footprint + peak coords + TF info.
    """
    regions = []
    if not os.path.isfile(bed_path):
        return regions

    with open(bed_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 17:
                continue
            try:
                chrom = parts[0]
                fp_start = int(parts[1])
                fp_end = int(parts[2])
                tf_motif = parts[3]
                motif_score = float(parts[4])
                strand = parts[5]
                peak_start = int(parts[7])
                peak_end = int(parts[8])
                fp_score = float(parts[16])
            except (ValueError, IndexError):
                continue

            regions.append({
                "chrom": chrom,
                "fp_start": fp_start,
                "fp_end": fp_end,
                "tf_motif": tf_motif,
                "motif_score": motif_score,
                "strand": strand,
                "peak_start": peak_start,
                "peak_end": peak_end,
                "fp_score": fp_score,
            })

    return regions


def aggregate_by_peak(regions):
    """Group footprints by their ATAC peak and aggregate TF info."""
    peak_groups = defaultdict(list)
    for r in regions:
        key = (r["chrom"], r["peak_start"], r["peak_end"])
        peak_groups[key].append(r)

    aggregated = []
    for (chrom, ps, pe), footprints in peak_groups.items():
        tfs = set()
        motif_scores = []
        fp_scores = []
        for fp in footprints:
            tfs.add(fp["tf_motif"])
            motif_scores.append(fp["motif_score"])
            fp_scores.append(fp["fp_score"])

        aggregated.append({
            "chrom": chrom,
            "peak_start": ps,
            "peak_end": pe,
            "tfs": tfs,
            "max_motif_score": max(motif_scores),
            "max_fp_score": max(fp_scores),
            "mean_motif_score": np.mean(motif_scores),
            "mean_fp_score": np.mean(fp_scores),
            "n_tfs": len(tfs),
            "footprints": footprints,
        })

    return aggregated


def predict_batch(model, sequences, tf_features, tabular_features, device,
                  task="multiclass"):
    """Run inference on a batch.

    task='multiclass' → softmax probs over classes; argmax pred index.
    task='multilabel' → sigmoid probs per label; preds is the raw probs
                       (caller applies thresholds for the 3-bucket call).
    """
    with torch.no_grad():
        seq_t = torch.from_numpy(sequences).float().to(device)
        tf_t = torch.from_numpy(tf_features).float().to(device)
        tab_t = torch.from_numpy(tabular_features).float().to(device)

        logits = model(seq_t, tf_t, tab_t)
        if task == "multilabel":
            probs = torch.sigmoid(logits)
            return None, probs.cpu().numpy()
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)
    return preds.cpu().numpy(), probs.cpu().numpy()


def predict_batch_ensemble(models, sequences, tf_features, tabular_features,
                           device, task="multilabel"):
    """Run a (model, weight) list and return the weighted-average probs.
    For multilabel, probs are per-label sigmoid; weights are renormalized
    to sum to 1. Returns (None, probs) for multilabel, (preds, probs) for
    multiclass (preds = argmax of the averaged probs)."""
    total_w = sum(w for _, w in models) or 1.0
    agg = None
    with torch.no_grad():
        seq_t = torch.from_numpy(sequences).float().to(device)
        tf_t = torch.from_numpy(tf_features).float().to(device)
        tab_t = torch.from_numpy(tabular_features).float().to(device)
        for model, weight in models:
            logits = model(seq_t, tf_t, tab_t)
            probs = (torch.sigmoid(logits) if task == "multilabel"
                     else torch.softmax(logits, dim=1))
            contrib = (weight / total_w) * probs
            agg = contrib if agg is None else agg + contrib
    agg_np = agg.cpu().numpy()
    if task == "multilabel":
        return None, agg_np
    return agg_np.argmax(axis=1), agg_np


def categorize_multilabel(probs, t_enh, t_sil, label_list):
    """Map per-row sigmoid probs (n, 2) → bucket name in
    {"enhancer", "silencer", "enhancer_silencer", "neither"}.
    Assumes label_list order is [is_enhancer, is_silencer]."""
    p_enh = probs[:, label_list.index("is_enhancer")]
    p_sil = probs[:, label_list.index("is_silencer")]
    enh = p_enh >= t_enh
    sil = p_sil >= t_sil
    out = np.empty(len(probs), dtype=object)
    out[enh & sil] = "enhancer_silencer"
    out[enh & ~sil] = "enhancer"
    out[~enh & sil] = "silencer"
    out[~enh & ~sil] = "neither"
    return out


# ============================================================
# Parallel per-cell inference workers
# ============================================================
# Each worker process claims one GPU (round-robin via a shared queue),
# loads the ensemble/single model on it, opens its own pysam FASTA handle,
# and processes a shard of cells end-to-end (parse → featurize → infer →
# write CSV). Workers return only small category-count dicts, so there's no
# IPC of large feature arrays. CUDA is initialized in the worker (spawn
# context), never the parent.

_W = {}  # per-worker global state


def _build_models_for_device(device, use_ensemble, model_dir):
    """Load single best model or the weighted ensemble onto `device`.
    Note: `torch.backends.cudnn.enabled = False` is set at worker entry
    (see `_init_worker`) to work around a system-wide CUDNN_STATUS_
    NOT_INITIALIZED failure on this box's env. Fall-back path is slower
    but correct.
    Returns (models, label_list, task) where models is a list of
    (model, weight)."""
    def load_one(p):
        ck = torch.load(p, map_location=device, weights_only=False)
        mt = ck.get("model_type", "cnn")
        cls = MODEL_REGISTRY.get(mt, RegulatoryClassifier)
        m = cls(n_classes=ck["n_classes"], n_tfs=ck["n_tfs"],
                seq_length=ck["seq_length"]).to(device)
        m.load_state_dict(ck["model_state_dict"])
        m.eval()
        return m, ck
    models = []
    if use_ensemble:
        with open(os.path.join(model_dir, "ensemble", "training_results.json")) as f:
            ens = json.load(f)
        ref = None
        for name in ens["component_models"]:
            m, ck = load_one(os.path.join(model_dir, name, "best_model.pt"))
            models.append((m, float(ens["weights"][name])))
            ref = ck
        ckref = ref
    else:
        m, ckref = load_one(os.path.join(model_dir, "best_model.pt"))
        models.append((m, 1.0))
    return models, ckref["label_list"], ckref.get("task", "multiclass")


def _init_worker(gpu_queue, use_ensemble, model_dir, tf_vocab_path, fasta_path,
                 enh_thr, sil_thr, thr, batch_size):
    # Disable cuDNN — this box's env throws CUDNN_STATUS_NOT_INITIALIZED
    # on the first conv1d call even for a fresh tensor on an unused GPU.
    # Torch's native (non-cuDNN) conv path works fine; slightly slower.
    torch.backends.cudnn.enabled = False
    gpu_id = gpu_queue.get()
    dev = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    models, label_list, task = _build_models_for_device(dev, use_ensemble, model_dir)
    with open(tf_vocab_path) as f:
        tf_to_idx = json.load(f)["tf_to_idx"]
    fa = pysam.FastaFile(fasta_path)
    _W.update(
        device=dev, models=models, label_list=label_list, task=task,
        tf_to_idx=tf_to_idx, n_tfs=len(tf_to_idx), fasta=fa,
        valid_chroms=set(fa.references),
        standard_chroms={f"chr{i}" for i in range(1, 23)} | {"chrX", "chrY"},
        enh_thr=enh_thr, sil_thr=sil_thr, thr=thr, batch_size=batch_size,
    )


def _process_cell_worker(task):
    """task = (sample_out_dir, celltype, bed_path, out_csv).
    Returns a dict of category/class counts for this cell (empty if skipped)."""
    sample_out_dir, celltype, bed_path, out_csv = task
    if os.path.isfile(out_csv) and os.path.getsize(out_csv) > 0:
        return {}

    raw = parse_uncharacterized_bed(bed_path)
    if not raw:
        return {}
    peaks = aggregate_by_peak(raw)
    peaks = [p for p in peaks
             if p["chrom"] in _W["valid_chroms"] and p["chrom"] in _W["standard_chroms"]]
    if not peaks:
        return {}

    n = len(peaks)
    sequences = np.zeros((n, SEQ_LENGTH, 4), dtype=np.float32)
    tf_features = np.zeros((n, _W["n_tfs"]), dtype=np.float32)
    tabular_features = np.zeros((n, 5), dtype=np.float32)
    for i, peak in enumerate(peaks):
        seq = extract_sequence(_W["fasta"], peak["chrom"],
                               peak["peak_start"], peak["peak_end"])
        sequences[i] = one_hot_encode(seq)
        for tf in peak["tfs"]:
            j = _W["tf_to_idx"].get(tf)
            if j is not None:
                tf_features[i, j] = 1.0
        tabular_features[i, 0] = peak["max_motif_score"]
        tabular_features[i, 1] = peak["max_fp_score"]
        tabular_features[i, 2] = peak["mean_motif_score"]
        tabular_features[i, 3] = peak["mean_fp_score"]
        tabular_features[i, 4] = peak["n_tfs"]

    task_type = _W["task"]
    bs = _W["batch_size"]
    all_probs, all_preds = [], []
    for start in range(0, n, bs):
        end = min(start + bs, n)
        preds, probs = predict_batch_ensemble(
            _W["models"], sequences[start:end], tf_features[start:end],
            tabular_features[start:end], _W["device"], task=task_type,
        )
        all_probs.append(probs)
        if preds is not None:
            all_preds.append(preds)
    all_probs = np.concatenate(all_probs)
    if task_type == "multiclass":
        all_preds = np.concatenate(all_preds)

    label_list = _W["label_list"]
    local_stats = defaultdict(int)
    os.makedirs(os.path.join(sample_out_dir, celltype), exist_ok=True)
    tmp_csv = out_csv + ".tmp"
    with open(tmp_csv, "w", newline="") as f:
        writer = csv.writer(f)
        if task_type == "multilabel":
            writer.writerow(["chrom", "peak_start", "peak_end", "category",
                             "p_enhancer", "p_silencer", "n_tfs"])
            cats = categorize_multilabel(all_probs, _W["enh_thr"], _W["sil_thr"], label_list)
            enh_idx = label_list.index("is_enhancer")
            sil_idx = label_list.index("is_silencer")
            for i, peak in enumerate(peaks):
                cat = str(cats[i])
                writer.writerow([
                    peak["chrom"], peak["peak_start"], peak["peak_end"], cat,
                    f"{float(all_probs[i, enh_idx]):.4f}",
                    f"{float(all_probs[i, sil_idx]):.4f}",
                    peak["n_tfs"],
                ])
                local_stats[cat] += 1
        else:
            writer.writerow(["chrom", "peak_start", "peak_end", "predicted_class",
                             "confidence", "high_confidence", "n_tfs"]
                            + [f"prob_{l}" for l in label_list])
            for i, peak in enumerate(peaks):
                pred_class = label_list[all_preds[i]]
                confidence = float(all_probs[i, all_preds[i]])
                high_conf = confidence >= _W["thr"]
                writer.writerow([
                    peak["chrom"], peak["peak_start"], peak["peak_end"],
                    pred_class, f"{confidence:.4f}", "YES" if high_conf else "NO",
                    peak["n_tfs"],
                ] + [f"{p:.4f}" for p in all_probs[i]])
                local_stats[pred_class] += 1
                if high_conf:
                    local_stats[f"{pred_class}_high_conf"] += 1
    os.replace(tmp_csv, out_csv)  # atomic
    return dict(local_stats)


def main():
    parser = argparse.ArgumentParser(description="Predict regulatory class for uncharacterized regions")
    parser.add_argument("--sample", type=str, default=None,
                        help="Process only this sample group. Default: all.")
    parser.add_argument("--threshold", type=float, default=0.8,
                        help="(multiclass) Confidence threshold for high-confidence "
                             "predictions.")
    parser.add_argument("--enhancer-threshold", type=float, default=0.5,
                        help="(multilabel) Sigmoid threshold for enhancer call.")
    parser.add_argument("--silencer-threshold", type=float, default=0.5,
                        help="(multilabel) Sigmoid threshold for silencer call.")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-workers", type=int, default=10)
    parser.add_argument("--workers-per-gpu", type=int, default=2,
                        help="Parallel inference workers per GPU. Total "
                             "workers = n_gpus × this. Each worker loads the "
                             "model(s) on its GPU and processes a shard of "
                             "cells end-to-end (parse+featurize+infer+write).")
    parser.add_argument("--ensemble", action="store_true",
                        help="Use the CNN+Hybrid weighted ensemble for "
                             "inference (loads each component model and "
                             "averages their sigmoid outputs by the weights "
                             "saved in model_output/ensemble/training_results.json). "
                             "Without this flag, uses the single best_model.pt.")
    parser.add_argument("--model-dir", type=str, default=None,
                        help="Override the trained-model directory. Default "
                             "is `model_output` (v1). Pass `model_output_v2` "
                             "for the v2 retrain.")
    parser.add_argument("--output-base", type=str, default=None,
                        help="Override the predictions output base dir. "
                             "Default is `predictions`. Useful when running "
                             "multiple model variants side-by-side.")
    args = parser.parse_args()
    global MODEL_DIR, OUTPUT_BASE
    if args.model_dir:
        MODEL_DIR = (args.model_dir if os.path.isabs(args.model_dir)
                     else os.path.join(BASE_DIR, "unbound_characetrize", args.model_dir))
        print(f"[override] MODEL_DIR = {MODEL_DIR}")
    if args.output_base:
        OUTPUT_BASE = (args.output_base if os.path.isabs(args.output_base)
                       else os.path.join(BASE_DIR, "unbound_characetrize", args.output_base))
        print(f"[override] OUTPUT_BASE = {OUTPUT_BASE}")

    os.makedirs(OUTPUT_BASE, exist_ok=True)

    # Lightweight metadata read (CPU only — workers load full models on GPUs).
    # We just need task + label_list here for the final summary.
    ckpt_path = os.path.join(MODEL_DIR, "best_model.pt")
    if not os.path.isfile(ckpt_path):
        print(f"ERROR: Model checkpoint not found: {ckpt_path}")
        sys.exit(1)
    if args.ensemble:
        ens_results = os.path.join(MODEL_DIR, "ensemble", "training_results.json")
        if not os.path.isfile(ens_results):
            print(f"ERROR: --ensemble requested but {ens_results} not found.")
            sys.exit(1)
    _meta = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    label_list = _meta["label_list"]
    task = _meta.get("task", "multiclass")
    del _meta
    tf_vocab_path = os.path.join(TRAINING_DATA_DIR, "tf_vocabulary.json")
    print(f"Inference task: {task}. Classes: {label_list}. "
          f"Ensemble: {bool(args.ensemble)}")
    if task == "multilabel":
        print(f"Multilabel thresholds: enhancer={args.enhancer_threshold}, "
              f"silencer={args.silencer_threshold}")

    # Discover sample groups
    if args.sample:
        samples = [args.sample]
    else:
        pattern = os.path.join(BASE_DIR, "*_sorted_scBAMs_UNKNOWN_TFBS")
        samples = []
        for d in sorted(glob.glob(pattern)):
            if os.path.isdir(d):
                samples.append(os.path.basename(d).replace("_sorted_scBAMs_UNKNOWN_TFBS", ""))

    print(f"\nSample groups to predict: {samples}")

    global_stats = defaultdict(int)

    # ---- Build the full task list across all samples ----
    tasks = []
    for sample in samples:
        unknown_dir = os.path.join(BASE_DIR, f"{sample}_sorted_scBAMs_UNKNOWN_TFBS")
        if not os.path.isdir(unknown_dir):
            print(f"  {sample}: No UNKNOWN_TFBS directory found, skipping.")
            continue
        sample_out_dir = os.path.join(OUTPUT_BASE, sample)
        os.makedirs(sample_out_dir, exist_ok=True)
        for celltype in sorted(os.listdir(unknown_dir)):
            ct_dir = os.path.join(unknown_dir, celltype)
            if not os.path.isdir(ct_dir):
                continue
            for bed_file in sorted(os.listdir(ct_dir)):
                if bed_file.endswith("_uncharacterized.bed"):
                    bed_path = os.path.join(ct_dir, bed_file)
                    stem = bed_file.replace("_uncharacterized.bed", "")
                    out_csv = os.path.join(sample_out_dir, celltype,
                                           f"{stem}_predictions.csv")
                    tasks.append((sample_out_dir, celltype, bed_path, out_csv))

    n_cells_total = len(tasks)
    n_already = sum(1 for t in tasks
                    if os.path.isfile(t[3]) and os.path.getsize(t[3]) > 0)
    print(f"\nTotal cells: {n_cells_total:,} ({n_already:,} already done, "
          f"{n_cells_total - n_already:,} to predict)")

    # ---- Parallel dispatch: spawn-context pool, each worker pinned to a GPU ----
    import multiprocessing as mp
    n_gpus = max(torch.cuda.device_count(), 1)
    workers_per_gpu = max(1, args.workers_per_gpu)
    n_workers = n_gpus * workers_per_gpu
    print(f"Dispatching across {n_workers} workers "
          f"({n_gpus} GPUs × {workers_per_gpu} workers/GPU)")

    ctx = mp.get_context("spawn")  # spawn: safe CUDA init in children
    gpu_q = ctx.Manager().Queue()
    for w in range(n_workers):
        gpu_q.put(w % n_gpus)

    init_args = (
        gpu_q, bool(args.ensemble), MODEL_DIR, tf_vocab_path, GENOME_FASTA,
        args.enhancer_threshold, args.silencer_threshold, args.threshold,
        args.batch_size,
    )
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx,
                             initializer=_init_worker, initargs=init_args) as ex:
        for stats in ex.map(_process_cell_worker, tasks, chunksize=4):
            for k, v in stats.items():
                global_stats[k] += v
            done += 1
            if done % 2000 == 0:
                elapsed = time.time() - t0
                rate = done / max(elapsed, 1e-6)
                eta = (n_cells_total - done) / max(rate, 1e-6)
                print(f"  {done}/{n_cells_total} cells "
                      f"({rate*60:.0f} cells/min, ETA {eta/3600:.1f} h)")
    print(f"  All {n_cells_total:,} cells processed in {(time.time()-t0)/60:.1f} min.")

    # Summary
    print(f"\n{'='*60}")
    print("Global Prediction Summary")
    print(f"{'='*60}")
    if task == "multilabel":
        for cat in ("enhancer", "silencer", "enhancer_silencer", "neither"):
            print(f"  {cat}: {global_stats.get(cat, 0):,}")
    else:
        for label in label_list:
            total = global_stats.get(label, 0)
            high = global_stats.get(f"{label}_high_conf", 0)
            print(f"  {label}: {total} total, {high} high-confidence (>={args.threshold})")

    # Save summary
    summary_path = os.path.join(OUTPUT_BASE, "prediction_summary.json")
    with open(summary_path, "w") as f:
        json.dump(dict(global_stats), f, indent=2)
    print(f"\nSummary saved to {summary_path}")
    print("Done.")


if __name__ == "__main__":
    main()
