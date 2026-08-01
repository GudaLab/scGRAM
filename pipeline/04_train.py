#!/usr/bin/env python3
"""
04_train.py
============
Train multiple model architectures for regulatory region classification:
  1. 1D CNN (Basset-style) — local motif detection
  2. Transformer — long-range sequence dependencies
  3. Hybrid CNN-Transformer — local + global
  4. XGBoost — gradient boosted trees on engineered features

Each DL model is trained with:
  - Multi-GPU DistributedDataParallel (DDP)
  - Focal Loss for class imbalance
  - Chromosome-based train/val/test split
  - Reverse complement augmentation
  - Cosine annealing + early stopping

After all models are trained, they are compared on the test set.
The best model is selected and an optional ensemble is created.

Usage:
    # Single GPU (for debugging):
    python 04_train.py --data-dir /path/to/training_data

    # Multi-GPU via torchrun:
    torchrun --nproc_per_node=4 04_train.py --data-dir /path/to/training_data

    # Train only one model type:
    python 04_train.py --model-type cnn
"""

import argparse
import importlib
import json
import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import BatchSampler, DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
import torch.nn as _nn  # avoid clobber of nn name in legacy code paths

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_model_mod = importlib.import_module("03_model")
RegulatoryClassifier = _model_mod.RegulatoryClassifier
TransformerClassifier = _model_mod.TransformerClassifier
HybridCNNTransformer = _model_mod.HybridCNNTransformer
RegulomeDataset = _model_mod.RegulomeDataset
FocalLoss = _model_mod.FocalLoss
MODEL_REGISTRY = _model_mod.MODEL_REGISTRY


# ---- Chromosome split ----

TRAIN_CHROMS = {f"chr{i}" for i in range(1, 18)}
VAL_CHROMS = {f"chr{i}" for i in range(18, 21)}
TEST_CHROMS = {"chr21", "chr22", "chrX"}


# ---- Label schemes ----
# The raw dataset has 7 classes (see dataset_info.json label_list). Some have
# too few samples to train as distinct targets (TFBS: 882; cis_regulatory:
# 21,739). `5class` remaps the rarest two into other_regulatory at read time;
# no rows are dropped. `7class` keeps the original targets.
#
# Each scheme specifies:
#   label_list : new class names (in new index order)
#   remap      : 7-wide array mapping original_idx -> new_idx
# The original index order is:
#   0:TFBS 1:cis_regulatory 2:enhancer 3:genic 4:other_regulatory
#   5:promoter 6:silencer
ORIGINAL_LABELS = ["TFBS", "cis_regulatory", "enhancer", "genic",
                   "other_regulatory", "promoter", "silencer"]

LABEL_SCHEMES = {
    "7class": {
        "label_list": ORIGINAL_LABELS,
        # identity
        "remap": np.array([0, 1, 2, 3, 4, 5, 6], dtype=np.int64),
    },
    "5class": {
        # enhancer, genic, other_regulatory, promoter, silencer
        # (merged: TFBS + cis_regulatory -> other_regulatory)
        "label_list": ["enhancer", "genic", "other_regulatory", "promoter", "silencer"],
        # original idx -> new idx
        # TFBS(0), cis_regulatory(1)      -> other_regulatory (new idx 2)
        # enhancer(2)                     -> 0
        # genic(3)                        -> 1
        # other_regulatory(4)             -> 2
        # promoter(5)                     -> 3
        # silencer(6)                     -> 4
        "remap": np.array([2, 2, 0, 1, 2, 3, 4], dtype=np.int64),
    },
}


class DistributedWeightedRandomSampler(Sampler):
    """Class-balanced random sampler for DDP. Two-step draw: (1) pick class
    uniformly over non-empty classes, (2) pick a row uniformly within that
    class. Mathematically equivalent to per-sample weighted sampling with
    weight_i = 1/count_{label(i)}, but works at any scale — torch.multinomial
    is only invoked on the n_classes-sized class-weight vector (≤ 2^24 limit
    never matters).

    Net effect: each class contributes ~equally to each batch, so rare classes
    (TFBS / cis_regulatory-merged-in / other_regulatory, etc.) are oversampled
    without dropping majority rows.

    Each rank gets num_samples // world_size draws per epoch. Seeds are
    derived from (base_seed, epoch, rank) for reproducibility + disjoint
    per-rank streams + per-epoch reshuffle via set_epoch().
    """

    def __init__(self, labels, num_samples, world_size=1, rank=0,
                 num_classes=None, base_seed=0):
        labels_np = np.asarray(labels, dtype=np.int64)
        if num_classes is None:
            num_classes = int(labels_np.max()) + 1
        self.num_classes = int(num_classes)

        counts = np.bincount(labels_np, minlength=num_classes).astype(np.int64)
        self.counts = counts

        # Per-class index pools. Each tensor holds the dataset-global indices
        # of every row with that label.
        self._class_indices = []
        for c in range(num_classes):
            idxs = np.nonzero(labels_np == c)[0].astype(np.int64)
            self._class_indices.append(torch.from_numpy(idxs))

        # Uniform over non-empty classes: P(class=c) = 1 / (# non-empty classes).
        # This yields a 2-step P(sample i) = 1/(n_classes × count_{label(i)}),
        # which is the standard class-balanced oversampling target.
        class_w = np.ones(num_classes, dtype=np.float64)
        class_w[counts == 0] = 0.0
        if class_w.sum() <= 0:
            raise ValueError("No non-empty classes in labels.")
        class_w = class_w / class_w.sum()
        self.class_weights = torch.as_tensor(class_w, dtype=torch.double)

        self.total = int(num_samples)
        self.world_size = int(world_size)
        self.rank = int(rank)
        self.base_seed = int(base_seed)
        self.epoch = 0
        self.per_rank = max(1, self.total // self.world_size)

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self):
        return self.per_rank

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.base_seed * 1_000_003 + self.epoch * 101 + self.rank)

        # Step 1: draw class labels (categorical of size num_classes — well
        # under the 2^24 multinomial limit).
        class_draws = torch.multinomial(
            self.class_weights, self.per_rank,
            replacement=True, generator=g,
        )

        # Step 2: for each class, fill in uniform random pool indices.
        out = torch.empty(self.per_rank, dtype=torch.int64)
        for c in range(self.num_classes):
            mask = class_draws == c
            n = int(mask.sum().item())
            if n == 0:
                continue
            pool = self._class_indices[c]
            if pool.numel() == 0:
                # Empty class: fall back to any valid index to avoid a crash.
                out[mask] = 0
                continue
            rand_positions = torch.randint(
                0, pool.numel(), (n,), generator=g, dtype=torch.int64,
            )
            out[mask] = pool[rand_positions]
        return iter(out.tolist())


class ChunkSortedBatchSampler(BatchSampler):
    """Wraps a Sampler into batches, then sorts indices within each batch by
    chunk id so that mmap reads within a batch hit the same files sequentially.
    This is a pure I/O optimization: downstream SGD / BatchNorm / gradient math
    are invariant to in-batch order, so training dynamics and convergence are
    unchanged. Only the on-disk access pattern changes from random to
    sequential-per-chunk.

    Args:
        sampler: any Sampler producing dataset-global indices.
        batch_size: batch size per rank.
        drop_last: drop the final partial batch if True.
        chunk_of_index: 1-D array-like mapping global index -> chunk id.
    """

    def __init__(self, sampler, batch_size, drop_last, chunk_of_index,
                 start_batch=0):
        super().__init__(sampler, batch_size, drop_last)
        # np.asarray so that indexing returns a numpy array for sort.argsort
        self.chunk_of_index = np.asarray(chunk_of_index)
        # Mutable: can be updated between epochs to skip the first N batches
        # on resume from a mid-epoch checkpoint. Reset to 0 after the first
        # epoch on resume.
        self.start_batch = int(start_batch)

    def __iter__(self):
        for batch_idx, batch in enumerate(super().__iter__()):
            if batch_idx < self.start_batch:
                continue
            # Sort by chunk id; np.argsort is ~6 µs for 512 ints.
            batch_arr = np.asarray(batch, dtype=np.int64)
            order = np.argsort(self.chunk_of_index[batch_arr], kind="stable")
            yield batch_arr[order].tolist()

    def __len__(self):
        return max(0, super().__len__() - self.start_batch)


def setup_ddp():
    if "RANK" in os.environ:
        import datetime
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        # Default NCCL watchdog timeout is 10 min — far too short if any
        # single rank stalls on I/O (e.g. val eval on cold NFS). Bump to
        # 2 h so slow legitimate work doesn't tank the run. Override via
        # TORCH_NCCL_WATCHDOG_TIMEOUT_SEC env var.
        timeout_sec = int(os.environ.get("TORCH_NCCL_WATCHDOG_TIMEOUT_SEC", 7200))
        dist.init_process_group(
            "nccl", rank=rank, world_size=world_size,
            timeout=datetime.timedelta(seconds=timeout_sec),
        )
        torch.cuda.set_device(local_rank)
        if rank == 0:
            print(f"DDP initialized: world_size={world_size}, "
                  f"NCCL timeout={timeout_sec}s")
        return rank, local_rank, world_size
    return 0, 0, 1


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main(rank):
    return rank == 0


def compute_class_weights(labels, n_classes):
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    weights = 1.0 / counts
    weights = weights / weights.sum() * n_classes
    return weights.tolist()


class SmoothBCEWithLogitsLoss(torch.nn.Module):
    """BCEWithLogitsLoss with target smoothing.

    targets become y · (1 - 2ε) + ε,   ε ∈ [0, 0.5)

    Reduces the gradient at confidently-correct predictions, which prevents
    the probability head from drifting to ±∞ once ranking has saturated.
    This is what was making our val loss rise after epoch ~3 while AUROC
    plateaued — the model kept becoming MORE confident about predictions
    it had already ranked correctly, blowing up BCE on rare classes.
    """
    def __init__(self, pos_weight=None, smoothing=0.0):
        super().__init__()
        if not 0.0 <= smoothing < 0.5:
            raise ValueError(f"smoothing must be in [0, 0.5); got {smoothing}")
        self.smoothing = float(smoothing)
        self.bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits, targets):
        if self.smoothing > 0.0:
            targets = targets * (1.0 - 2.0 * self.smoothing) + self.smoothing
        return self.bce(logits, targets)


def evaluate(model, dataloader, criterion, device, n_classes, label_list,
             world_size=1, rank=0, amp_dtype=None):
    """Evaluate a model on a dataloader.

    DDP-aware: when world_size > 1, ALL ranks must call this function with
    their shard of the dataloader (DistributedSampler-wrapped). Each rank
    processes its shard in parallel; predictions/labels/probs are gathered
    on every rank and metrics are computed from the aggregate.

    Previous design ran eval only on rank 0 over a DistributedSampler shard,
    which (a) processed just 1/world_size of the data and (b) left ranks 1..N
    blocking on the next collective op for however long rank 0's eval took —
    triggering NCCL's 10-minute watchdog timeout on large datasets.
    """
    model.eval()
    local_preds, local_labels, local_probs = [], [], []
    total_loss, n_batches = 0.0, 0

    with torch.no_grad():
        for seq, tf_feat, tab_feat, labels in dataloader:
            seq = seq.to(device)
            tf_feat = tf_feat.to(device)
            tab_feat = tab_feat.to(device)
            labels = labels.to(device)

            with _autocast_ctx(amp_dtype):
                logits = model(seq, tf_feat, tab_feat)
                loss = criterion(logits, labels)
            # Cast back to fp32 for downstream metric stability
            logits = logits.float()
            total_loss += loss.item()
            n_batches += 1

            probs = torch.softmax(logits, dim=1)
            local_preds.append(logits.argmax(dim=1).cpu().numpy())
            local_labels.append(labels.cpu().numpy())
            local_probs.append(probs.cpu().numpy())

    local_preds = (np.concatenate(local_preds)
                   if local_preds else np.array([], dtype=np.int64))
    local_labels = (np.concatenate(local_labels)
                    if local_labels else np.array([], dtype=np.int64))
    local_probs = (np.concatenate(local_probs, axis=0)
                   if local_probs else np.empty((0, n_classes), dtype=np.float32))

    if world_size > 1:
        # Gather per-rank arrays onto every rank, then concatenate.
        # all_gather_object is simpler than tensor all_gather (no padding
        # needed for variable shard sizes) and the cost is acceptable — for
        # a 38M-row val set this is ~1 GB of pickled data across the ring.
        gathered_preds = [None] * world_size
        gathered_labels = [None] * world_size
        gathered_probs = [None] * world_size
        dist.all_gather_object(gathered_preds, local_preds)
        dist.all_gather_object(gathered_labels, local_labels)
        dist.all_gather_object(gathered_probs, local_probs)
        # Reduce loss counters
        loss_t = torch.tensor([total_loss, float(n_batches)], device=device)
        dist.all_reduce(loss_t, op=dist.ReduceOp.SUM)
        total_loss = float(loss_t[0].item())
        n_batches = int(loss_t[1].item())
        all_preds = np.concatenate(gathered_preds)
        all_labels = np.concatenate(gathered_labels)
        all_probs = np.concatenate(gathered_probs, axis=0)
    else:
        all_preds = local_preds
        all_labels = local_labels
        all_probs = local_probs

    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    try:
        auroc = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")
    except ValueError:
        auroc = 0.0

    report = classification_report(
        all_labels, all_preds, target_names=label_list, zero_division=0, output_dict=True,
    )
    cm = confusion_matrix(all_labels, all_preds)

    return {
        "loss": total_loss / max(n_batches, 1),
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "auroc": auroc,
        "report": report,
        "confusion_matrix": cm,
        "predictions": all_preds,
        "labels": all_labels,
        "probabilities": all_probs,
    }


def evaluate_multilabel(model, dataloader, criterion, device, n_labels,
                        label_list, world_size=1, rank=0, threshold=0.5,
                        amp_dtype=None):
    """Distributed eval for the multilabel binary task. Each rank processes
    its DataLoader shard; logits and labels are all_gathered onto every
    rank; per-label AUROC / AUPRC / F1@threshold are computed from the
    aggregate. Loss is summed via all_reduce."""
    model.eval()
    local_logits, local_labels = [], []
    total_loss, n_batches = 0.0, 0

    with torch.no_grad():
        for seq, tf_feat, tab_feat, labels in dataloader:
            seq = seq.to(device)
            tf_feat = tf_feat.to(device)
            tab_feat = tab_feat.to(device)
            labels = labels.to(device)
            with _autocast_ctx(amp_dtype):
                logits = model(seq, tf_feat, tab_feat)  # (B, n_labels)
                loss = criterion(logits, labels.float())
            # Cast back to fp32 for metric numerical stability
            logits = logits.float()
            total_loss += loss.item()
            n_batches += 1
            local_logits.append(logits.detach().cpu().numpy())
            local_labels.append(labels.detach().cpu().numpy())

    if local_logits:
        local_logits = np.concatenate(local_logits, axis=0)
        local_labels = np.concatenate(local_labels, axis=0)
    else:
        local_logits = np.empty((0, n_labels), dtype=np.float32)
        local_labels = np.empty((0, n_labels), dtype=np.float32)

    if world_size > 1:
        gathered_logits = [None] * world_size
        gathered_labels = [None] * world_size
        dist.all_gather_object(gathered_logits, local_logits)
        dist.all_gather_object(gathered_labels, local_labels)
        loss_t = torch.tensor([total_loss, float(n_batches)], device=device)
        dist.all_reduce(loss_t, op=dist.ReduceOp.SUM)
        total_loss = float(loss_t[0].item())
        n_batches = int(loss_t[1].item())
        all_logits = np.concatenate(gathered_logits, axis=0)
        all_labels = np.concatenate(gathered_labels, axis=0)
    else:
        all_logits = local_logits
        all_labels = local_labels

    # Sigmoid via numerically-stable formulation: 1 / (1 + exp(-x))
    all_probs = 1.0 / (1.0 + np.exp(-all_logits.astype(np.float64)))
    all_preds = (all_probs >= threshold).astype(np.int64)
    int_labels = all_labels.astype(np.int64)

    per_label = {}
    aurocs, auprcs, f1s = [], [], []
    for c, name in enumerate(label_list):
        y_true = int_labels[:, c]
        y_score = all_probs[:, c]
        y_pred = all_preds[:, c]
        try:
            auroc = float(roc_auc_score(y_true, y_score)) if y_true.sum() > 0 else 0.0
        except ValueError:
            auroc = 0.0
        try:
            auprc = float(average_precision_score(y_true, y_score)) if y_true.sum() > 0 else 0.0
        except ValueError:
            auprc = 0.0
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        per_label[name] = {
            "auroc": auroc, "auprc": auprc, "f1_at_0p5": f1,
            "n_pos": int(y_true.sum()), "n_total": int(len(y_true)),
        }
        aurocs.append(auroc); auprcs.append(auprc); f1s.append(f1)

    return {
        "loss": total_loss / max(n_batches, 1),
        "macro_auroc": float(np.mean(aurocs)),
        "macro_auprc": float(np.mean(auprcs)),
        "macro_f1": float(np.mean(f1s)),
        "per_label": per_label,
        "predictions": all_preds,
        "labels": int_labels,
        "probabilities": all_probs.astype(np.float32),
    }


_AMP_DTYPE_MAP = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}


def _autocast_ctx(amp_dtype):
    """Return a torch.autocast context manager for cuda forward passes, or
    a no-op contextmanager when amp_dtype is None (fp32)."""
    if amp_dtype is None:
        from contextlib import nullcontext
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=amp_dtype)


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch,
                    start_batch_idx=0, save_fn=None, save_every=500,
                    total_batches=None, amp_dtype=None,
                    scheduler=None, max_grad_norm=1.0,
                    scheduler_steps_per_batch=False):
    """Iterate one epoch of training.

    Args:
        start_batch_idx: index of the first GLOBAL batch this call processes
            (the BatchSampler is expected to have already skipped batches
            0..start_batch_idx-1 of this epoch, so the dataloader's first
            yielded batch corresponds to global batch start_batch_idx).
        save_fn: optional callable taking (epoch, next_batch_idx). Called
            every `save_every` batches. Should atomically write last_ckpt.pt
            with the current state.
        save_every: how many GLOBAL batches between saves.
        total_batches: total number of batches in a fresh epoch (for the
            progress print's "X/Y" denominator). Falls back to
            `len(dataloader) + start_batch_idx` if None.
    """
    model.train()
    total_loss, n_batches = 0.0, 0

    if total_batches is None:
        total_batches = len(dataloader) + start_batch_idx

    for local_idx, (seq, tf_feat, tab_feat, labels) in enumerate(dataloader):
        global_batch_idx = local_idx + start_batch_idx
        seq = seq.to(device)
        tf_feat = tf_feat.to(device)
        tab_feat = tab_feat.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        # bf16 autocast wraps both forward + loss for tensor-core acceleration.
        # bf16 does not require GradScaler (range is fp32-compatible); fp16
        # would. We default to bf16 for stability + speed on A5000/A6000.
        with _autocast_ctx(amp_dtype):
            logits = model(seq, tf_feat, tab_feat)
            loss = criterion(logits, labels)
        loss.backward()
        if max_grad_norm and max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           max_norm=max_grad_norm)
        optimizer.step()
        # Per-batch scheduler step — used with LinearLR warmup → cosine.
        # The per-epoch scheduler.step() in the outer loop is skipped when
        # this flag is True so the schedule advances exactly once per batch.
        if scheduler_steps_per_batch and scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        n_batches += 1

        # Print progress every 100 global batches
        if global_batch_idx > 0 and global_batch_idx % 100 == 0:
            print(f"    Epoch {epoch} | Batch {global_batch_idx}/{total_batches} "
                  f"| Loss: {total_loss/n_batches:.4f}")

        # Per-N-batches checkpoint save (atomic .tmp + rename inside save_fn).
        # We save AFTER the optimizer step so the saved state reflects the
        # state to resume FROM at the next batch.
        if (save_fn is not None and save_every > 0
                and (global_batch_idx + 1) % save_every == 0):
            save_fn(epoch, global_batch_idx + 1)

    return total_loss / max(n_batches, 1)


def _broadcast_skip(skip_local, world_size, device):
    """Broadcast a per-rank skip decision from rank 0 so all ranks return
    together. Without this, a rank-0 early return would deadlock DDP."""
    if world_size <= 1:
        return skip_local
    skip_t = torch.tensor([1 if skip_local else 0], device=device,
                          dtype=torch.int32)
    dist.broadcast(skip_t, src=0)
    return bool(skip_t.item())


def _load_cached_test_metrics(model_out_dir):
    """Reconstruct test_metrics from on-disk artifacts of a finished model."""
    with open(os.path.join(model_out_dir, "training_results.json")) as f:
        cached = json.load(f)
    preds = np.load(os.path.join(model_out_dir, "test_predictions.npz"))
    return {
        "loss": 0.0,
        "macro_f1": cached["test_macro_f1"],
        "weighted_f1": cached["test_weighted_f1"],
        "auroc": cached["test_auroc"],
        "report": cached["test_report"],
        "confusion_matrix": np.array(cached["confusion_matrix"]),
        "predictions": preds["predictions"],
        "labels": preds["labels"],
        "probabilities": preds["probabilities"],
    }


def train_dl_model(model_type, model_cls, train_dataset, val_dataset, test_dataset,
                   n_classes, n_tfs, seq_length, label_list, class_weights,
                   args, rank, local_rank, world_size, device, task="multiclass"):
    """Train a single DL model and return test metrics.

    Crash recovery:
      - Per-model marker `<model_out_dir>/done.marker` — if present, this model
        has already completed test eval; skip training entirely and return
        cached metrics. Lets a re-run resume after CNN+Transformer done but
        Hybrid mid-training, without re-touching CNN/Transformer.
      - Per-epoch `last_ckpt.pt` — written atomically at the end of every
        epoch with model + optimizer + scheduler + epoch + history + best F1
        + patience counter. On re-run, training resumes from epoch N+1 with
        the full optimizer momentum / LR schedule state intact.
    """
    model_out_dir = os.path.join(args.output_dir, model_type)
    done_marker = os.path.join(model_out_dir, "done.marker")
    last_ckpt_path = os.path.join(model_out_dir, "last_ckpt.pt")

    # ---- Skip if model is already complete ----
    skip = False
    if is_main(rank):
        if os.path.isfile(done_marker):
            print(f"\n  [{model_type}] done.marker present — skipping training, "
                  f"loading cached test metrics.")
            skip = True
    skip = _broadcast_skip(skip, world_size, device)
    if skip:
        if is_main(rank):
            return _load_cached_test_metrics(model_out_dir)
        return None

    if is_main(rank):
        os.makedirs(model_out_dir, exist_ok=True)
        print(f"\n{'='*70}")
        print(f"  TRAINING MODEL: {model_type.upper()}")
        print(f"{'='*70}")

    # Build model
    model = model_cls(n_classes=n_classes, n_tfs=n_tfs, seq_length=seq_length).to(device)

    # ---- Resume model weights if last_ckpt.pt present (BEFORE wrapping in DDP) ----
    # Skip resume if the saved checkpoint was for a different task — the
    # output head dimension won't match (multiclass=n_classes, multilabel=2).
    resume_state = None
    if os.path.isfile(last_ckpt_path):
        try:
            candidate = torch.load(last_ckpt_path, map_location=device,
                                   weights_only=False)
            saved_task = candidate.get("task", "multiclass")
            if saved_task != task:
                if is_main(rank):
                    print(f"  [resume] last_ckpt.pt was for task='{saved_task}' "
                          f"but current task='{task}'; ignoring (training from scratch).")
            else:
                model.load_state_dict(candidate["model_state_dict"])
                resume_state = candidate
                if is_main(rank):
                    print(f"  [resume] loaded weights from last_ckpt.pt "
                          f"(was at epoch {resume_state['epoch']}, "
                          f"task='{saved_task}')")
        except Exception as e:
            if is_main(rank):
                print(f"  [resume] last_ckpt.pt unreadable ({e}); starting fresh")
            resume_state = None

    if is_main(rank):
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Parameters: {n_params:,}")

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    # Data loaders
    # Training uses a WeightedRandomSampler (DDP-aware) to oversample rare
    # classes without dropping any rows. num_samples per epoch is capped at
    # min(dataset_size, args.samples_per_epoch) so extreme imbalance does
    # not force astronomically long epochs.
    multilabel = (task == "multilabel")
    if multilabel:
        # Multilabel: imbalance is handled by BCE pos_weight, not by a
        # weighted sampler (multi-label "class" balancing is ill-defined
        # since one row can be positive for >1 label).
        train_sampler = (
            DistributedSampler(train_dataset, num_replicas=world_size,
                               rank=rank, shuffle=True)
            if world_size > 1 else None
        )
    else:
        n_classes_train = int(train_dataset.labels.max()) + 1
        samples_per_epoch = args.samples_per_epoch
        if samples_per_epoch <= 0:
            samples_per_epoch = len(train_dataset)
        # Ensure minimum coverage: at least 1 effective pass over minority classes
        samples_per_epoch = min(samples_per_epoch, len(train_dataset))
        train_sampler = DistributedWeightedRandomSampler(
            labels=train_dataset.labels,
            num_samples=samples_per_epoch,
            world_size=world_size,
            rank=rank,
            num_classes=n_classes_train,
            base_seed=args.seed,
        )
    # Wrap the sampler so that within each batch the indices are chunk-sorted —
    # converts random-access mmap reads into sequential reads per chunk, which
    # is a big win over NFS. Mathematically a no-op (same samples, same batches;
    # only in-batch order changes, and SGD/Adam/BN are order-invariant).
    train_batch_sampler = ChunkSortedBatchSampler(
        sampler=(train_sampler if train_sampler is not None
                 else list(range(len(train_dataset)))),
        batch_size=args.batch_size,
        drop_last=True,
        chunk_of_index=train_dataset.index[:, 0],
    )
    # Val / test: use DistributedSampler so each rank covers 1/world_size of
    # the data, and wrap with ChunkSortedBatchSampler for the same sequential
    # NFS read pattern we use on train. Without chunk-sort, val/test reads
    # were random-access on cold pages — rank 0's val pass on the 38M-row
    # set took hours and blew through NCCL's watchdog timeout.
    val_sampler = (
        DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        if world_size > 1 else None
    )
    val_batch_sampler = ChunkSortedBatchSampler(
        sampler=(val_sampler if val_sampler is not None
                 else list(range(len(val_dataset)))),
        batch_size=args.batch_size,
        drop_last=False,
        chunk_of_index=val_dataset.index[:, 0],
    )
    test_sampler = (
        DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        if world_size > 1 else None
    )
    test_batch_sampler = ChunkSortedBatchSampler(
        sampler=(test_sampler if test_sampler is not None
                 else list(range(len(test_dataset)))),
        batch_size=args.batch_size,
        drop_last=False,
        chunk_of_index=test_dataset.index[:, 0],
    )

    # batch_sampler is mutually exclusive with batch_size/shuffle/sampler/drop_last
    # in the DataLoader constructor, so pass only batch_sampler here.
    train_loader = DataLoader(train_dataset, batch_sampler=train_batch_sampler,
                              num_workers=args.num_workers,
                              pin_memory=True,
                              persistent_workers=(args.num_workers > 0))
    val_loader = DataLoader(val_dataset, batch_sampler=val_batch_sampler,
                            num_workers=args.num_workers, pin_memory=True,
                            persistent_workers=(args.num_workers > 0))
    test_loader = DataLoader(test_dataset, batch_sampler=test_batch_sampler,
                             num_workers=args.num_workers, pin_memory=True,
                             persistent_workers=(args.num_workers > 0))

    # Loss, optimizer, scheduler
    if multilabel:
        # Per-label pos_weight = N_neg / N_pos; for the silencer head this
        # is ~290, which is exactly what BCEWithLogitsLoss is designed to
        # take. enhancer head is ~1.0 (essentially balanced).
        if hasattr(train_dataset.labels, "shape") and train_dataset.labels.ndim == 2:
            n_train = train_dataset.labels.shape[0]
            pos_counts = train_dataset.labels.astype(np.int64).sum(axis=0)
        else:
            raise RuntimeError("multilabel mode but train_dataset.labels is not 2-D")
        pos_weight_arr = np.array(
            [(n_train - p) / max(int(p), 1) for p in pos_counts], dtype=np.float32,
        )
        raw_pw = pos_weight_arr.tolist()
        if args.pos_weight_cap is not None and args.pos_weight_cap > 0:
            pos_weight_arr = np.minimum(pos_weight_arr,
                                        float(args.pos_weight_cap))
        if is_main(rank):
            if args.pos_weight_cap:
                print(f"  multilabel pos_weight raw={raw_pw} "
                      f"→ capped@{args.pos_weight_cap} "
                      f"→ {pos_weight_arr.tolist()}")
            else:
                print(f"  multilabel pos_weight: {pos_weight_arr.tolist()}")
            if args.label_smoothing > 0:
                print(f"  label smoothing ε={args.label_smoothing}")
        criterion = SmoothBCEWithLogitsLoss(
            pos_weight=torch.from_numpy(pos_weight_arr).to(device),
            smoothing=args.label_smoothing,
        )
    else:
        criterion = FocalLoss(alpha=class_weights, gamma=args.gamma).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4,
                      betas=(0.9, args.adamw_beta2))
    # Scheduler: bare CosineAnnealingLR (per-epoch) by default; SequentialLR
    # of LinearLR warmup → CosineAnnealingLR (per-batch) when --warmup-steps
    # > 0. The per-batch mode requires removing the per-epoch scheduler.step()
    # below to avoid double-stepping.
    if args.warmup_steps > 0:
        # steps_per_epoch is known after the BatchSampler is built; we
        # approximate here from the full_epoch_batches computed below.
        # T_max passed in steps, so compute total_steps after batch sampler exists.
        scheduler = None  # constructed after batch sampler is built; see below
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Early-stop policy.
    # - multiclass / multilabel+auroc: best_metric = max-so-far of a "higher
    #   is better" score (F1 or AUROC). New best when current > best.
    # - multilabel+loss: best_metric = min-so-far of val loss. We keep the
    #   "current > best" comparison path by storing NEGATIVE loss (so
    #   "higher = better" still holds and no branches change).
    es_mode_loss = bool(multilabel and args.early_stop_metric == "loss")
    init_best = float("-inf") if es_mode_loss else 0.0

    # Restore optimizer / scheduler / training-loop state from resume.
    start_epoch = 1
    initial_start_batch_idx = 0  # only nonzero on resume from mid-epoch save
    best_metric = init_best
    patience_counter = 0
    history = (
        {"train_loss": [], "val_loss": [], "val_macro_auroc": [],
         "val_macro_auprc": [], "val_macro_f1": []}
        if multilabel
        else {"train_loss": [], "val_loss": [], "val_macro_f1": [], "val_auroc": []}
    )
    if resume_state is not None:
        try:
            optimizer.load_state_dict(resume_state["optimizer_state_dict"])
            scheduler.load_state_dict(resume_state["scheduler_state_dict"])
            # Two save flavors:
            #   end-of-epoch (legacy):     epoch=N, next_batch_idx absent or 0
            #   mid-epoch (new):           epoch=N, next_batch_idx=K (>0)
            saved_next_batch = int(resume_state.get("next_batch_idx", 0))
            if saved_next_batch > 0:
                start_epoch = int(resume_state["epoch"])
                initial_start_batch_idx = saved_next_batch
            else:
                start_epoch = int(resume_state["epoch"]) + 1
                initial_start_batch_idx = 0
            best_metric = float(resume_state.get("best_metric",
                                resume_state.get("best_val_f1", 0.0)))
            patience_counter = int(resume_state.get("patience_counter", 0))
            history = resume_state.get("history", history)
            if is_main(rank):
                metric_name = "macro_auroc" if multilabel else "macro_f1"
                where = (f"epoch {start_epoch} batch {initial_start_batch_idx}"
                         if initial_start_batch_idx > 0
                         else f"epoch {start_epoch}")
                print(f"  [resume] resuming from {where} "
                      f"(best {metric_name} so far: {best_metric:.4f}, "
                      f"patience: {patience_counter}/{args.patience})")
        except Exception as e:
            if is_main(rank):
                print(f"  [resume] failed to restore optimizer/scheduler ({e}); "
                      f"keeping model weights but starting LR schedule fresh")
            start_epoch = 1
            initial_start_batch_idx = 0
            best_metric = init_best
            patience_counter = 0
            history = (
                {"train_loss": [], "val_loss": [], "val_macro_auroc": [],
                 "val_macro_auprc": [], "val_macro_f1": []}
                if multilabel
                else {"train_loss": [], "val_loss": [],
                      "val_macro_f1": [], "val_auroc": []}
            )

    # Full per-epoch batch count, BEFORE any start_batch skip is applied.
    # ChunkSortedBatchSampler.__len__ subtracts start_batch, so adding it
    # back recovers the unskipped total. Used as the denominator in
    # progress prints so "X/Y" stays stable across resumes.
    full_epoch_batches = len(train_batch_sampler) + train_batch_sampler.start_batch

    # If warmup is enabled, construct the per-step scheduler now that we know
    # full_epoch_batches. total_steps measured in optimizer.step() calls = one
    # per batch. The per-epoch scheduler.step() at the bottom of the loop is
    # SKIPPED in this mode (see the `if scheduler is not None and ...` guard).
    if args.warmup_steps > 0 and scheduler is None:
        total_steps = max(args.epochs * full_epoch_batches, args.warmup_steps + 1)
        warmup_sched = LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                                total_iters=args.warmup_steps)
        cosine_sched = CosineAnnealingLR(
            optimizer, T_max=total_steps - args.warmup_steps, eta_min=1e-6)
        scheduler = SequentialLR(optimizer, [warmup_sched, cosine_sched],
                                 milestones=[args.warmup_steps])
        if is_main(rank):
            print(f"  LR schedule: linear warmup {args.warmup_steps} steps "
                  f"→ cosine over {total_steps - args.warmup_steps} steps "
                  f"(per-batch stepping)")

    # Build the per-N-batches save closure once. Only rank 0 persists state;
    # other ranks no-op (calling save_fn from non-main rank would race writes).
    def _save_mid_epoch(cur_epoch, next_batch_idx):
        if not is_main(rank):
            return
        eval_model_local = model.module if hasattr(model, "module") else model
        last_ckpt = {
            "epoch": cur_epoch,
            "next_batch_idx": int(next_batch_idx),
            "model_type": model_type,
            "model_state_dict": eval_model_local.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_metric": best_metric,
            "task": task,
            "patience_counter": patience_counter,
            "history": history,
            "n_classes": n_classes,
            "n_tfs": n_tfs,
            "seq_length": seq_length,
            "label_list": label_list,
        }
        tmp_ckpt = last_ckpt_path + ".tmp"
        torch.save(last_ckpt, tmp_ckpt)
        os.replace(tmp_ckpt, last_ckpt_path)

    for epoch in range(start_epoch, args.epochs + 1):
        # Set BatchSampler's start_batch for this epoch:
        # nonzero only on the very first epoch when resuming mid-epoch.
        if epoch == start_epoch and initial_start_batch_idx > 0:
            train_batch_sampler.start_batch = initial_start_batch_idx
        else:
            train_batch_sampler.start_batch = 0

        if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)

        epoch_start = time.time()
        amp_dtype = _AMP_DTYPE_MAP.get(getattr(args, "amp_dtype", "fp32"))
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            start_batch_idx=train_batch_sampler.start_batch,
            save_fn=_save_mid_epoch,
            save_every=args.save_every_n_batches,
            total_batches=full_epoch_batches,
            amp_dtype=amp_dtype,
            scheduler=scheduler,
            max_grad_norm=args.max_grad_norm,
            scheduler_steps_per_batch=(args.warmup_steps > 0),
        )

        # Distributed val eval: every rank processes its shard; results are
        # all_gathered inside evaluate(). Must be called by ALL ranks to
        # avoid NCCL timeouts on the all_gather collective.
        eval_model = model.module if hasattr(model, "module") else model
        if multilabel:
            val_metrics = evaluate_multilabel(
                eval_model, val_loader, criterion, device,
                n_classes, label_list,
                world_size=world_size, rank=rank,
                amp_dtype=amp_dtype,
            )
        else:
            val_metrics = evaluate(
                eval_model, val_loader, criterion, device,
                n_classes, label_list,
                world_size=world_size, rank=rank,
                amp_dtype=amp_dtype,
            )

        if is_main(rank):
            epoch_time = time.time() - epoch_start
            if multilabel:
                print(f"  [{model_type}] Epoch {epoch}/{args.epochs} ({epoch_time:.1f}s) "
                      f"| Train Loss: {train_loss:.4f} "
                      f"| Val AUROC: {val_metrics['macro_auroc']:.4f} "
                      f"| Val AUPRC: {val_metrics['macro_auprc']:.4f} "
                      f"| Val F1@0.5: {val_metrics['macro_f1']:.4f}")
                for name, pl in val_metrics["per_label"].items():
                    print(f"      {name}: AUROC={pl['auroc']:.4f} "
                          f"AUPRC={pl['auprc']:.4f} F1@0.5={pl['f1_at_0p5']:.4f} "
                          f"(n_pos={pl['n_pos']:,}/{pl['n_total']:,})")
                # Higher-is-better sign: AUROC straight; loss → negated.
                if es_mode_loss:
                    current_metric = -float(val_metrics["loss"])
                else:
                    current_metric = val_metrics["macro_auroc"]
                history["train_loss"].append(train_loss)
                history["val_loss"].append(val_metrics["loss"])
                history["val_macro_auroc"].append(val_metrics["macro_auroc"])
                history["val_macro_auprc"].append(val_metrics["macro_auprc"])
                history["val_macro_f1"].append(val_metrics["macro_f1"])
            else:
                print(f"  [{model_type}] Epoch {epoch}/{args.epochs} ({epoch_time:.1f}s) "
                      f"| Train Loss: {train_loss:.4f} | Val F1: {val_metrics['macro_f1']:.4f} "
                      f"| Val AUROC: {val_metrics['auroc']:.4f}")
                current_metric = val_metrics["macro_f1"]
                history["train_loss"].append(train_loss)
                history["val_loss"].append(val_metrics["loss"])
                history["val_macro_f1"].append(val_metrics["macro_f1"])
                history["val_auroc"].append(val_metrics["auroc"])

            # High-confidence early stop: only count as new best if the
            # improvement exceeds args.min_delta. Reviewer-recommended to
            # avoid stopping when val loss is just plateauing noisily.
            if current_metric > best_metric + args.min_delta:
                best_metric = current_metric
                patience_counter = 0
                checkpoint = {
                    "epoch": epoch,
                    "model_type": model_type,
                    "model_state_dict": eval_model.state_dict(),
                    "best_metric": best_metric,
                    "task": task,
                    "n_classes": n_classes,
                    "n_tfs": n_tfs,
                    "seq_length": seq_length,
                    "label_list": label_list,
                }
                if multilabel:
                    checkpoint["val_macro_auroc"] = val_metrics["macro_auroc"]
                    checkpoint["val_macro_auprc"] = val_metrics["macro_auprc"]
                    checkpoint["val_macro_f1"] = val_metrics["macro_f1"]
                else:
                    checkpoint["val_macro_f1"] = best_metric
                    checkpoint["val_auroc"] = val_metrics["auroc"]
                torch.save(checkpoint, os.path.join(model_out_dir, "best_model.pt"))
                if es_mode_loss:
                    metric_label = "Val-Loss"
                    shown = -best_metric  # un-negate for display
                else:
                    metric_label = "AUROC" if multilabel else "F1"
                    shown = best_metric
                print(f"    -> New best! {metric_label}={shown:.4f}")
            else:
                patience_counter += 1

            # Per-epoch resumable checkpoint — atomic write so a crash
            # mid-save can never leave a corrupt last_ckpt.pt.
            # next_batch_idx=0 is the sentinel for "epoch fully completed,
            # resume from epoch+1". Distinguishes this from per-N-batches
            # mid-epoch saves (next_batch_idx > 0).
            last_ckpt = {
                "epoch": epoch,
                "next_batch_idx": 0,
                "model_type": model_type,
                "model_state_dict": eval_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_metric": best_metric,
                "task": task,
                "patience_counter": patience_counter,
                "history": history,
                "n_classes": n_classes,
                "n_tfs": n_tfs,
                "seq_length": seq_length,
                "label_list": label_list,
            }
            tmp_ckpt = last_ckpt_path + ".tmp"
            torch.save(last_ckpt, tmp_ckpt)
            os.replace(tmp_ckpt, last_ckpt_path)

            if patience_counter >= args.patience:
                print(f"    Early stopping triggered at epoch {epoch}.")
                # NOTE: do NOT break here. Only rank 0 reaches this branch,
                # and breaking only on rank 0 leaves ranks 1..N waiting on
                # the broadcast below — guaranteed deadlock until NCCL
                # watchdog fires. The break is performed by ALL ranks
                # together after the broadcast.

        # Per-epoch scheduler step ONLY when not stepping per-batch. With
        # --warmup-steps > 0, train_one_epoch advances the scheduler once per
        # batch and this call would double-step.
        if args.warmup_steps == 0:
            scheduler.step()

        # Cross-rank early-stop sync: rank 0 broadcasts whether to stop;
        # all ranks break together so the test-eval barrier downstream
        # has every rank participating.
        if world_size > 1:
            stop = torch.tensor([1 if patience_counter >= args.patience else 0],
                                device=device)
            dist.broadcast(stop, src=0)
            if stop.item() == 1:
                break
        else:
            if patience_counter >= args.patience:
                break

    # Test evaluation: all ranks load the best checkpoint and run eval on
    # their shard. Gather happens inside evaluate(). Rank 0 handles I/O.
    if world_size > 1:
        dist.barrier()  # ensure rank 0 has finished writing best_model.pt
    best_ckpt_path = os.path.join(model_out_dir, "best_model.pt")
    ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    eval_model = model_cls(n_classes=n_classes, n_tfs=n_tfs, seq_length=seq_length).to(device)
    eval_model.load_state_dict(ckpt["model_state_dict"])
    test_amp_dtype = _AMP_DTYPE_MAP.get(getattr(args, "amp_dtype", "fp32"))
    if multilabel:
        test_metrics = evaluate_multilabel(
            eval_model, test_loader, criterion, device,
            n_classes, label_list,
            world_size=world_size, rank=rank,
            amp_dtype=test_amp_dtype,
        )
    else:
        test_metrics = evaluate(
            eval_model, test_loader, criterion, device,
            n_classes, label_list,
            world_size=world_size, rank=rank,
            amp_dtype=test_amp_dtype,
        )

    if is_main(rank):
        print(f"\n  [{model_type}] TEST RESULTS:")
        if multilabel:
            print(f"    Macro AUROC:   {test_metrics['macro_auroc']:.4f}")
            print(f"    Macro AUPRC:   {test_metrics['macro_auprc']:.4f}")
            print(f"    Macro F1@0.5: {test_metrics['macro_f1']:.4f}")
            for name, pl in test_metrics["per_label"].items():
                print(f"      {name}: AUROC={pl['auroc']:.4f} "
                      f"AUPRC={pl['auprc']:.4f} F1@0.5={pl['f1_at_0p5']:.4f} "
                      f"(n_pos={pl['n_pos']:,}/{pl['n_total']:,})")
            results = {
                "model_type": model_type,
                "task": task,
                "best_epoch": ckpt["epoch"],
                "best_val_macro_auroc": float(ckpt.get("val_macro_auroc",
                                                       ckpt.get("best_metric", 0.0))),
                "test_macro_auroc": float(test_metrics["macro_auroc"]),
                "test_macro_auprc": float(test_metrics["macro_auprc"]),
                "test_macro_f1": float(test_metrics["macro_f1"]),
                "test_per_label": test_metrics["per_label"],
                "history": history,
            }
        else:
            print(f"    Macro-F1:    {test_metrics['macro_f1']:.4f}")
            print(f"    Weighted-F1: {test_metrics['weighted_f1']:.4f}")
            print(f"    AUROC:       {test_metrics['auroc']:.4f}")
            print(classification_report(
                test_metrics["labels"], test_metrics["predictions"],
                target_names=label_list, zero_division=0,
            ))
            results = {
                "model_type": model_type,
                "task": task,
                "best_epoch": ckpt["epoch"],
                "best_val_macro_f1": float(ckpt.get("val_macro_f1",
                                                    ckpt.get("best_metric", 0.0))),
                "test_macro_f1": float(test_metrics["macro_f1"]),
                "test_weighted_f1": float(test_metrics["weighted_f1"]),
                "test_auroc": float(test_metrics["auroc"]),
                "test_report": test_metrics["report"],
                "confusion_matrix": test_metrics["confusion_matrix"].tolist(),
                "history": history,
            }
        with open(os.path.join(model_out_dir, "training_results.json"), "w") as f:
            json.dump(results, f, indent=2)

        np.savez_compressed(
            os.path.join(model_out_dir, "test_predictions.npz"),
            predictions=test_metrics["predictions"],
            labels=test_metrics["labels"],
            probabilities=test_metrics["probabilities"],
        )

        # ---- Mark this model done and drop the now-redundant last_ckpt ----
        # The marker is the "this model is complete" handshake for resume.
        # last_ckpt.pt is only useful for in-progress training, so once we
        # have training_results.json + test_predictions.npz + done.marker,
        # the per-epoch checkpoint is dead weight.
        with open(done_marker + ".tmp", "w") as f:
            f.write(f"time={time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"task={task}\n")
            f.write(f"best_epoch={ckpt['epoch']}\n")
            if multilabel:
                f.write(f"best_val_macro_auroc="
                        f"{float(ckpt.get('val_macro_auroc', ckpt.get('best_metric', 0.0))):.6f}\n")
                f.write(f"test_macro_auroc={float(test_metrics['macro_auroc']):.6f}\n")
                f.write(f"test_macro_auprc={float(test_metrics['macro_auprc']):.6f}\n")
            else:
                f.write(f"best_val_f1="
                        f"{float(ckpt.get('val_macro_f1', ckpt.get('best_metric', 0.0))):.6f}\n")
                f.write(f"test_macro_f1={float(test_metrics['macro_f1']):.6f}\n")
        os.replace(done_marker + ".tmp", done_marker)
        if os.path.isfile(last_ckpt_path):
            try:
                os.remove(last_ckpt_path)
            except OSError:
                pass
        print(f"  [{model_type}] done.marker written; last_ckpt.pt removed.")

    return test_metrics


def train_xgboost(train_dataset, val_dataset, test_dataset,
                  n_classes, label_list, output_dir):
    """XGBoost / HistGradientBoostingClassifier is disabled for this dataset.

    Rationale:
      - The fit requires all features in memory simultaneously. At 529M rows
        × (n_tfs + tabular + seq-derived) float32 features ≈ 1.8 TB, this
        exceeds node RAM.
      - The only way to shrink it is to subsample training data, which the
        user has explicitly forbidden.
      - DL models (CNN / Transformer / Hybrid) remain and will be ensembled
        without an XGBoost component.

    If GPU XGBoost with QuantileDMatrix streaming becomes a hard requirement,
    swap this stub for a proper external-memory xgb.train loop.
    """
    print(f"\n{'='*70}")
    print(f"  XGBOOST: DISABLED")
    print(f"{'='*70}")
    print("  Skipping XGBoost — in-memory fit infeasible (~1.8 TB features) and")
    print("  subsampling is disallowed by user policy. DL-only ensemble will")
    print("  be built from cnn + transformer + hybrid.")
    return None


def _train_xgboost_legacy_disabled(train_dataset, val_dataset, test_dataset,
                                   n_classes, label_list, output_dir):
    """Kept for reference; never called. The active implementation is the
    no-op stub above."""
    try:
        from sklearn.ensemble import GradientBoostingClassifier  # noqa: F401
    except ImportError:
        print("  scikit-learn not available, skipping XGBoost.")
        return None

    xgb_dir = os.path.join(output_dir, "xgboost")
    os.makedirs(xgb_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  TRAINING MODEL: XGBOOST (GradientBoosting)")
    print(f"{'='*70}")

    def extract_features(dataset):
        """Extract tabular features + k-mer counts from sequences."""
        n = len(dataset)
        # TF features + tabular features
        tf_dim = dataset.tf_features.shape[1]
        tab_dim = dataset.tabular_features.shape[1]

        # k-mer features: compute 3-mer and 4-mer frequencies from one-hot sequences
        # Instead of full k-mer, use summary stats from sequences for efficiency
        n_seq_features = 20  # GC content, dinucleotide freqs, etc.
        total_dim = tf_dim + tab_dim + n_seq_features

        X = np.zeros((n, total_dim), dtype=np.float32)
        y = dataset.labels.copy()

        for i in range(n):
            seq_onehot = dataset.sequences[i]  # (seq_len, 4)

            # TF and tabular
            X[i, :tf_dim] = dataset.tf_features[i]
            X[i, tf_dim:tf_dim + tab_dim] = dataset.tabular_features[i]

            # Sequence-derived features
            offset = tf_dim + tab_dim
            # Base composition (A, C, G, T fractions)
            base_fracs = seq_onehot.mean(axis=0)
            X[i, offset:offset + 4] = base_fracs

            # GC content
            X[i, offset + 4] = base_fracs[1] + base_fracs[2]

            # Dinucleotide frequencies (16 values, compressed to top ones)
            # Use argmax per position to get sequence, then count dinucs
            seq_idx = seq_onehot.argmax(axis=1)
            dinuc_counts = np.zeros(16, dtype=np.float32)
            for j in range(len(seq_idx) - 1):
                di = seq_idx[j] * 4 + seq_idx[j + 1]
                dinuc_counts[di] += 1
            dinuc_freqs = dinuc_counts / max(dinuc_counts.sum(), 1)
            # Top 15 dinucleotide features
            X[i, offset + 5:offset + 20] = dinuc_freqs[:15]

            if i % 50000 == 0 and i > 0:
                print(f"    Feature extraction: {i}/{n}")

        return X, y

    print("  Extracting training features...")
    X_train, y_train = extract_features(train_dataset)
    print(f"    Train: {X_train.shape}")

    print("  Extracting validation features...")
    X_val, y_val = extract_features(val_dataset)
    print(f"    Val: {X_val.shape}")

    print("  Extracting test features...")
    X_test, y_test = extract_features(test_dataset)
    print(f"    Test: {X_test.shape}")

    # Subsample if too large for sklearn (>500K)
    max_train = 500000
    if len(X_train) > max_train:
        print(f"  Subsampling training data from {len(X_train)} to {max_train}")
        idx = np.random.choice(len(X_train), max_train, replace=False)
        X_train = X_train[idx]
        y_train = y_train[idx]

    print("  Training GradientBoostingClassifier...")
    from sklearn.ensemble import HistGradientBoostingClassifier

    clf = HistGradientBoostingClassifier(
        max_iter=500,
        max_depth=8,
        learning_rate=0.05,
        min_samples_leaf=50,
        l2_regularization=1.0,
        max_bins=255,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        verbose=1,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)

    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    try:
        auroc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
    except ValueError:
        auroc = 0.0

    print(f"\n  [xgboost] TEST RESULTS:")
    print(f"    Macro-F1:    {macro_f1:.4f}")
    print(f"    Weighted-F1: {weighted_f1:.4f}")
    print(f"    AUROC:       {auroc:.4f}")
    print(classification_report(y_test, y_pred, target_names=label_list, zero_division=0))

    # Save model and results
    with open(os.path.join(xgb_dir, "model.pkl"), "wb") as f:
        pickle.dump(clf, f)

    results = {
        "model_type": "xgboost",
        "test_macro_f1": float(macro_f1),
        "test_weighted_f1": float(weighted_f1),
        "test_auroc": float(auroc),
        "test_report": classification_report(y_test, y_pred, target_names=label_list,
                                             zero_division=0, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    with open(os.path.join(xgb_dir, "training_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    np.savez_compressed(
        os.path.join(xgb_dir, "test_predictions.npz"),
        predictions=y_pred,
        labels=y_test,
        probabilities=y_prob,
    )

    return {
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "auroc": auroc,
        "predictions": y_pred,
        "labels": y_test,
        "probabilities": y_prob,
    }


def create_ensemble(output_dir, label_list, n_classes):
    """Create a weighted ensemble from all trained models' test predictions."""
    print(f"\n{'='*70}")
    print(f"  CREATING ENSEMBLE")
    print(f"{'='*70}")

    model_results = {}
    for model_type in ["cnn", "transformer", "hybrid", "xgboost"]:
        pred_path = os.path.join(output_dir, model_type, "test_predictions.npz")
        result_path = os.path.join(output_dir, model_type, "training_results.json")
        if os.path.isfile(pred_path) and os.path.isfile(result_path):
            data = np.load(pred_path)
            with open(result_path) as f:
                res = json.load(f)
            model_results[model_type] = {
                "probabilities": data["probabilities"],
                "labels": data["labels"],
                "macro_f1": res["test_macro_f1"],
            }

    if len(model_results) < 2:
        print("  Less than 2 models available, skipping ensemble.")
        return

    # Use validation F1 as weights (softmax-normalized)
    f1_scores = {k: v["macro_f1"] for k, v in model_results.items()}
    print(f"  Individual model F1 scores: {f1_scores}")

    # Softmax normalization of F1 scores for weights
    f1_vals = np.array(list(f1_scores.values()))
    weights = np.exp(f1_vals * 10) / np.exp(f1_vals * 10).sum()  # temperature=0.1
    model_names = list(f1_scores.keys())
    print(f"  Ensemble weights: {dict(zip(model_names, [f'{w:.3f}' for w in weights]))}")

    # Weighted average of probabilities
    labels = list(model_results.values())[0]["labels"]
    ensemble_probs = np.zeros_like(list(model_results.values())[0]["probabilities"],
                                   dtype=np.float64)
    for i, name in enumerate(model_names):
        ensemble_probs += weights[i] * model_results[name]["probabilities"]

    ens_dir = os.path.join(output_dir, "ensemble")
    os.makedirs(ens_dir, exist_ok=True)

    # Multilabel (labels is 2-D, one column per independent sigmoid head) vs
    # multiclass (labels is 1-D class index). The two need different metric
    # computation: multilabel thresholds each head independently; multiclass
    # uses argmax.
    is_multilabel = (np.asarray(labels).ndim == 2)

    if is_multilabel:
        ensemble_preds = (ensemble_probs >= 0.5).astype(np.int64)
        per_label = {}
        aurocs, auprcs, f1s = [], [], []
        for c, name in enumerate(label_list):
            y_true = labels[:, c]
            y_score = ensemble_probs[:, c]
            y_pred = ensemble_preds[:, c]
            try:
                auroc_c = float(roc_auc_score(y_true, y_score)) if y_true.sum() > 0 else 0.0
            except ValueError:
                auroc_c = 0.0
            try:
                auprc_c = float(average_precision_score(y_true, y_score)) if y_true.sum() > 0 else 0.0
            except ValueError:
                auprc_c = 0.0
            f1_c = float(f1_score(y_true, y_pred, zero_division=0))
            per_label[name] = {"auroc": auroc_c, "auprc": auprc_c, "f1_at_0p5": f1_c,
                               "n_pos": int(y_true.sum())}
            aurocs.append(auroc_c); auprcs.append(auprc_c); f1s.append(f1_c)
        macro_auroc = float(np.mean(aurocs))
        macro_auprc = float(np.mean(auprcs))
        macro_f1 = float(np.mean(f1s))
        print(f"\n  [ENSEMBLE] TEST RESULTS (multilabel):")
        print(f"    Macro AUROC:  {macro_auroc:.4f}")
        print(f"    Macro AUPRC:  {macro_auprc:.4f}")
        print(f"    Macro F1@0.5: {macro_f1:.4f}")
        for name, pl in per_label.items():
            print(f"      {name}: AUROC={pl['auroc']:.4f} AUPRC={pl['auprc']:.4f} "
                  f"F1@0.5={pl['f1_at_0p5']:.4f}")
        results = {
            "model_type": "ensemble",
            "task": "multilabel",
            "component_models": model_names,
            "weights": dict(zip(model_names, weights.tolist())),
            "component_f1_scores": f1_scores,
            "test_macro_auroc": macro_auroc,
            "test_macro_auprc": macro_auprc,
            "test_macro_f1": macro_f1,
            "test_per_label": per_label,
        }
    else:
        ensemble_preds = ensemble_probs.argmax(axis=1)
        macro_f1 = f1_score(labels, ensemble_preds, average="macro", zero_division=0)
        weighted_f1 = f1_score(labels, ensemble_preds, average="weighted", zero_division=0)
        try:
            auroc = roc_auc_score(labels, ensemble_probs, multi_class="ovr", average="macro")
        except ValueError:
            auroc = 0.0
        print(f"\n  [ENSEMBLE] TEST RESULTS:")
        print(f"    Macro-F1:    {macro_f1:.4f}")
        print(f"    Weighted-F1: {weighted_f1:.4f}")
        print(f"    AUROC:       {auroc:.4f}")
        print(classification_report(labels, ensemble_preds, target_names=label_list, zero_division=0))
        results = {
            "model_type": "ensemble",
            "component_models": model_names,
            "weights": dict(zip(model_names, weights.tolist())),
            "component_f1_scores": f1_scores,
            "test_macro_f1": float(macro_f1),
            "test_weighted_f1": float(weighted_f1),
            "test_auroc": float(auroc),
            "test_report": classification_report(labels, ensemble_preds, target_names=label_list,
                                                 zero_division=0, output_dict=True),
            "confusion_matrix": confusion_matrix(labels, ensemble_preds).tolist(),
        }

    with open(os.path.join(ens_dir, "training_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    np.savez_compressed(
        os.path.join(ens_dir, "test_predictions.npz"),
        predictions=ensemble_preds,
        labels=labels,
        probabilities=ensemble_probs,
    )
    return results


def select_best_model(output_dir):
    """Compare all models and select the best one."""
    print(f"\n{'='*70}")
    print(f"  MODEL COMPARISON & SELECTION")
    print(f"{'='*70}\n")

    comparison = {}
    for model_type in ["cnn", "transformer", "hybrid", "xgboost", "ensemble"]:
        result_path = os.path.join(output_dir, model_type, "training_results.json")
        if os.path.isfile(result_path):
            with open(result_path) as f:
                res = json.load(f)
            comparison[model_type] = {
                "macro_f1": res["test_macro_f1"],
                "weighted_f1": res.get("test_weighted_f1", 0),
                "auroc": res.get("test_auroc", 0),
            }

    if not comparison:
        print("  No model results found!")
        return

    # Print comparison table
    print(f"  {'Model':<15} {'Macro-F1':>10} {'Weighted-F1':>12} {'AUROC':>10}")
    print(f"  {'-'*15} {'-'*10} {'-'*12} {'-'*10}")
    for name, metrics in sorted(comparison.items(), key=lambda x: -x[1]["macro_f1"]):
        print(f"  {name:<15} {metrics['macro_f1']:>10.4f} {metrics['weighted_f1']:>12.4f} {metrics['auroc']:>10.4f}")

    best_model = max(comparison, key=lambda k: comparison[k]["macro_f1"])
    print(f"\n  BEST MODEL: {best_model} (Macro-F1 = {comparison[best_model]['macro_f1']:.4f})")

    # Copy best model to top-level for use by prediction script
    if best_model in ["cnn", "transformer", "hybrid"]:
        import shutil
        src = os.path.join(output_dir, best_model, "best_model.pt")
        dst = os.path.join(output_dir, "best_model.pt")
        if os.path.isfile(src):
            shutil.copy2(src, dst)
            print(f"  Copied {src} -> {dst}")
    elif best_model == "ensemble":
        # For ensemble, save the component info so prediction script can load all models
        ens_info = os.path.join(output_dir, "ensemble", "training_results.json")
        with open(ens_info) as f:
            ens_data = json.load(f)
        ens_data["is_ensemble"] = True
        with open(os.path.join(output_dir, "best_model_info.json"), "w") as f:
            json.dump(ens_data, f, indent=2)
        # Also copy the best single DL model for fallback
        best_single = max(
            {k: v for k, v in comparison.items() if k not in ["ensemble", "xgboost"]},
            key=lambda k: comparison[k]["macro_f1"],
            default=None,
        )
        if best_single:
            import shutil
            src = os.path.join(output_dir, best_single, "best_model.pt")
            dst = os.path.join(output_dir, "best_model.pt")
            if os.path.isfile(src):
                shutil.copy2(src, dst)
                print(f"  Copied best single DL model ({best_single}) -> {dst}")

    # Save comparison
    with open(os.path.join(output_dir, "model_comparison.json"), "w") as f:
        json.dump(comparison, f, indent=2)

    return best_model


def main():
    parser = argparse.ArgumentParser(description="Train regulatory region classifiers")
    parser.add_argument("--data-dir", type=str,
                        default="/work/avinash/user/CA/CA/unbound_characetrize/training_data")
    parser.add_argument("--output-dir", type=str,
                        default="/work/avinash/user/CA/CA/unbound_characetrize/model_output")
    parser.add_argument("--model-type", type=str, default="all",
                        help="Which model(s) to train. "
                             "Values: 'all' (cnn+transformer+hybrid), "
                             "single name (cnn/transformer/hybrid/xgboost), "
                             "or comma-separated subset (e.g. 'cnn,hybrid'). "
                             "Default: all")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp-dtype", type=str, default="bf16",
                        choices=["fp32", "bf16", "fp16"],
                        help="Mixed-precision dtype for autocast on cuda. "
                             "bf16 (default) is ~2-3x faster on tensor-core "
                             "GPUs with no accuracy loss for this workload "
                             "and no GradScaler required. fp32 disables AMP.")
    parser.add_argument("--label-scheme", type=str, default="5class",
                        choices=list(LABEL_SCHEMES.keys()),
                        help="7class = original targets; 5class merges "
                             "TFBS + cis_regulatory into other_regulatory. "
                             "All rows are retained either way — only the "
                             "classification target is regrouped. Ignored "
                             "when --task=multilabel.")
    parser.add_argument("--task", type=str, default="multiclass",
                        choices=["multiclass", "multilabel"],
                        help="multiclass: softmax over label-scheme classes "
                             "(legacy). multilabel: independent sigmoid "
                             "heads (see --multilabel-set). Requires 2e.")
    parser.add_argument("--multilabel-set", type=str, default="basic",
                        choices=["basic", "full"],
                        help="basic: lab2 (is_enhancer, is_silencer) — the "
                             "original 2-head setup. full: lab4 "
                             "(is_enhancer, is_promoter, is_genic, "
                             "is_silencer) — predicts every annotation type "
                             "in the source data; downstream code can "
                             "re-bucket (e.g. reg = enh ∨ prom ∨ sil).")
    parser.add_argument("--early-stop-metric", type=str, default="auroc",
                        choices=["auroc", "loss"],
                        help="auroc: stop when val macro-AUROC stops "
                             "improving (higher is better). loss: stop "
                             "when val BCE loss stops decreasing (lower "
                             "is better) — reviewer-recommended; AUROC can "
                             "saturate while probabilities still drift.")
    parser.add_argument("--pos-weight-cap", type=float, default=None,
                        help="If set, clip per-label BCE pos_weight to this "
                             "max. Default None = raw N_neg/N_pos (which can "
                             "reach ~20 for silencer and over-inflate the "
                             "logits). 5.0 is a reasonable cap.")
    parser.add_argument("--label-smoothing", type=float, default=0.0,
                        help="BCE target smoothing ε ∈ [0, 0.2]. Targets "
                             "become y·(1-2ε)+ε. Reduces over-confidence "
                             "and rising val loss.")
    parser.add_argument("--min-delta", type=float, default=0.0,
                        help="Minimum improvement on the early-stop metric "
                             "to count as a new best. Combined with a high "
                             "--patience, this gives high-confidence "
                             "convergence: with min-delta=1e-3 and "
                             "patience=8, training only stops after 8 "
                             "consecutive epochs whose improvement is "
                             "below 1e-3. Applied to AUROC and to -loss "
                             "identically.")
    parser.add_argument("--warmup-steps", type=int, default=0,
                        help="Number of LR linear-warmup steps before "
                             "cosine annealing. 0 = no warmup (cosine from "
                             "step 0, the current default). 1000 is the "
                             "recommended value for Hybrid/Transformer to "
                             "avoid attention divergence in the first few "
                             "hundred steps. When >0, scheduler steps "
                             "per-batch instead of per-epoch.")
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
                        help="Gradient L2-norm clip threshold. 0 disables. "
                             "Default 1.0 matches prior hardcoded behavior.")
    parser.add_argument("--adamw-beta2", type=float, default=0.999,
                        help="AdamW β2 (second-moment EMA decay). Default "
                             "0.999. Set 0.95 for Transformer-heavy models "
                             "where the slower-adapting variance estimate "
                             "destabilizes attention during warmup.")
    parser.add_argument("--samples-per-epoch", type=int, default=20_000_000,
                        help="Number of weighted samples drawn per epoch "
                             "across ALL ranks. 0 = full dataset size per "
                             "epoch (slow). Default 20M gives ~8 min/epoch "
                             "on 8 A5000 @ bs=512.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-chroms", type=str, default=None,
                        help="Comma-separated chrom names overriding the default "
                             "train split (e.g. 'chr1,chr2,...'). For LOCO-CV.")
    parser.add_argument("--val-chroms", type=str, default=None,
                        help="Comma-separated val chroms (override).")
    parser.add_argument("--test-chroms", type=str, default=None,
                        help="Comma-separated test/held-out chroms (override).")
    parser.add_argument("--save-every-n-batches", type=int, default=500,
                        help="Per-N-batches resumable checkpoint cadence inside "
                             "an epoch. 0 disables; otherwise last_ckpt.pt is "
                             "written every N global batches with a "
                             "next_batch_idx field so a crash mid-epoch "
                             "restores up to the last save point. Default 500 "
                             "(~30-60 min at warm uint8 pace).")
    args = parser.parse_args()

    rank, local_rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if is_main(rank):
        print(f"Training with {world_size} GPU(s) on {device}")
        os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset info (original 7-class metadata).
    with open(os.path.join(args.data_dir, "dataset_info.json")) as f:
        info = json.load(f)

    orig_label_list = info["label_list"]
    orig_counts = info.get("class_counts", {})
    n_tfs = info["n_tfs"]
    seq_length = info["seq_length"]

    multilabel = (args.task == "multilabel")
    if multilabel:
        # Multilabel ignores label-scheme; basic = 2 heads, full = 4 heads.
        full_ml = (args.multilabel_set == "full")
        if full_ml:
            label_list = ["is_enhancer", "is_promoter", "is_genic", "is_silencer"]
        else:
            label_list = ["is_enhancer", "is_silencer"]
        n_classes = len(label_list)
        label_remap = None
        # Pull pos counts from dataset_info_mmap (written by 02e).
        try:
            with open(os.path.join(args.data_dir, "mmap", "dataset_info_mmap.json")) as f:
                mmap_info = json.load(f)
            pos_counts_key = "multilabel4_pos_counts" if full_ml else "multilabel_pos_counts"
            pos_counts = mmap_info.get(pos_counts_key, {})
        except FileNotFoundError:
            pos_counts = {}
        if is_main(rank):
            print(f"Dataset: {info['n_regions']} regions, {n_tfs} TFs")
            print(f"Task: multilabel-{args.multilabel_set} "
                  f"({n_classes} sigmoid heads)")
            print(f"Targets: {label_list}")
            if pos_counts:
                print(f"Positive counts: {pos_counts}")
        # In multilabel mode, skip the rest of the multiclass scheme block.
        scheme = None
    else:
        scheme = LABEL_SCHEMES[args.label_scheme]
        # The `remap` array in LABEL_SCHEMES is indexed by the POSITION of each
        # class in ORIGINAL_LABELS. Build a mapping that matches the dataset's
        # actual label order (dataset_info.json gives the vocabulary).
        if orig_label_list != ORIGINAL_LABELS:
            position_of = {name: i for i, name in enumerate(ORIGINAL_LABELS)}
            label_remap = np.array(
                [scheme["remap"][position_of[name]] for name in orig_label_list],
                dtype=np.int64,
            )
        else:
            label_remap = scheme["remap"]
        label_list = scheme["label_list"]
        n_classes = len(label_list)

        if is_main(rank):
            print(f"Dataset: {info['n_regions']} regions, {n_tfs} TFs")
            print(f"Label scheme: {args.label_scheme} ({n_classes} classes)")
            print(f"Classes: {label_list}")
            print(f"Original 7-class counts: {orig_counts}")
            effective = {name: 0 for name in label_list}
            for orig_name, cnt in orig_counts.items():
                new_idx = label_remap[orig_label_list.index(orig_name)]
                effective[label_list[new_idx]] += int(cnt)
            print(f"Effective class counts ({args.label_scheme}): {effective}")

    # Load datasets — lazy / mmap-backed.
    if is_main(rank):
        print("\nLoading datasets (lazy mmap)...")
    # Resolve chromosome splits — CLI overrides (for LOCO-CV) else defaults.
    def _parse_chroms(s, default):
        if not s:
            return default
        return {c.strip() for c in s.split(",") if c.strip()}
    tr_chroms = _parse_chroms(args.train_chroms, TRAIN_CHROMS)
    va_chroms = _parse_chroms(args.val_chroms, VAL_CHROMS)
    te_chroms = _parse_chroms(args.test_chroms, TEST_CHROMS)
    if is_main(rank):
        print(f"  Chrom splits — train:{sorted(tr_chroms)} "
              f"val:{sorted(va_chroms)} test:{sorted(te_chroms)}")

    if multilabel:
        ds_kwargs = dict(multilabel=True, multilabel_set=args.multilabel_set)
        train_dataset = RegulomeDataset(args.data_dir, chromosomes=tr_chroms,
                                        augment=True, **ds_kwargs)
        val_dataset = RegulomeDataset(args.data_dir, chromosomes=va_chroms,
                                      augment=False, **ds_kwargs)
        test_dataset = RegulomeDataset(args.data_dir, chromosomes=te_chroms,
                                       augment=False, **ds_kwargs)
        class_weights = None  # not used in multilabel; BCE pos_weight set inside
    else:
        common = dict(augment=False, label_remap=label_remap)
        train_dataset = RegulomeDataset(args.data_dir, chromosomes=tr_chroms,
                                        augment=True, label_remap=label_remap)
        val_dataset = RegulomeDataset(args.data_dir, chromosomes=va_chroms, **common)
        test_dataset = RegulomeDataset(args.data_dir, chromosomes=te_chroms, **common)
        class_weights = compute_class_weights(train_dataset.labels, n_classes)

    if is_main(rank):
        print(f"  Train: {len(train_dataset):,} | Val: {len(val_dataset):,} | Test: {len(test_dataset):,}")

    # Determine which models to train. XGBoost is disabled regardless of task,
    # and even more clearly not applicable in multilabel mode.
    # `--model-type` accepts:
    #   "all"                 → cnn + transformer + hybrid
    #   "xgboost"             → just xgboost (currently disabled)
    #   "cnn,hybrid" (etc.)   → explicit comma-separated subset (any order)
    #   single name           → just that one model
    if args.model_type == "all":
        dl_models = ["cnn", "transformer", "hybrid"]
        do_xgboost = True
    elif args.model_type == "xgboost":
        dl_models = []
        do_xgboost = True
    elif "," in args.model_type:
        valid = {"cnn", "transformer", "hybrid"}
        dl_models = [m.strip() for m in args.model_type.split(",") if m.strip()]
        bad = [m for m in dl_models if m not in valid]
        if bad:
            raise ValueError(f"Unknown model name(s) in --model-type: {bad}; "
                             f"valid choices are {sorted(valid)}")
        do_xgboost = False
    else:
        dl_models = [args.model_type]
        do_xgboost = False

    # Train DL models sequentially
    for model_type in dl_models:
        # Check if already trained — skip ONLY if done.marker is present.
        # best_model.pt alone isn't enough (test eval may not have run).
        marker_path = os.path.join(args.output_dir, model_type, "done.marker")
        if os.path.isfile(marker_path):
            if is_main(rank):
                print(f"\n  {model_type} done.marker present, skipping.")
            continue

        model_cls = MODEL_REGISTRY[model_type]
        train_dl_model(
            model_type, model_cls,
            train_dataset, val_dataset, test_dataset,
            n_classes, n_tfs, seq_length, label_list, class_weights,
            args, rank, local_rank, world_size, device,
            task=args.task,
        )

    # Train XGBoost (only on main process, not distributed)
    if do_xgboost and is_main(rank):
        xgb_result_path = os.path.join(args.output_dir, "xgboost", "training_results.json")
        if os.path.isfile(xgb_result_path):
            print(f"\n  xgboost already trained, skipping.")
        else:
            train_xgboost(train_dataset, val_dataset, test_dataset,
                          n_classes, label_list, args.output_dir)

    # Ensemble + comparison (main process only). Runs for ANY trained subset,
    # not just "all" — create_ensemble needs >=2 models with test_predictions,
    # select_best_model always runs to write the top-level best_model.pt that
    # 05_predict_uncharacterized.py loads.
    if is_main(rank) and len(dl_models) >= 1:
        if len(dl_models) >= 2:
            create_ensemble(args.output_dir, label_list, n_classes)
        select_best_model(args.output_dir)

    if is_main(rank):
        print(f"\nAll training complete. Results in {args.output_dir}")

    cleanup_ddp()


if __name__ == "__main__":
    main()
