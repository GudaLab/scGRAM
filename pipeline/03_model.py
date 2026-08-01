#!/usr/bin/env python3
"""
03_model.py
============
Multi-modal regulatory region classifier:
  - Branch 1: 1D CNN on one-hot encoded DNA sequences (Basset/ChromBPNet-style)
  - Branch 2: Dense network on TF binding features (binary TF vector + tabular)
  - Fusion: concatenation + dense layers + softmax classification

Also defines:
  - FocalLoss for class imbalance handling
  - RegulomeDataset for loading training data
"""

import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


# ========== Dataset ==========

class RegulomeDataset(Dataset):
    """Lazy, memory-mapped Dataset backed by per-chunk uncompressed .npy files
    produced by 02b_convert_to_mmap.py. No eager concatenation — the full 529M
    rows do not fit in RAM (4.2 TB if decompressed).

    Features:
      - chromosome filtering via per-chunk uint8 chrom codes (no 22 GB CSV read)
      - optional label remapping (e.g. 7-class → 5-class) via a lookup array
      - per-worker np.memmap cache: chunk files are paged in by the OS, so
        multiple DataLoader workers share disk pages without duplicating RAM

    Args:
        data_dir: path to training_data/ (which must contain mmap/ subdir)
        chromosomes: set of chrom names like {"chr1", ...} to include;
                     None = include all
        augment: if True, apply reverse-complement augmentation
        label_remap: None, OR a 1-D numpy array mapping original label index
                     to new label index (len == original n_classes). Applied
                     at read time so no disk mutation.
    """

    def __init__(self, data_dir, chromosomes=None, augment=False,
                 label_remap=None, multilabel=False, multilabel_set="basic"):
        """
        Args:
            data_dir: training_data root
            chromosomes: set of chrom names to keep (None = all)
            augment: revcomp augmentation flag
            label_remap: optional 1-D int64 array mapping original 7-class
                index → remapped class index (for the 5-class scheme).
                Ignored when multilabel=True.
            multilabel: if True, load per-peak multi-label tensor and
                __getitem__ returns a float32 label vector of width k.
                If False, falls back to the multiclass int64 path.
            multilabel_set: which multi-label layout to load when
                multilabel=True:
                    "basic" → lab2_NNNN.npy (n,2) — is_enhancer, is_silencer
                    "full"  → lab4_NNNN.npy (n,4) — is_enhancer,
                              is_promoter, is_genic, is_silencer
        """
        self.data_dir = data_dir
        self.augment = augment
        self.multilabel = bool(multilabel)
        if multilabel_set not in ("basic", "full"):
            raise ValueError(
                f"multilabel_set must be 'basic' or 'full'; got {multilabel_set!r}"
            )
        self.multilabel_set = multilabel_set
        self._ml_prefix = "lab4" if multilabel_set == "full" else "lab2"
        self._ml_ncols = 4 if multilabel_set == "full" else 2
        self.mmap_dir = os.path.join(data_dir, "mmap")

        info_path = os.path.join(self.mmap_dir, "dataset_info_mmap.json")
        if not os.path.isfile(info_path):
            info_path = os.path.join(data_dir, "dataset_info.json")
        with open(info_path) as f:
            self.info = json.load(f)

        n_chunks = self.info["n_chunks"]
        self.chunk_size = self.info.get("chunk_size", 50000)
        self.seq_length = self.info.get("seq_length", 512)
        self.n_tfs = self.info.get("n_tfs", 879)
        self.n_tabular = 5

        # Optional label remap (numpy array for fast fancy-indexing).
        if label_remap is not None:
            self.label_remap = np.asarray(label_remap, dtype=np.int64)
        else:
            self.label_remap = None

        # Resolve chromosome-code filter. If no mmap chrom file exists (e.g.
        # running against the old .npz layout), fall through to the legacy
        # pandas metadata.csv path — but log a warning.
        chrom_code_map = self.info.get("chrom_code_map")
        if chromosomes is not None:
            if chrom_code_map is None:
                raise RuntimeError(
                    "dataset_info_mmap.json has no chrom_code_map — run "
                    "02b_convert_to_mmap.py before training."
                )
            allowed_codes = set()
            for name in chromosomes:
                if name in chrom_code_map:
                    allowed_codes.add(int(chrom_code_map[name]))
            self._allowed_codes = allowed_codes
        else:
            self._allowed_codes = None

        # ---- Detect repacked uint8 sequence layout (from 02d) ----
        # If seq8_NNNN.npy files exist, prefer them: 16× smaller than the
        # float32 one-hot, so the dataset fits much more comfortably in OS
        # page cache. We unpack to one-hot on the fly in __getitem__.
        sample_seq8 = os.path.join(self.mmap_dir, "seq8_0000.npy")
        self._use_seq8 = (
            self.info.get("has_seq8", False) and os.path.isfile(sample_seq8)
        )
        seq_prefix = "seq8" if self._use_seq8 else "seq"

        # ---- Detect bit-packed TF layout (from 02f) ----
        # tfb_NNNN.npy has shape (n, ceil(n_tfs/8)) uint8 — 32× smaller than
        # the float32 (n, n_tfs) layout. np.unpackbits at read time recovers
        # the original boolean mask, then cast to float32. Cheap (~5 µs/row).
        sample_tfb = os.path.join(self.mmap_dir, "tfb_0000.npy")
        self._use_tf_bits = (
            self.info.get("has_tf_bits", False) and os.path.isfile(sample_tfb)
        )
        tf_prefix = "tfb" if self._use_tf_bits else "tf"
        # Packed dim and the true n_tfs (for slicing off the pad bits).
        self._tf_packed_dim = self.info.get("tf_bits_packed_dim",
                                            (self.n_tfs + 7) // 8)

        # Per-chunk lookup arrays: only small per-chunk data lives in RAM.
        # self.index[i] = (chunk_idx, local_row_idx) for the i-th global row.
        # self.labels[i] = remapped label for the i-th global row.
        # Building the index reads only lab_* and chrom_* per chunk (~5 MB
        # per chunk), never the big seq/tf arrays.
        self._chunk_paths = {}
        index_chunks = []
        label_chunks = []
        n_total = 0
        for ci in range(n_chunks):
            seq_path = os.path.join(self.mmap_dir, f"{seq_prefix}_{ci:04d}.npy")
            tf_path = os.path.join(self.mmap_dir, f"{tf_prefix}_{ci:04d}.npy")
            tab_path = os.path.join(self.mmap_dir, f"tab_{ci:04d}.npy")
            lab_path = os.path.join(self.mmap_dir, f"lab_{ci:04d}.npy")
            chrom_path = os.path.join(self.mmap_dir, f"chrom_{ci:04d}.npy")
            if not all(os.path.isfile(p) for p in (seq_path, tf_path, tab_path, lab_path, chrom_path)):
                continue

            # Pick the label file:
            #   multilabel + basic  → lab2_*.npy (n,2) uint8
            #   multilabel + full   → lab4_*.npy (n,4) uint8
            #   multiclass          → lab_*.npy  (n,)  int64
            # 02e_build_multilabel.py produces lab2 / lab4.
            if self.multilabel:
                ml_path = os.path.join(
                    self.mmap_dir, f"{self._ml_prefix}_{ci:04d}.npy")
                if not os.path.isfile(ml_path):
                    continue
                labs = np.load(ml_path, mmap_mode="r")  # (n, k) uint8
            else:
                labs = np.load(lab_path, mmap_mode="r")  # (n,) int64

            if self._allowed_codes is not None:
                chroms = np.load(chrom_path, mmap_mode="r")  # (n,) uint8
                mask = np.isin(chroms, list(self._allowed_codes))
                local_idxs = np.nonzero(mask)[0].astype(np.int64)
                selected_labs = np.asarray(labs[local_idxs])
            else:
                local_idxs = np.arange(labs.shape[0], dtype=np.int64)
                selected_labs = np.asarray(labs[:])

            if not self.multilabel and self.label_remap is not None:
                selected_labs = self.label_remap[selected_labs]

            if len(local_idxs) == 0:
                continue

            ci_col = np.full(len(local_idxs), ci, dtype=np.int32)
            index_chunks.append(np.stack([ci_col, local_idxs.astype(np.int32)], axis=1))
            if self.multilabel:
                # Keep as uint8 here to save RAM; cast to float32 in __getitem__.
                label_chunks.append(selected_labs.astype(np.uint8, copy=False))
            else:
                label_chunks.append(selected_labs.astype(np.int64, copy=False))
            self._chunk_paths[ci] = {
                "seq": seq_path, "tf": tf_path, "tab": tab_path,
                "lab": (ml_path if self.multilabel else lab_path),
                "chrom": chrom_path,
            }
            n_total += len(local_idxs)

        if not index_chunks:
            raise RuntimeError("No chunks matched the given chromosome filter.")

        self.index = np.concatenate(index_chunks, axis=0)  # (N, 2) int32
        self.labels = np.concatenate(label_chunks, axis=0)  # (N,) int64
        assert len(self.index) == len(self.labels) == n_total

        # Per-worker mmap cache — lazily created on first __getitem__ in a
        # worker process, so each DataLoader worker has its own file handles
        # and doesn't share mmap pages via fork copy-on-write.
        self._mmaps = None

    def _ensure_mmaps(self):
        if self._mmaps is None:
            self._mmaps = {}
        return self._mmaps

    def _get_mmaps(self, ci):
        m = self._ensure_mmaps()
        if ci not in m:
            paths = self._chunk_paths[ci]
            m[ci] = {
                "seq": np.load(paths["seq"], mmap_mode="r"),
                "tf": np.load(paths["tf"], mmap_mode="r"),
                "tab": np.load(paths["tab"], mmap_mode="r"),
            }
        return m[ci]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        ci, local = self.index[idx]
        mm = self._get_mmaps(int(ci))
        # Copy the row out of the mmap so downstream ops don't hold a view
        # into the memory-mapped file.
        if self._use_seq8:
            # Repacked layout: seq is (L,) uint8 with values 0=A,1=C,2=G,3=T,
            # 255=N. Unpack to one-hot (L, 4) float32 in NumPy — cheap, ~1 µs
            # per row at L=512.
            base = np.array(mm["seq"][local], dtype=np.uint8, copy=True)  # (L,)
            seq = np.zeros((base.shape[0], 4), dtype=np.float32)
            valid = base < 4
            # one-hot for ACGT positions
            seq[valid, base[valid]] = 1.0
            # uniform 0.25 for N / unknown
            seq[~valid] = 0.25
        else:
            seq = np.array(mm["seq"][local], dtype=np.float32, copy=True)   # (L, 4)
        if self._use_tf_bits:
            # tfb is (n, packed_dim) uint8. Unpack along last axis, slice
            # off pad bits to get exactly n_tfs, cast to float32.
            packed_row = np.array(mm["tf"][local], dtype=np.uint8, copy=True)
            unpacked = np.unpackbits(packed_row)[:self.n_tfs]  # (n_tfs,) uint8
            tf_feat = unpacked.astype(np.float32, copy=False)
        else:
            tf_feat = np.array(mm["tf"][local], dtype=np.float32, copy=True)  # (n_tfs,)
        tab_feat = np.array(mm["tab"][local], dtype=np.float32, copy=True)  # (5,)

        if self.augment and np.random.random() < 0.5:
            seq = seq[::-1].copy()
            seq = seq[:, [3, 2, 1, 0]]  # A<->T, C<->G

        if self.multilabel:
            # self.labels is (N, 2) uint8; cast to float32 for BCEWithLogits
            label_t = torch.from_numpy(
                np.array(self.labels[idx], dtype=np.float32, copy=True)
            )
        else:
            label_t = torch.tensor(int(self.labels[idx]), dtype=torch.long)

        return (
            torch.from_numpy(seq),
            torch.from_numpy(tf_feat),
            torch.from_numpy(tab_feat),
            label_t,
        )


# ========== Focal Loss ==========

class FocalLoss(nn.Module):
    """Focal Loss for multi-class classification with class imbalance.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
        else:
            self.alpha = None

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.gamma

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_weight = alpha_t * focal_weight

        loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# ========== Model Architecture ==========

class ConvBlock(nn.Module):
    """1D Convolution + BatchNorm + ReLU + MaxPool block."""

    def __init__(self, in_channels, out_channels, kernel_size, pool_size=4):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding="same")
        self.bn = nn.BatchNorm1d(out_channels)
        self.pool = nn.MaxPool1d(pool_size)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.pool(x)
        return x


class RegulatoryClassifier(nn.Module):
    """Multi-modal regulatory region classifier.

    Architecture:
        Sequence branch: 3-layer 1D CNN → global avg+max pool → dense
        TF branch: dense on TF binding features + tabular features
        Fusion: concatenation → dense layers → classification head
    """

    def __init__(self, n_classes, n_tfs, seq_length=512, n_tabular=5,
                 conv_channels=(320, 256, 128), kernel_sizes=(8, 8, 4),
                 pool_sizes=(4, 4, 4), dropout=0.3):
        super().__init__()

        self.seq_length = seq_length
        self.n_tfs = n_tfs
        self.n_classes = n_classes

        # ---- Sequence branch (1D CNN) ----
        # Input: (batch, 4, seq_length) — channels-first for Conv1d
        layers = []
        in_ch = 4
        for out_ch, ks, ps in zip(conv_channels, kernel_sizes, pool_sizes):
            layers.append(ConvBlock(in_ch, out_ch, ks, ps))
            in_ch = out_ch
        self.seq_conv = nn.Sequential(*layers)

        # Global pooling: avg + max concatenated
        self.seq_pool_dim = conv_channels[-1] * 2  # avg + max

        self.seq_dense = nn.Sequential(
            nn.Linear(self.seq_pool_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ---- TF binding branch ----
        tf_input_dim = n_tfs + n_tabular
        self.tf_branch = nn.Sequential(
            nn.Linear(tf_input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ---- Fusion ----
        fusion_dim = 256 + 128  # seq_dense output + tf_branch output
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ---- Classification head ----
        self.classifier = nn.Linear(128, n_classes)

    def forward(self, seq, tf_feat, tab_feat):
        """
        Args:
            seq: (batch, seq_length, 4) — one-hot encoded DNA
            tf_feat: (batch, n_tfs) — binary TF presence
            tab_feat: (batch, n_tabular) — tabular features
        """
        # Sequence branch: permute to (batch, 4, seq_length) for Conv1d
        x_seq = seq.permute(0, 2, 1)
        x_seq = self.seq_conv(x_seq)  # (batch, last_conv_ch, reduced_len)

        # Global pooling
        avg_pool = x_seq.mean(dim=2)  # (batch, last_conv_ch)
        max_pool = x_seq.max(dim=2).values  # (batch, last_conv_ch)
        x_seq = torch.cat([avg_pool, max_pool], dim=1)  # (batch, last_conv_ch*2)

        x_seq = self.seq_dense(x_seq)  # (batch, 256)

        # TF branch
        x_tf = torch.cat([tf_feat, tab_feat], dim=1)  # (batch, n_tfs + n_tabular)
        x_tf = self.tf_branch(x_tf)  # (batch, 128)

        # Fusion
        x = torch.cat([x_seq, x_tf], dim=1)  # (batch, 384)
        x = self.fusion(x)  # (batch, 128)

        # Classification
        logits = self.classifier(x)  # (batch, n_classes)
        return logits

    def get_embeddings(self, seq, tf_feat, tab_feat):
        """Return the 128-dim fusion embeddings (before classifier)."""
        x_seq = seq.permute(0, 2, 1)
        x_seq = self.seq_conv(x_seq)
        avg_pool = x_seq.mean(dim=2)
        max_pool = x_seq.max(dim=2).values
        x_seq = torch.cat([avg_pool, max_pool], dim=1)
        x_seq = self.seq_dense(x_seq)

        x_tf = torch.cat([tf_feat, tab_feat], dim=1)
        x_tf = self.tf_branch(x_tf)

        x = torch.cat([x_seq, x_tf], dim=1)
        x = self.fusion(x)
        return x


# ========== Transformer-based Model ==========

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""

    def __init__(self, d_model, max_len=1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerClassifier(nn.Module):
    """Transformer-based regulatory region classifier.

    Architecture:
        Sequence: Linear projection of one-hot (4 -> d_model) + positional encoding
                  → N transformer encoder layers → [CLS] token pooling → dense
        TF branch: same as CNN model
        Fusion → classification
    """

    def __init__(self, n_classes, n_tfs, seq_length=512, n_tabular=5,
                 d_model=128, nhead=8, num_layers=4, dim_feedforward=512,
                 dropout=0.2):
        super().__init__()

        self.seq_length = seq_length
        self.n_classes = n_classes
        self.d_model = d_model

        # Sequence embedding: project 4-dim one-hot to d_model
        self.seq_embed = nn.Linear(4, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=seq_length + 1)

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.seq_norm = nn.LayerNorm(d_model)

        self.seq_dense = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # TF binding branch (same as CNN model)
        tf_input_dim = n_tfs + n_tabular
        self.tf_branch = nn.Sequential(
            nn.Linear(tf_input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Fusion
        fusion_dim = 256 + 128
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Linear(128, n_classes)

    def forward(self, seq, tf_feat, tab_feat):
        batch_size = seq.size(0)

        # Embed sequence: (batch, seq_len, 4) -> (batch, seq_len, d_model)
        x_seq = self.seq_embed(seq)

        # Prepend [CLS] token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x_seq = torch.cat([cls_tokens, x_seq], dim=1)  # (batch, seq_len+1, d_model)

        # Add positional encoding
        x_seq = self.pos_encoding(x_seq)

        # Transformer encoder
        x_seq = self.transformer(x_seq)
        x_seq = self.seq_norm(x_seq)

        # Take [CLS] token output
        x_cls = x_seq[:, 0]  # (batch, d_model)
        x_cls = self.seq_dense(x_cls)  # (batch, 256)

        # TF branch
        x_tf = torch.cat([tf_feat, tab_feat], dim=1)
        x_tf = self.tf_branch(x_tf)  # (batch, 128)

        # Fusion
        x = torch.cat([x_cls, x_tf], dim=1)
        x = self.fusion(x)

        logits = self.classifier(x)
        return logits

    def get_embeddings(self, seq, tf_feat, tab_feat):
        batch_size = seq.size(0)
        x_seq = self.seq_embed(seq)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x_seq = torch.cat([cls_tokens, x_seq], dim=1)
        x_seq = self.pos_encoding(x_seq)
        x_seq = self.transformer(x_seq)
        x_seq = self.seq_norm(x_seq)
        x_cls = x_seq[:, 0]
        x_cls = self.seq_dense(x_cls)
        x_tf = torch.cat([tf_feat, tab_feat], dim=1)
        x_tf = self.tf_branch(x_tf)
        x = torch.cat([x_cls, x_tf], dim=1)
        x = self.fusion(x)
        return x


# ========== Hybrid CNN-Transformer ==========

class HybridCNNTransformer(nn.Module):
    """Hybrid: CNN for local motif detection → Transformer for global context.

    The CNN extracts local motif features, then the transformer captures
    relationships between motifs across the sequence. This combines the
    strengths of both architectures.
    """

    def __init__(self, n_classes, n_tfs, seq_length=512, n_tabular=5,
                 d_model=128, nhead=8, num_layers=2, dropout=0.2):
        super().__init__()

        self.n_classes = n_classes

        # CNN front-end: extract local motif features
        self.cnn_front = nn.Sequential(
            nn.Conv1d(4, 128, kernel_size=8, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),  # seq_len/2
            nn.Conv1d(128, d_model, kernel_size=4, padding=2),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.MaxPool1d(2),  # seq_len/4
        )

        reduced_len = seq_length // 4
        self.pos_encoding = PositionalEncoding(d_model, max_len=reduced_len + 1)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.seq_norm = nn.LayerNorm(d_model)

        self.seq_dense = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # TF branch
        tf_input_dim = n_tfs + n_tabular
        self.tf_branch = nn.Sequential(
            nn.Linear(tf_input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(256 + 128, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Linear(128, n_classes)

    def forward(self, seq, tf_feat, tab_feat):
        batch_size = seq.size(0)

        # CNN: (batch, seq_len, 4) -> permute -> (batch, 4, seq_len)
        x = seq.permute(0, 2, 1)
        x = self.cnn_front(x)  # (batch, d_model, seq_len/4)

        # Permute for transformer: (batch, seq_len/4, d_model)
        x = x.permute(0, 2, 1)

        # Add [CLS] token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.pos_encoding(x)

        # Transformer
        x = self.transformer(x)
        x = self.seq_norm(x)
        x_cls = x[:, 0]
        x_cls = self.seq_dense(x_cls)

        # TF branch
        x_tf = torch.cat([tf_feat, tab_feat], dim=1)
        x_tf = self.tf_branch(x_tf)

        # Fusion
        x = torch.cat([x_cls, x_tf], dim=1)
        x = self.fusion(x)

        return self.classifier(x)

    def get_embeddings(self, seq, tf_feat, tab_feat):
        batch_size = seq.size(0)
        x = seq.permute(0, 2, 1)
        x = self.cnn_front(x)
        x = x.permute(0, 2, 1)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.pos_encoding(x)
        x = self.transformer(x)
        x = self.seq_norm(x)
        x_cls = x[:, 0]
        x_cls = self.seq_dense(x_cls)
        x_tf = torch.cat([tf_feat, tab_feat], dim=1)
        x_tf = self.tf_branch(x_tf)
        x = torch.cat([x_cls, x_tf], dim=1)
        x = self.fusion(x)
        return x


# ========== Model Registry ==========

MODEL_REGISTRY = {
    "cnn": RegulatoryClassifier,
    "transformer": TransformerClassifier,
    "hybrid": HybridCNNTransformer,
}


def build_model(dataset_info_path, model_type="cnn"):
    """Build model from dataset info.

    Args:
        dataset_info_path: path to dataset_info.json
        model_type: one of "cnn", "transformer", "hybrid"
    """
    with open(dataset_info_path) as f:
        info = json.load(f)

    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_type: {model_type}. Choose from {list(MODEL_REGISTRY.keys())}")

    cls = MODEL_REGISTRY[model_type]
    model = cls(
        n_classes=info["n_classes"],
        n_tfs=info["n_tfs"],
        seq_length=info["seq_length"],
    )
    return model, info
