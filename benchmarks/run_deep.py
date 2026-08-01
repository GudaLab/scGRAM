#!/usr/bin/env python3
"""
benchmarks/run_deep.py
======================
Deep-model benchmark dispatcher. Two phases per model:

  Phase 1 (embed):  pass every peak's 512-bp DNA string through the model's
                    frozen encoder; cache the per-peak embedding (fp16) to
                    per-chunk mmap files alongside the existing seq8/tfb.
                    Phase 1 is the expensive one (hours).

  Phase 2 (head):   train a small MLP head on (embedding + TF + tabular)
                    → 4 sigmoid outputs (is_enhancer, is_promoter, is_genic,
                    is_silencer). DDP across all GPUs. Fast (minutes/epoch
                    once embeddings cached).

Supported models (--model):
    basset      — Basset CNN architecture, trained from scratch (no HF DL)
    sei         — Sei (Chen 2022); off-the-shelf weights from HuggingFace
    dnabert2    — DNABERT-2-117M  (BPE-tokenized BERT)
    hyenadna    — HyenaDNA-medium-160k-seqlen
    nt          — Nucleotide-Transformer-v2-500M-multi-species
    caduceus    — Caduceus-PH (RC-equivariant Mamba)

Each model directory under benchmarks/results/<model>/ ends up holding:
    embeddings/embed_NNNN.npy   (n, d) fp16  one per chunk
    results.json                test metrics
    test_predictions.npz        raw probs + labels

Usage:
    python benchmarks/run_deep.py --model dnabert2 --phase embed
    python benchmarks/run_deep.py --model dnabert2 --phase head
    python benchmarks/run_deep.py --model dnabert2 --phase all
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (average_precision_score, f1_score, roc_auc_score)
from torch.utils.data import DataLoader, Dataset, Sampler


class ChunkSortedBatchSampler(Sampler):
    """Yield batches whose samples come from the same chunk_id, so workers
    do sequential mmap reads instead of NFS-random per-sample seeks.

    Strategy:
    - bucket sample indices by chunk_id
    - shuffle the order of chunks (epoch-level randomness)
    - within each chunk, shuffle samples
    - emit `batch_size` indices at a time, yielding partial batches at chunk
      boundaries so we keep coverage exact
    """
    def __init__(self, ci_arr, batch_size, shuffle=True, seed=0):
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        ci = np.asarray(ci_arr)
        order = np.argsort(ci, kind="stable")
        ci_sorted = ci[order]
        boundaries = np.concatenate(([0],
                                     np.nonzero(np.diff(ci_sorted))[0] + 1,
                                     [len(ci_sorted)]))
        self.chunk_groups = [order[boundaries[i]:boundaries[i + 1]]
                             for i in range(len(boundaries) - 1)]
        self._epoch = 0

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self._epoch)
        self._epoch += 1
        groups = self.chunk_groups
        order = rng.permutation(len(groups)) if self.shuffle else np.arange(len(groups))
        for gi in order:
            g = groups[gi]
            if self.shuffle:
                g = g[rng.permutation(len(g))]
            for s in range(0, len(g), self.batch_size):
                yield g[s:s + self.batch_size].tolist()

    def __len__(self):
        return sum((len(g) + self.batch_size - 1) // self.batch_size
                   for g in self.chunk_groups)

BASE = "/path/to/scgram"
MMAP = os.path.join(BASE, "training_data", "mmap")
INFO_PATH = os.path.join(MMAP, "dataset_info_mmap.json")
OUT_ROOT = os.path.join(BASE, "benchmarks", "results")

TRAIN_CHROMS = {f"chr{i}" for i in range(1, 18)}
VAL_CHROMS = {f"chr{i}" for i in range(18, 21)}
TEST_CHROMS = {"chr21", "chr22", "chrX"}
LABELS_FULL = ["is_enhancer", "is_promoter", "is_genic", "is_silencer"]

BASE_TO_CHAR = np.array(list("ACGTN" + "N" * 251), dtype="U1")  # uint8 → char

MODEL_HF = {
    "dnabert2":  ("zhihan1996/DNABERT-2-117M",                "trust_remote_code"),
    "hyenadna":  ("LongSafari/hyenadna-medium-160k-seqlen-hf","trust_remote_code"),
    "nt":        ("InstaDeepAI/nucleotide-transformer-v2-500m-multi-species", "trust_remote_code"),
    "caduceus":  ("kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16", "trust_remote_code"),
    "sei":       ("FunctionLab/sei",                          "trust_remote_code"),
    "basset":    (None, ""),  # trained from scratch
}


# ----------------------- chunk iteration -----------------------

def load_info():
    with open(INFO_PATH) as f:
        return json.load(f)


def chunks_for(info, chroms):
    code_map = info["chrom_code_map"]
    allowed = {int(code_map[c]) for c in chroms if c in code_map}
    n_chunks = info["n_chunks"]
    for ci in range(n_chunks):
        chrom_p = os.path.join(MMAP, f"chrom_{ci:04d}.npy")
        if not os.path.isfile(chrom_p):
            continue
        chrom = np.load(chrom_p, mmap_mode="r")
        mask = np.isin(chrom, list(allowed))
        if mask.any():
            yield ci, np.nonzero(mask)[0]


def seq8_to_strings(seq8_block):
    """(n, L) uint8 → list[str] of length-L DNA strings; N for >3."""
    arr = seq8_block.copy()
    arr[arr > 3] = 4   # map N to 'N' (index 4)
    chars = BASE_TO_CHAR[arr]    # (n, L) U1
    return ["".join(row) for row in chars]


# ----------------------- model loaders -----------------------

class BassetEncoder(nn.Module):
    """Basset (Kelley et al 2016) sequence encoder: 3 conv blocks + 2 FC.
    Outputs a 512-d embedding. Trained from scratch — used as a SCRATCH-CNN
    baseline against our own RegulatoryClassifier architecture."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(4,  300, 19), nn.BatchNorm1d(300), nn.ReLU(), nn.MaxPool1d(3),
            nn.Conv1d(300, 200, 11), nn.BatchNorm1d(200), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(200, 200, 7),  nn.BatchNorm1d(200), nn.ReLU(), nn.MaxPool1d(4),
            nn.Flatten(),
            # Conv stack on L=512: 512→494→164(MP3)→154→38(MP4)→32→8(MP4)
            # Output: (B, 200, 8) → Flatten → (B, 1600). Previous value 2000
            # was wrong (off by 400). Verified via shape error from the run.
            nn.Linear(1600, 1000), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(1000, 512),  nn.ReLU(),
        )
        self.embed_dim = 512

    def forward_seq_onehot(self, x):
        # x: (B, L, 4) → (B, 4, L) for Conv1d
        if x.shape[1] == 4 and x.shape[2] != 4:
            return self.net(x)
        return self.net(x.transpose(1, 2))


class SeiEncoder(nn.Module):
    """Sei (Chen 2022) encoder up through dconv5, mean-pooled to 960-d.

    The published architecture takes 4096-bp one-hot input and outputs
    21,907 chromatin-feature predictions via a huge classifier head. We
    drop the spline+classifier and use the post-dconv5 feature map
    (B, 960, L'), mean-pooled over L' as a 960-d embedding.

    For our 512-bp window: zero-pad to 4096 on both sides so the
    convolutional receptive field still operates at the design width.
    """
    def __init__(self, sei_pth_path):
        super().__init__()
        self.lconv1 = nn.Sequential(
            nn.Conv1d(4, 480, 9, padding=4),
            nn.Conv1d(480, 480, 9, padding=4))
        self.conv1 = nn.Sequential(
            nn.Conv1d(480, 480, 9, padding=4), nn.ReLU(inplace=True),
            nn.Conv1d(480, 480, 9, padding=4), nn.ReLU(inplace=True))
        self.lconv2 = nn.Sequential(
            nn.MaxPool1d(4, 4), nn.Dropout(0.2),
            nn.Conv1d(480, 640, 9, padding=4),
            nn.Conv1d(640, 640, 9, padding=4))
        self.conv2 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Conv1d(640, 640, 9, padding=4), nn.ReLU(inplace=True),
            nn.Conv1d(640, 640, 9, padding=4), nn.ReLU(inplace=True))
        self.lconv3 = nn.Sequential(
            nn.MaxPool1d(4, 4), nn.Dropout(0.2),
            nn.Conv1d(640, 960, 9, padding=4),
            nn.Conv1d(960, 960, 9, padding=4))
        self.conv3 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Conv1d(960, 960, 9, padding=4), nn.ReLU(inplace=True),
            nn.Conv1d(960, 960, 9, padding=4), nn.ReLU(inplace=True))
        self.dconv1 = nn.Sequential(
            nn.Dropout(0.10),
            nn.Conv1d(960, 960, 5, dilation=2, padding=4), nn.ReLU(inplace=True))
        self.dconv2 = nn.Sequential(
            nn.Dropout(0.10),
            nn.Conv1d(960, 960, 5, dilation=4, padding=8), nn.ReLU(inplace=True))
        self.dconv3 = nn.Sequential(
            nn.Dropout(0.10),
            nn.Conv1d(960, 960, 5, dilation=8, padding=16), nn.ReLU(inplace=True))
        self.dconv4 = nn.Sequential(
            nn.Dropout(0.10),
            nn.Conv1d(960, 960, 5, dilation=16, padding=32), nn.ReLU(inplace=True))
        self.dconv5 = nn.Sequential(
            nn.Dropout(0.10),
            nn.Conv1d(960, 960, 5, dilation=25, padding=50), nn.ReLU(inplace=True))
        self.embed_dim = 960
        self._load_pretrained(sei_pth_path)

    def _load_pretrained(self, pth):
        # Sei checkpoint uses keys like "module.model.lconv1.0.weight"
        raw = torch.load(pth, map_location="cpu")
        if isinstance(raw, dict) and "state_dict" in raw:
            raw = raw["state_dict"]
        sd = {}
        for k, v in raw.items():
            if k.startswith("module.model."):
                k2 = k[len("module.model."):]
            elif k.startswith("module."):
                k2 = k[len("module."):]
            else:
                k2 = k
            # only keep encoder layers (skip classifier + spline)
            if k2.startswith(("lconv", "conv", "dconv")):
                sd[k2] = v
        missing, unexpected = self.load_state_dict(sd, strict=False)
        # Expected missing: nothing in encoder (dropout/maxpool have no params)
        # Expected unexpected: empty if we filtered correctly

    def forward_seq_onehot(self, x):
        # x: (B, L, 4) → (B, 4, L); pad to 4096 with zeros (matches N input)
        if x.shape[1] == 4 and x.shape[2] != 4:
            x_bcL = x
        else:
            x_bcL = x.transpose(1, 2)  # (B, 4, L)
        B, C, L = x_bcL.shape
        if L < 4096:
            pad = 4096 - L
            left = pad // 2
            right = pad - left
            x_bcL = nn.functional.pad(x_bcL, (left, right), value=0.0)
        elif L > 4096:
            x_bcL = x_bcL[:, :, :4096]
        # Encoder forward
        lout1 = self.lconv1(x_bcL)
        out1 = self.conv1(lout1)
        lout2 = self.lconv2(out1 + lout1)
        out2 = self.conv2(lout2)
        lout3 = self.lconv3(out2 + lout2)
        out3 = self.conv3(lout3)
        d1 = self.dconv1(out3 + lout3)
        c1 = out3 + d1
        d2 = self.dconv2(c1)
        c2 = c1 + d2
        d3 = self.dconv3(c2)
        c3 = c2 + d3
        d4 = self.dconv4(c3)
        c4 = c3 + d4
        d5 = self.dconv5(c4)
        out = c4 + d5  # (B, 960, L')
        # global mean pool over spatial dim
        return out.mean(dim=2)  # (B, 960)


SEI_PTH = "/path/to/scgram/benchmarks/sei_src/model/sei.pth"


def load_encoder(model_name, device):
    """Returns (encoder, embed_dim, tokenizer_or_None, takes_strings_bool).

    Raises RuntimeError on load failure (e.g. Sei doesn't always expose a
    clean AutoModel interface — the SLURM driver catches and skips).
    """
    if model_name == "basset":
        enc = BassetEncoder().to(device).eval()
        return enc, enc.embed_dim, None, False
    if model_name == "sei":
        enc = SeiEncoder(SEI_PTH).to(device).eval()
        return enc, enc.embed_dim, None, False
    try:
        from transformers import AutoModel, AutoTokenizer, AutoConfig
        hf_id, flag = MODEL_HF[model_name]
        kw = {"trust_remote_code": True} if flag else {}
        tok = AutoTokenizer.from_pretrained(hf_id, **kw)
        # DNABERT-2's custom modeling code references config.pad_token_id,
        # but its config.json (BertConfig) doesn't define one. Recent
        # transformers versions raise AttributeError instead of returning
        # None. Patch the config before loading the model.
        if model_name == "dnabert2":
            # DNABERT-2's custom modeling ships a triton flash-attention
            # kernel that uses tl.dot(...,trans_b=True), removed in triton
            # ≥ 2.1. We run dnabert2 in a separate venv (dnabert2-env) with
            # torch 2.0.1 / transformers 4.29 / triton 2.0. The kernel
            # itself still won't compile (newer triton in pip's wheel cache),
            # but the modeling code has a fallback path: if
            # bert_layers.flash_attn_qkvpacked_func is None, it uses
            # vanilla scaled_dot_product attention. Monkey-patch the loaded
            # module to force that fallback. Also patch pad_token_id.
            cfg = AutoConfig.from_pretrained(hf_id, **kw)
            if not hasattr(cfg, "pad_token_id") or cfg.pad_token_id is None:
                cfg.pad_token_id = getattr(tok, "pad_token_id", 0) or 0
            mdl = AutoModel.from_pretrained(hf_id, config=cfg, **kw).to(device).eval()
            import sys
            for _name, _mod in sys.modules.items():
                if "bert_layers" in _name and hasattr(_mod, "flash_attn_qkvpacked_func"):
                    _mod.flash_attn_qkvpacked_func = None
        elif model_name == "caduceus":
            # Caduceus-PH uses bidirectional Mamba with weight tying between
            # mamba_fwd and mamba_rev. The tying is set up at __init__ but
            # transformers' load_state_dict replaces parameter tensors, which
            # breaks the shared-reference tying. Result: mamba_rev weights
            # end up as the random init from __init__, not tied to mamba_fwd
            # as the config dictates. Call the model's tie_weights() method
            # after load to re-establish the tying.
            mdl = AutoModel.from_pretrained(hf_id, **kw).to(device).eval()
            if hasattr(mdl, "tie_weights"):
                mdl.tie_weights()
            elif hasattr(mdl, "maybe_weight_tie_mamba"):
                mdl.maybe_weight_tie_mamba()
        elif model_name == "nt":
            # NT v2-500m's auto_map registers AutoModelForMaskedLM (custom
            # EsmForMaskedLM with SwiGLU-style FFN) but NOT AutoModel.
            # Using AutoModel falls back to the generic EsmModel which
            # has the WRONG FFN size — 29 encoder layers reinit randomly.
            # Use the custom MaskedLM class then extract its `.esm`
            # encoder. Also stub `find_pruneable_heads_and_indices` if
            # the running transformers version dropped it.
            import transformers.pytorch_utils as _pu
            if not hasattr(_pu, "find_pruneable_heads_and_indices"):
                import torch as _t
                _pu.find_pruneable_heads_and_indices = lambda *a, **kw: (
                    set(), _t.tensor([], dtype=_t.long))
            from transformers import AutoModelForMaskedLM as _AM4MLM
            _full = _AM4MLM.from_pretrained(hf_id, **kw).to(device).eval()
            mdl = _full.esm  # base encoder, returns last_hidden_state
        else:
            mdl = AutoModel.from_pretrained(hf_id, **kw).to(device).eval()
    except Exception as e:
        raise RuntimeError(f"[{model_name}] failed to load from HF: {e}") from e
    for p in mdl.parameters():
        p.requires_grad_(False)
    # Probe embedding dim with a single dummy forward
    with torch.no_grad():
        sample = tok("ACGT" * 10, return_tensors="pt").to(device)
        out = mdl(**sample)
        emb = (out.last_hidden_state if hasattr(out, "last_hidden_state")
               else out[0])
        embed_dim = int(emb.shape[-1])
    return mdl, embed_dim, tok, True


# ----------------------- phase 1: embed -----------------------

def phase_embed(args, info):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = os.path.join(OUT_ROOT, args.model, "embeddings")
    os.makedirs(out_dir, exist_ok=True)
    enc, embed_dim, tok, takes_strings = load_encoder(args.model, device)
    print(f"[{args.model}] embed_dim={embed_dim} on {device}")

    chunks_done = sorted(int(f.split("_")[1].split(".")[0])
                         for f in os.listdir(out_dir) if f.startswith("embed_"))
    print(f"  {len(chunks_done)} chunks already embedded; resuming")

    BS = args.embed_batch
    SPLITS = TRAIN_CHROMS | VAL_CHROMS | TEST_CHROMS
    # Only keep chunks that contain ≥1 SPLITS peak (skip pure-chrY chunks).
    ci_iter = list(chunks_for(info, SPLITS))
    # Multi-GPU sharding: each worker keeps chunks where ci%shard_mod==rank.
    # All workers write to the same out_dir; resume covers any overlap.
    if args.shard_mod > 1:
        ci_iter = [(ci, idxs) for (ci, idxs) in ci_iter
                   if (ci % args.shard_mod) == args.shard_rank]
        print(f"  shard {args.shard_rank}/{args.shard_mod}: "
              f"{len(ci_iter)} chunks assigned")
    t0 = time.time()
    for n_seen, (ci, _idxs_unused) in enumerate(ci_iter, 1):
        outp = os.path.join(out_dir, f"embed_{ci:04d}.npy")
        if os.path.isfile(outp):
            continue
        # CRITICAL: embed ALL peaks in the chunk, NOT just SPLITS-filtered
        # ones. Then EmbedDataset indexes by `local_idx` (the peak's position
        # within the chunk) which matches the TF/lab files. Filtering by
        # SPLITS here while EmbedDataset filters by TRAIN/VAL/TEST would
        # silently misalign embeddings with labels — every per-label AUROC
        # would be near-random.
        seq8 = np.array(np.load(os.path.join(MMAP, f"seq8_{ci:04d}.npy"),
                                mmap_mode="r"))   # (n_chunk, L) uint8
        n_in = seq8.shape[0]
        embs = np.zeros((n_in, embed_dim), dtype=np.float16)
        s = 0
        while s < n_in:
            cur_bs = BS
            while True:
                batch = seq8[s:s + cur_bs]
                try:
                    if takes_strings:
                        strs = seq8_to_strings(batch)
                        enc_in = tok(strs, return_tensors="pt", padding=True,
                                     truncation=True, max_length=512).to(device)
                        # bf16 autocast halves activation memory and ~2x
                        # forward throughput on A6000 — essential for HF
                        # foundation models without flash-attn (vanilla
                        # attention at L=512 is bandwidth-bound in fp32).
                        # bf16 chosen over fp16 because HyenaDNA's conv
                        # layers throw CUDNN_STATUS_NOT_INITIALIZED with
                        # fp16 inputs; dnabert2/NT work fine with either.
                        with torch.no_grad(), torch.autocast(
                                device_type="cuda", dtype=torch.bfloat16):
                            out = enc(**enc_in)
                            hs = (out.last_hidden_state
                                  if hasattr(out, "last_hidden_state")
                                  else out[0])
                            # Masked mean-pool — padding tokens dilute a plain
                            # mean and bias embeddings for variable-length
                            # batches (DNABERT-2 BPE produces unequal lengths).
                            am = enc_in.get("attention_mask")
                            if am is not None:
                                m = am.unsqueeze(-1).to(hs.dtype)
                                emb = (hs * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
                            else:
                                emb = hs.mean(dim=1)
                    else:
                        # Basset: one-hot
                        oh = np.zeros((len(batch), batch.shape[1], 4),
                                      dtype=np.float32)
                        valid = batch < 4
                        rows, cols = np.nonzero(valid)
                        oh[rows, cols, batch[valid].astype(np.int64)] = 1.0
                        oh[~valid] = 0.25
                        with torch.no_grad():
                            emb = enc.forward_seq_onehot(
                                torch.from_numpy(oh).to(device))
                    # .float() first because bf16 autocast can leave emb as
                    # bf16, which .numpy() doesn't support (caduceus path).
                    embs[s:s + cur_bs] = emb.detach().float().cpu().numpy().astype(np.float16)
                    s += cur_bs
                    break
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if cur_bs <= 1:
                        raise
                    cur_bs = max(1, cur_bs // 2)
                    print(f"  OOM at bs={cur_bs*2}; halving to {cur_bs}",
                          flush=True)
        tmp = outp + ".inflight.npy"
        np.save(tmp, embs); os.replace(tmp, outp)
        if n_seen % 20 == 0:
            elapsed = time.time() - t0
            rate = n_seen / max(elapsed, 1)
            eta_min = (len(ci_iter) - n_seen) / max(rate, 1e-6) / 60
            print(f"  embedded {n_seen}/{len(ci_iter)}  "
                  f"({rate:.2f} chunks/s, ETA {eta_min:.1f} min)", flush=True)
    print(f"[{args.model}] embeddings complete in {(time.time()-t0)/60:.1f} min")


# ----------------------- phase 2: head -----------------------

class EmbedDataset(Dataset):
    """Streams (embedding, TF, tab, lab4) per peak, filtered by chrom.

    phase_embed writes (chunk_size, d) embeddings — every row in the chunk,
    NOT chrom-filtered. So we index by `local_idx` (peak's position within
    the chunk), which is the same index used for the TF / tab / lab files.
    No more kth_arr — that was the silent-misalignment bug.
    """
    def __init__(self, model_name, chroms, info, label_set="full"):
        self.model_name = model_name
        self.lab_prefix = "lab4" if label_set == "full" else "lab2"
        self.n_tfs = info["n_tfs"]
        chunks = list(chunks_for(info, chroms))
        total = sum(len(idxs) for _, idxs in chunks)
        ci_arr = np.empty(total, dtype=np.int32)
        local_arr = np.empty(total, dtype=np.int32)
        off = 0
        for ci, idxs in chunks:
            n = len(idxs)
            ci_arr[off:off + n] = ci
            local_arr[off:off + n] = idxs.astype(np.int32)
            off += n
        self.ci_arr = ci_arr
        self.local_arr = local_arr
        self._cache = {}
        self.emb_root = os.path.join(OUT_ROOT, model_name, "embeddings")

    def __len__(self):
        return len(self.ci_arr)

    def _mm(self, ci):
        if ci not in self._cache:
            # Bound memory: evict oldest cached chunk when over MAX_CACHED
            # (NT-500M chunks are ~102 MB each; 8 cached × N workers can
            # easily blow past 1 TB RAM and trigger swap thrash).
            MAX_CACHED = 32
            if len(self._cache) >= MAX_CACHED:
                oldest = next(iter(self._cache))
                del self._cache[oldest]
            self._cache[ci] = {
                "emb": np.load(os.path.join(self.emb_root, f"embed_{ci:04d}.npy"),
                               mmap_mode="r"),
                "tfb": np.load(os.path.join(MMAP, f"tfb_{ci:04d}.npy"),
                               mmap_mode="r"),
                "tab": np.load(os.path.join(MMAP, f"tab_{ci:04d}.npy"),
                               mmap_mode="r"),
                "lab": np.load(os.path.join(MMAP, f"{self.lab_prefix}_{ci:04d}.npy"),
                               mmap_mode="r"),
            }
        return self._cache[ci]

    def __getitem__(self, i):
        ci = int(self.ci_arr[i])
        local = int(self.local_arr[i])
        mm = self._mm(ci)
        emb = np.asarray(mm["emb"][local], dtype=np.float32)
        tfb_row = np.asarray(mm["tfb"][local], dtype=np.uint8)
        tf = np.unpackbits(tfb_row)[:self.n_tfs].astype(np.float32)
        tab = np.asarray(mm["tab"][local], dtype=np.float32)
        lab = np.asarray(mm["lab"][local], dtype=np.float32)
        return (torch.from_numpy(emb), torch.from_numpy(tf),
                torch.from_numpy(tab), torch.from_numpy(lab))


class HeadMLP(nn.Module):
    def __init__(self, embed_dim, n_tfs, n_tab, n_out, hidden=512):
        super().__init__()
        self.seq_branch = nn.Sequential(
            nn.Linear(embed_dim, hidden), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(hidden, 128))
        self.tf_branch = nn.Sequential(
            nn.Linear(n_tfs, 256), nn.GELU(),
            nn.Linear(256, 128), nn.GELU())
        self.fuse = nn.Sequential(
            nn.Linear(128 + 128 + n_tab, 256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, n_out))

    def forward(self, emb, tf, tab):
        return self.fuse(torch.cat([self.seq_branch(emb),
                                    self.tf_branch(tf), tab], dim=1))


def eval_multilabel(probs, labels, label_list):
    per = {}
    for c, name in enumerate(label_list):
        y = labels[:, c]; p = probs[:, c]
        pred = (p >= 0.5).astype(np.uint8)
        per[name] = {
            "auroc": float(roc_auc_score(y, p)) if y.sum() and (1-y).sum() else float("nan"),
            "auprc": float(average_precision_score(y, p)),
            "f1_at_0p5": float(f1_score(y, pred, zero_division=0)),
            "n_pos": int(y.sum()), "n_total": int(len(y)),
        }
    return {
        "per_label": per,
        "macro_auroc": float(np.nanmean([per[n]["auroc"] for n in label_list])),
        "macro_auprc": float(np.nanmean([per[n]["auprc"] for n in label_list])),
        "macro_f1": float(np.nanmean([per[n]["f1_at_0p5"] for n in label_list])),
    }


def phase_head(args, info):
    device = torch.device("cuda")
    out_dir = os.path.join(OUT_ROOT, args.model)
    os.makedirs(out_dir, exist_ok=True)

    # Probe embed_dim from a cached chunk
    emb_root = os.path.join(out_dir, "embeddings")
    sample = next(f for f in os.listdir(emb_root) if f.startswith("embed_"))
    embed_dim = int(np.load(os.path.join(emb_root, sample), mmap_mode="r").shape[1])
    print(f"[{args.model}/head] embed_dim={embed_dim}")

    n_out = 4 if args.label_set == "full" else 2
    labels = LABELS_FULL if args.label_set == "full" else ["is_enhancer", "is_silencer"]

    ds_tr = EmbedDataset(args.model, TRAIN_CHROMS, info, args.label_set)
    ds_va = EmbedDataset(args.model, VAL_CHROMS, info, args.label_set)
    ds_te = EmbedDataset(args.model, TEST_CHROMS, info, args.label_set)
    print(f"  train={len(ds_tr):,} val={len(ds_va):,} test={len(ds_te):,}")

    # NFS random-access kills throughput at default settings. Use a
    # ChunkSortedBatchSampler so each batch's samples come from the same
    # mmap'd chunk file (sequential reads instead of 4096-way per-batch
    # random seeks). Workers are persistent so per-chunk caches survive
    # across epochs.
    dl_kw = dict(num_workers=args.workers, pin_memory=True,
                 prefetch_factor=4, persistent_workers=(args.workers > 0))
    tr_sampler = ChunkSortedBatchSampler(ds_tr.ci_arr, args.head_batch,
                                         shuffle=True, seed=0)
    va_sampler = ChunkSortedBatchSampler(ds_va.ci_arr, args.head_batch,
                                         shuffle=False)
    te_sampler = ChunkSortedBatchSampler(ds_te.ci_arr, args.head_batch,
                                         shuffle=False)
    dl_tr = DataLoader(ds_tr, batch_sampler=tr_sampler, **dl_kw)
    dl_va = DataLoader(ds_va, batch_sampler=va_sampler, **dl_kw)
    dl_te = DataLoader(ds_te, batch_sampler=te_sampler, **dl_kw)

    # Seeds for reproducibility
    torch.manual_seed(0); np.random.seed(0)
    model = HeadMLP(embed_dim, info["n_tfs"], 5, n_out).to(device)

    # Per-label pos_weight from the WHOLE train set — read labels directly
    # via mmap (cheap: ~5 bytes/peak), skipping the DataLoader's embedding/
    # TF decode. Previous version iterated DataLoader and broke at 5M rows;
    # could bias pos_weight by chunk order.
    counts = np.zeros(n_out, dtype=np.int64); n_tot = 0
    for ci, idxs in chunks_for(info, TRAIN_CHROMS):
        lab = np.asarray(np.load(
            os.path.join(MMAP, f"{ds_tr.lab_prefix}_{ci:04d}.npy"),
            mmap_mode="r")[idxs], dtype=np.uint8)
        counts += lab.sum(axis=0).astype(np.int64); n_tot += len(lab)
    pw = np.clip([(n_tot - c) / max(c, 1) for c in counts],
                 0, 5.0).astype(np.float32)
    print(f"  pos_weight (capped@5, full train n={n_tot:,}): {pw.tolist()}")
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.from_numpy(pw).to(device))

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_val = float("inf"); patience = 0

    for ep in range(1, args.epochs + 1):
        model.train(); tot = n = 0
        for emb, tf, tab, lab in dl_tr:
            emb, tf, tab, lab = [t.to(device, non_blocking=True)
                                 for t in (emb, tf, tab, lab)]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(emb, tf, tab)
                loss = crit(logits, lab)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            tot += loss.item() * len(lab); n += len(lab)
        train_loss = tot / max(n, 1)

        model.eval()
        with torch.no_grad():
            vlss = vn = 0
            for emb, tf, tab, lab in dl_va:
                emb, tf, tab, lab = [t.to(device) for t in (emb, tf, tab, lab)]
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(emb, tf, tab)
                vlss += crit(logits, lab).item() * len(lab); vn += len(lab)
            val_loss = vlss / max(vn, 1)
        print(f"  ep{ep}/{args.epochs}  train={train_loss:.4f}  val={val_loss:.4f}",
              flush=True)
        # High-confidence early stop: improvement must exceed min_delta
        if val_loss < best_val - args.min_delta:
            best_val = val_loss; patience = 0
            # Atomic save — guards against partial-write corruption on kill
            head_path = os.path.join(out_dir, "head.pt")
            torch.save(model.state_dict(), head_path + ".tmp")
            os.replace(head_path + ".tmp", head_path)
        else:
            patience += 1
            if patience >= args.patience:
                print(f"  early stop @ ep{ep} "
                      f"(best val={best_val:.4f})"); break

    # Test — load best checkpoint (weights_only avoids the pickle warning
    # in newer PyTorch and limits attack surface)
    model.load_state_dict(torch.load(os.path.join(out_dir, "head.pt"),
                                     weights_only=True))
    model.eval()
    all_p, all_y = [], []
    with torch.no_grad():
        for emb, tf, tab, lab in dl_te:
            emb, tf, tab = [t.to(device) for t in (emb, tf, tab)]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                p = torch.sigmoid(model(emb, tf, tab)).float().cpu().numpy()
            all_p.append(p); all_y.append(lab.numpy())
    probs = np.concatenate(all_p); ys = np.concatenate(all_y).astype(np.uint8)
    metrics = eval_multilabel(probs, ys, labels)
    metrics.update(model=args.model, label_set=args.label_set,
                   n_train=len(ds_tr), n_test=len(ds_te))
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    np.savez(os.path.join(out_dir, "test_predictions.npz"),
             probs=probs.astype(np.float32), labels=ys)
    print(f"[{args.model}] Macro AUROC={metrics['macro_auroc']:.4f}  "
          f"AUPRC={metrics['macro_auprc']:.4f}  F1={metrics['macro_f1']:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_HF.keys()))
    ap.add_argument("--phase", default="all", choices=["embed", "head", "all"])
    ap.add_argument("--label-set", default="full", choices=["basic", "full"])
    ap.add_argument("--embed-batch", type=int, default=64)
    ap.add_argument("--head-batch", type=int, default=4096)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--shard-rank", type=int, default=0,
                    help="this shard's rank (0..shard-mod-1); for multi-GPU split")
    ap.add_argument("--shard-mod", type=int, default=1,
                    help="number of shards; each shard does chunks where ci%shard-mod==rank")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--min-delta", type=float, default=1e-3,
                    help="Minimum val-loss decrease that counts as an "
                         "improvement (reviewer-recommended high-confidence "
                         "early stop). Default 1e-3 with patience 6 means "
                         "we stop only after 6 consecutive epochs whose "
                         "val-loss didn't drop by ≥1e-3.")
    args = ap.parse_args()
    info = load_info()
    if args.phase in ("embed", "all"):
        phase_embed(args, info)
    if args.phase in ("head", "all"):
        phase_head(args, info)


if __name__ == "__main__":
    main()
