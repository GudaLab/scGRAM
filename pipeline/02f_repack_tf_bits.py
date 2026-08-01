#!/usr/bin/env python3
"""
02f_repack_tf_bits.py
======================
Repack per-chunk TF feature arrays from float32 (n, n_tfs) to bit-packed
uint8 (n, ceil(n_tfs/8)). TF features are binary 0/1, so 1 bit/value is
lossless. With n_tfs=879 this shrinks each chunk from 168 MB → 5.5 MB
(32× smaller).

Total dataset I/O shrinkage:
    Before: seq8 (264 GB) + tf float32 (1.77 TB) + small = ~2 TB
    After:  seq8 (264 GB) + tf bits (58 GB)  + small = ~322 GB
    → 6.4× smaller, fits comfortably in 944 GB OS page cache.

Idempotent. Safe to re-run while training is offline.

Outputs in training_data/mmap/:
    tfb_NNNN.npy  shape (n, ceil(n_tfs/8))  dtype uint8  ~5.5 MB each

Updates dataset_info_mmap.json: sets has_tf_bits=true.

Usage:
    python 02f_repack_tf_bits.py [--workers 20]
"""

import argparse
import glob
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

BASE_DIR = "/path/to/data"
TRAINING_DIR = os.path.join(BASE_DIR, "unbound_characetrize", "training_data")
MMAP_DIR = os.path.join(TRAINING_DIR, "mmap")
DATASET_INFO_MMAP = os.path.join(MMAP_DIR, "dataset_info_mmap.json")


def atomic_write_npy(path, arr):
    if not path.endswith(".npy"):
        raise ValueError(f"expected .npy path; got {path}")
    tmp = path[:-4] + ".inflight.npy"
    np.save(tmp, arr, allow_pickle=False)
    os.replace(tmp, path)


def is_chunk_done(path, n_expected, packed_dim):
    if not os.path.isfile(path):
        return False
    try:
        from numpy.lib.format import (
            read_magic, read_array_header_1_0, read_array_header_2_0,
        )
        with open(path, "rb") as fh:
            v = read_magic(fh)
            hdr = read_array_header_1_0(fh) if v == (1, 0) else read_array_header_2_0(fh)
        return hdr[0] == (n_expected, packed_dim)
    except Exception:
        return False


def repack_one(task):
    """Read tf_NNNN.npy (n, n_tfs) float32, bit-pack along the n_tfs axis,
    write tfb_NNNN.npy (n, ceil(n_tfs/8)) uint8."""
    ci, src_path, dst_path = task
    arr = np.load(src_path, mmap_mode="r")  # (n, n_tfs) float32
    n, n_tfs = arr.shape
    # Binary: 0 or 1. Convert to uint8 explicitly so packbits doesn't see
    # nonzero floats as anomalies.
    bool_arr = (arr > 0.5).astype(np.uint8)
    # packbits along the last axis. Pads the trailing axis with zeros to a
    # multiple of 8; the unpack at read time slices back to n_tfs.
    packed = np.packbits(bool_arr, axis=1)  # (n, ceil(n_tfs/8))
    atomic_write_npy(dst_path, np.ascontiguousarray(packed, dtype=np.uint8))
    return ci, n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()

    if not os.path.isfile(DATASET_INFO_MMAP):
        sys.exit(f"missing {DATASET_INFO_MMAP}; run 02b_convert_to_mmap.py first")
    with open(DATASET_INFO_MMAP) as f:
        info = json.load(f)
    n_chunks = info["n_chunks"]
    n_tfs = info["n_tfs"]
    chunk_size = info["chunk_size"]
    packed_dim = (n_tfs + 7) // 8
    print(f"Dataset: {n_chunks:,} chunks, n_tfs={n_tfs} → packed_dim={packed_dim}")
    print(f"Per-chunk tf bytes before: {n_tfs * 4} → after: {packed_dim} "
          f"(~{n_tfs * 4 / packed_dim:.1f}× shrink)")

    # Sweep stale staging files
    n_stale = 0
    for pat in ("tfb_*.inflight.npy", "tfb_*.npy.tmp", "tfb_*.npy.tmp.npy"):
        for stale in glob.glob(os.path.join(MMAP_DIR, pat)):
            os.remove(stale)
            n_stale += 1
    if n_stale:
        print(f"Removed {n_stale} stale staging files.")

    print(f"\n[1/2] Scanning {n_chunks} chunks...")
    tasks = []
    skipped = 0
    missing = 0
    for ci in range(n_chunks):
        src = os.path.join(MMAP_DIR, f"tf_{ci:04d}.npy")
        dst = os.path.join(MMAP_DIR, f"tfb_{ci:04d}.npy")
        if not os.path.isfile(src):
            missing += 1
            continue
        # Pull row count from src header (last chunk may have < chunk_size).
        try:
            from numpy.lib.format import (
                read_magic, read_array_header_1_0, read_array_header_2_0,
            )
            with open(src, "rb") as fh:
                v = read_magic(fh)
                hdr = read_array_header_1_0(fh) if v == (1, 0) else read_array_header_2_0(fh)
            n_in_src = hdr[0][0]
        except Exception:
            n_in_src = chunk_size
        if is_chunk_done(dst, n_in_src, packed_dim):
            skipped += 1
            continue
        tasks.append((ci, src, dst))
    print(f"  already packed: {skipped}")
    print(f"  source missing: {missing}")
    print(f"  to pack:        {len(tasks)}")

    if tasks:
        print(f"\n[2/2] Bit-packing with {args.workers} workers...")
        t0 = time.time()
        done = 0
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(repack_one, t): t[0] for t in tasks}
            for fut in as_completed(futures):
                ci, n = fut.result()
                done += 1
                if done % 200 == 0 or done == len(tasks):
                    elapsed = time.time() - t0
                    rate = done / max(elapsed, 1e-6)
                    eta = (len(tasks) - done) / max(rate, 1e-6)
                    print(f"  {done}/{len(tasks)} packed "
                          f"({rate:.1f} chunks/s, ETA {eta/60:.1f} min)")
        print(f"  done in {(time.time()-t0)/60:.1f} min.")

    info["has_tf_bits"] = True
    info["tf_bits_packed_dim"] = packed_dim
    out_info = DATASET_INFO_MMAP
    with open(out_info + ".tmp", "w") as f:
        json.dump(info, f, indent=2)
    os.replace(out_info + ".tmp", out_info)
    print(f"\nSet has_tf_bits=true (packed_dim={packed_dim}) in "
          f"{DATASET_INFO_MMAP}")

    marker_dir = os.path.join(BASE_DIR, "unbound_characetrize", ".markers")
    os.makedirs(marker_dir, exist_ok=True)
    with open(os.path.join(marker_dir, "step2f_tf_bits.done"), "w") as f:
        f.write(f"time={time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"n_chunks={n_chunks}\nn_tfs={n_tfs}\npacked_dim={packed_dim}\n")
    print("Marker written: step2f_tf_bits.done")


if __name__ == "__main__":
    main()
