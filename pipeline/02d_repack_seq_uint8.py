#!/usr/bin/env python3
"""
02d_repack_seq_uint8.py
========================
Repack per-chunk sequence arrays from float32 one-hot to uint8 base index.

Original layout (from 02b):
    seq_NNNN.npy  shape (50000, 512, 4)  dtype float32  size ~391 MB

New layout:
    seq8_NNNN.npy shape (50000, 512)     dtype uint8    size ~25 MB

Encoding:
    A=0, C=1, G=2, T=3, N=255

For "N" or any non-ACGT row that 02_aggregate wrote as the uniform [0.25,
0.25, 0.25, 0.25] one-hot, we mark the position as 255 so the Dataset can
reconstruct an "N" one-hot of [0.25, 0.25, 0.25, 0.25] (matching the
original 02 semantics for unknown bases).

Result: dataset shrinks by 16× for the dominant tensor — total mmap data
goes from ~5.9 TB to ~2.0 TB and fits comfortably in OS page cache, with
room for /dev/shm caching of the hottest chunks if desired later.

Idempotent: skip chunks whose seq8_*.npy already exists with the right shape.
Atomic writes via .inflight.npy + os.replace (same pattern as 02b).

Usage:
    python 02d_repack_seq_uint8.py [--workers 20]
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


def is_chunk_done(seq8_path, n_expected, seq_length):
    if not os.path.isfile(seq8_path):
        return False
    try:
        from numpy.lib.format import (
            read_magic, read_array_header_1_0, read_array_header_2_0,
        )
        with open(seq8_path, "rb") as fh:
            v = read_magic(fh)
            hdr = read_array_header_1_0(fh) if v == (1, 0) else read_array_header_2_0(fh)
        shape = hdr[0]
        return shape == (n_expected, seq_length)
    except Exception:
        return False


def repack_one(task):
    """Convert one chunk's float32 one-hot seq -> uint8 base-index seq."""
    ci, src_path, dst_path = task
    arr = np.load(src_path, mmap_mode="r")  # (n, L, 4) float32
    n, L, c = arr.shape
    if c != 4:
        raise RuntimeError(f"chunk {ci}: expected 4-channel one-hot, got {arr.shape}")

    # Argmax across the channel dimension → base index 0..3.
    # For "N" / unknown rows where 02_aggregate wrote uniform [0.25]*4, every
    # channel is equal — np.argmax returns 0 (A), which would silently mislabel.
    # Detect those rows by checking if any channel exceeds 0.5; if not, mark
    # the position as 255 so the Dataset rebuilds an N one-hot.
    base = np.argmax(arr, axis=2).astype(np.uint8)        # (n, L)
    is_n = arr.max(axis=2) <= 0.5                         # (n, L) bool — uniform 0.25 rows
    base[is_n] = 255

    atomic_write_npy(dst_path, base)
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
    chunk_size = info["chunk_size"]
    seq_length = info["seq_length"]
    print(f"Dataset: {info['n_regions']:,} regions, {n_chunks:,} chunks, "
          f"chunk_size={chunk_size}, seq_length={seq_length}")

    # ---- Sweep stale staging files from prior killed runs ----
    print("\n[1/3] Sweeping stale staging files...")
    n_stale = 0
    for pat in ("seq8_*.inflight.npy", "seq8_*.npy.tmp", "seq8_*.npy.tmp.npy"):
        for stale in glob.glob(os.path.join(MMAP_DIR, pat)):
            os.remove(stale)
            n_stale += 1
    print(f"  removed {n_stale} stale files")

    # ---- Build task list ----
    print(f"\n[2/3] Scanning {n_chunks} chunks...")
    tasks = []
    skipped = 0
    missing = 0
    for ci in range(n_chunks):
        src = os.path.join(MMAP_DIR, f"seq_{ci:04d}.npy")
        dst = os.path.join(MMAP_DIR, f"seq8_{ci:04d}.npy")
        if not os.path.isfile(src):
            missing += 1
            continue
        # We'll allow last chunk to have fewer rows than chunk_size, so use
        # the source file's actual row count for validation.
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
        if is_chunk_done(dst, n_in_src, seq_length):
            skipped += 1
            continue
        tasks.append((ci, src, dst))
    print(f"  already repacked: {skipped}")
    print(f"  source missing:   {missing}")
    print(f"  to repack:        {len(tasks)}")

    # ---- Run conversion ----
    if tasks:
        print(f"\n[3/3] Repacking with {args.workers} workers...")
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
                    print(f"  {done}/{len(tasks)} repacked "
                          f"({rate:.1f} chunks/s, ETA {eta/60:.1f} min)")
        print(f"  done in {(time.time()-t0)/60:.1f} min.")

    # ---- Update dataset_info_mmap.json with seq8 capability flag ----
    info["has_seq8"] = True
    out = DATASET_INFO_MMAP + ".tmp"
    with open(out, "w") as f:
        json.dump(info, f, indent=2)
    os.replace(out, DATASET_INFO_MMAP)
    print(f"\nSet has_seq8=true in {DATASET_INFO_MMAP}")

    marker_dir = os.path.join(BASE_DIR, "unbound_characetrize", ".markers")
    os.makedirs(marker_dir, exist_ok=True)
    with open(os.path.join(marker_dir, "step2d_seq_uint8.done"), "w") as f:
        f.write(f"time={time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"n_chunks={n_chunks}\n")
    print("Marker written: step2d_seq_uint8.done")


if __name__ == "__main__":
    main()
