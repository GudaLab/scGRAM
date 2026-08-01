#!/usr/bin/env python3
"""
02b_convert_to_mmap.py
=======================
One-time conversion of compressed chunk_*.npz files into uncompressed per-chunk
.npy files suitable for memory-mapped access. Also derives per-row chromosome
codes from _unique_peaks.parquet to enable fast chromosome-based filtering
without reading the 22 GB metadata.csv.

Inputs:
    training_data/chunk_NNNN.npz         (compressed, 10,581 chunks)
    training_data/_unique_peaks.parquet  (for chrom codes)
    training_data/dataset_info.json      (for n_chunks / chunk_size)

Outputs (training_data/mmap/):
    seq_NNNN.npy    (50k, 512, 4)  float32  ~400 MB each
    tf_NNNN.npy     (50k, 879)     float32  ~176 MB each
    tab_NNNN.npy    (50k, 5)       float32  ~1 MB each
    lab_NNNN.npy    (50k,)         int64    ~400 KB each  (original 7-class)
    chrom_NNNN.npy  (50k,)         uint8    ~50 KB each
    dataset_info_mmap.json

Idempotent: re-running skips chunks whose outputs are all present with the
correct row count.

Usage:
    python 02b_convert_to_mmap.py [--workers 20]
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
PEAKS_PARQUET = os.path.join(TRAINING_DIR, "_unique_peaks.parquet")
DATASET_INFO = os.path.join(TRAINING_DIR, "dataset_info.json")
DUCKDB_TMP = os.path.join(BASE_DIR, "unbound_characetrize", "duckdb_tmp")

# Chromosome → uint8 code. Values 1..25 for standard human chroms; 0 for
# anything else (alt contigs, unknown, scaffolds).
CHROM_MAP = {f"chr{i}": i for i in range(1, 23)}
CHROM_MAP["chrX"] = 23
CHROM_MAP["chrY"] = 24
CHROM_MAP["chrM"] = 25


def expected_files(mmap_dir, ci):
    return {
        "seq":   os.path.join(mmap_dir, f"seq_{ci:04d}.npy"),
        "tf":    os.path.join(mmap_dir, f"tf_{ci:04d}.npy"),
        "tab":   os.path.join(mmap_dir, f"tab_{ci:04d}.npy"),
        "lab":   os.path.join(mmap_dir, f"lab_{ci:04d}.npy"),
        "chrom": os.path.join(mmap_dir, f"chrom_{ci:04d}.npy"),
    }


def is_chunk_done(mmap_dir, ci, n_expected):
    """Header-only validation: each .npy exists and its shape[0] == n_expected."""
    from numpy.lib.format import (
        read_magic, read_array_header_1_0, read_array_header_2_0
    )
    files = expected_files(mmap_dir, ci)
    if not all(os.path.isfile(p) for p in files.values()):
        return False
    try:
        for path in files.values():
            with open(path, "rb") as fh:
                v = read_magic(fh)
                if v == (1, 0):
                    hdr = read_array_header_1_0(fh)
                elif v == (2, 0):
                    hdr = read_array_header_2_0(fh)
                else:
                    return False
                if hdr[0][0] != n_expected:
                    return False
    except Exception:
        return False
    return True


def atomic_write_npy(path, arr):
    # np.save auto-appends ".npy" to any path not ending in ".npy", so the
    # staging name MUST also end in ".npy" — otherwise the file lands at
    # path + ".npy" and os.replace can't find the expected tmp.
    if not path.endswith(".npy"):
        raise ValueError(f"atomic_write_npy expects .npy path; got {path}")
    tmp = path[:-4] + ".inflight.npy"
    np.save(tmp, arr, allow_pickle=False)
    os.replace(tmp, path)


def convert_chunk(task):
    """Worker: read compressed chunk_NNNN.npz and write 5 uncompressed .npy files."""
    ci, chunk_path, mmap_dir, chrom_chunk = task
    files = expected_files(mmap_dir, ci)
    with np.load(chunk_path) as d:
        seq = np.ascontiguousarray(d["sequences"], dtype=np.float32)
        tf = np.ascontiguousarray(d["tf_features"], dtype=np.float32)
        tab = np.ascontiguousarray(d["tabular_features"], dtype=np.float32)
        lab = np.ascontiguousarray(d["labels"], dtype=np.int64)
    n = seq.shape[0]
    if len(chrom_chunk) != n:
        raise RuntimeError(
            f"chunk {ci}: chrom count {len(chrom_chunk)} != seq count {n}"
        )
    atomic_write_npy(files["seq"], seq)
    atomic_write_npy(files["tf"], tf)
    atomic_write_npy(files["tab"], tab)
    atomic_write_npy(files["lab"], lab)
    atomic_write_npy(files["chrom"], np.ascontiguousarray(chrom_chunk, dtype=np.uint8))
    return ci, n


def load_chrom_codes_from_parquet(n_total):
    """Return a uint8 numpy array of chrom codes, one per peak, in the same
    order the chunks were written (ORDER BY chrom, peak_start)."""
    import duckdb
    os.makedirs(DUCKDB_TMP, exist_ok=True)
    con = duckdb.connect(":memory:")
    con.execute("PRAGMA threads=8")
    con.execute(f"PRAGMA temp_directory='{DUCKDB_TMP}'")

    case_lines = []
    for name, code in CHROM_MAP.items():
        case_lines.append(f"            WHEN chrom = '{name}' THEN {code}")
    case_sql = "CASE\n" + "\n".join(case_lines) + "\n            ELSE 0\n        END"

    print(f"  Extracting chrom codes from parquet (single pass, uint8)...")
    t0 = time.time()
    result = con.execute(f"""
        SELECT CAST({case_sql} AS UTINYINT) AS code
        FROM read_parquet('{PEAKS_PARQUET}')
        ORDER BY chrom, peak_start
    """).fetchnumpy()
    codes = np.ascontiguousarray(result["code"], dtype=np.uint8)
    con.close()
    print(f"  got {len(codes):,} chrom codes in {time.time()-t0:.1f}s")
    if len(codes) != n_total:
        raise RuntimeError(
            f"parquet returned {len(codes)} rows, expected {n_total}"
        )
    return codes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()

    os.makedirs(MMAP_DIR, exist_ok=True)

    with open(DATASET_INFO) as f:
        info = json.load(f)
    n_chunks = info["n_chunks"]
    n_total = info["n_regions"]
    chunk_size = info["chunk_size"]
    print(f"Dataset: {n_total:,} regions, {n_chunks:,} chunks, chunk_size={chunk_size}")

    print("\n[1/3] Loading chromosome codes...")
    chrom_codes = load_chrom_codes_from_parquet(n_total)
    unique, counts = np.unique(chrom_codes, return_counts=True)
    code_to_name = {v: k for k, v in CHROM_MAP.items()}
    code_to_name[0] = "OTHER"
    print("  Chromosome distribution:")
    for u, c in zip(unique.tolist(), counts.tolist()):
        print(f"    code {u:2d} ({code_to_name.get(u, '?'):>6}): {c:>12,}")

    # Sweep stale staging files from prior killed or buggy runs. Patterns:
    #   *.inflight.npy     — current staging name
    #   *.npy.tmp          — never actually created (old suffix)
    #   *.npy.tmp.npy      — legacy bug: np.save auto-appended .npy
    print("\n[1b/3] Sweeping stale staging files...")
    stale_count = 0
    stale_bytes = 0
    for pat in ("*.inflight.npy", "*.npy.tmp", "*.npy.tmp.npy"):
        for stale in glob.glob(os.path.join(MMAP_DIR, pat)):
            try:
                stale_bytes += os.path.getsize(stale)
                os.remove(stale)
                stale_count += 1
            except OSError:
                pass
    if stale_count:
        print(f"  removed {stale_count} stale files "
              f"({stale_bytes/1024**3:.1f} GB)")
    else:
        print("  none found")

    print(f"\n[2/3] Scanning {n_chunks} chunks for existing outputs...")
    tasks = []
    already_done = 0
    missing_input = 0
    for ci in range(n_chunks):
        offset = ci * chunk_size
        chrom_chunk = chrom_codes[offset: offset + chunk_size]
        chunk_path = os.path.join(TRAINING_DIR, f"chunk_{ci:04d}.npz")
        if not os.path.isfile(chunk_path):
            missing_input += 1
            continue
        if is_chunk_done(MMAP_DIR, ci, len(chrom_chunk)):
            already_done += 1
            continue
        tasks.append((ci, chunk_path, MMAP_DIR, chrom_chunk))
    print(f"  already converted: {already_done}")
    print(f"  missing input:     {missing_input}")
    print(f"  to convert:        {len(tasks)}")

    # Free the big chrom_codes array now that each task owns its slice.
    del chrom_codes

    if tasks:
        print(f"\n[3/3] Converting with {args.workers} workers...")
        t0 = time.time()
        done = 0
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(convert_chunk, t): t[0] for t in tasks}
            for fut in as_completed(futures):
                ci, n = fut.result()
                done += 1
                if done % 100 == 0 or done == len(tasks):
                    elapsed = time.time() - t0
                    rate = done / max(elapsed, 1e-6)
                    eta_s = (len(tasks) - done) / max(rate, 1e-6)
                    print(f"  {done}/{len(tasks)} converted "
                          f"({rate:.1f} chunks/s, ETA {eta_s/60:.1f} min)")
        print(f"  done in {(time.time()-t0)/60:.1f} min.")
    else:
        print("\n[3/3] Nothing to convert.")

    info_mmap = dict(info)
    info_mmap["mmap_dir"] = MMAP_DIR
    info_mmap["chrom_code_map"] = CHROM_MAP
    info_mmap["chrom_code_map_inv"] = {str(v): k for k, v in CHROM_MAP.items()}
    info_mmap["chrom_code_map_inv"]["0"] = "OTHER"
    out_info = os.path.join(MMAP_DIR, "dataset_info_mmap.json")
    with open(out_info + ".tmp", "w") as f:
        json.dump(info_mmap, f, indent=2)
    os.replace(out_info + ".tmp", out_info)
    print(f"\nWrote {out_info}")

    # Also write the completion marker.
    marker_dir = os.path.join(BASE_DIR, "unbound_characetrize", ".markers")
    os.makedirs(marker_dir, exist_ok=True)
    with open(os.path.join(marker_dir, "step2b_mmap.done"), "w") as f:
        f.write(f"time={time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"n_chunks={n_chunks}\n")
    print("Marker written: step2b_mmap.done")


if __name__ == "__main__":
    main()
