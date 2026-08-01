#!/usr/bin/env python3
"""
02e_build_multilabel.py
========================
Derive per-peak multi-label tensors from _unique_peaks.parquet.all_labels.
Two layouts are produced (idempotent, can run independently):

    --label-set basic  (default)
        lab2_NNNN.npy  shape (n, 2)  uint8
        columns: is_enhancer, is_silencer

    --label-set full
        lab4_NNNN.npy  shape (n, 4)  uint8
        columns: is_enhancer, is_promoter, is_genic, is_silencer

A peak qualifies for label L if L is in its all_labels list; labels are
NOT mutually exclusive — a peak can carry several.

The full (4-label) set was added so the model trains against every
annotation type in the source data (reviewer feedback) and so reg-vs-
non-reg can be derived downstream by OR-ing whichever heads you choose.

Idempotent. Safe to re-run with either label-set.

Usage:
    python 02e_build_multilabel.py --label-set basic   # legacy 2-label
    python 02e_build_multilabel.py --label-set full    # 4-label
    python 02e_build_multilabel.py --label-set both    # both
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

BASE_DIR = "/path/to/data"
TRAINING_DIR = os.path.join(BASE_DIR, "unbound_characetrize", "training_data")
MMAP_DIR = os.path.join(TRAINING_DIR, "mmap")
PEAKS_PARQUET = os.path.join(TRAINING_DIR, "_unique_peaks.parquet")
DATASET_INFO_MMAP = os.path.join(MMAP_DIR, "dataset_info_mmap.json")
DUCKDB_TMP = os.path.join(BASE_DIR, "unbound_characetrize", "duckdb_tmp")


def atomic_write_npy(path, arr):
    if not path.endswith(".npy"):
        raise ValueError(f"expected .npy path; got {path}")
    tmp = path[:-4] + ".inflight.npy"
    np.save(tmp, arr, allow_pickle=False)
    os.replace(tmp, path)


def is_chunk_done(path, n_expected, n_cols):
    if not os.path.isfile(path):
        return False
    try:
        from numpy.lib.format import (
            read_magic, read_array_header_1_0, read_array_header_2_0,
        )
        with open(path, "rb") as fh:
            v = read_magic(fh)
            hdr = read_array_header_1_0(fh) if v == (1, 0) else read_array_header_2_0(fh)
        shape = hdr[0]
        return shape == (n_expected, n_cols)
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--label-set", choices=["basic", "full", "both"],
                        default="basic",
                        help="basic: lab2 (enh,sil); full: lab4 "
                             "(enh,prom,genic,sil); both: write both layouts")
    args = parser.parse_args()
    do_basic = args.label_set in ("basic", "both")
    do_full = args.label_set in ("full", "both")

    if not os.path.isfile(DATASET_INFO_MMAP):
        sys.exit(f"missing {DATASET_INFO_MMAP}; run 02b first")
    with open(DATASET_INFO_MMAP) as f:
        info = json.load(f)
    n_chunks = info["n_chunks"]
    n_total = info["n_regions"]
    chunk_size = info["chunk_size"]
    print(f"Dataset: {n_total:,} regions, {n_chunks:,} chunks "
          f"(chunk_size={chunk_size})")

    # Sweep stale staging files for whichever layouts we're writing
    stale_prefixes = []
    if do_basic: stale_prefixes.append("lab2")
    if do_full:  stale_prefixes.append("lab4")
    n_stale = 0
    for pre in stale_prefixes:
        for pat in (f"{pre}_*.inflight.npy", f"{pre}_*.npy.tmp",
                    f"{pre}_*.npy.tmp.npy"):
            for stale in glob.glob(os.path.join(MMAP_DIR, pat)):
                os.remove(stale)
                n_stale += 1
    if n_stale:
        print(f"Removed {n_stale} stale staging files.")

    # ---- Project per-label columns from parquet in one pass ----
    # SELECT only what we need; UTINYINT keeps the result compact.
    import duckdb
    os.makedirs(DUCKDB_TMP, exist_ok=True)
    con = duckdb.connect(":memory:")
    con.execute(f"PRAGMA threads={args.workers}")
    con.execute(f"PRAGMA temp_directory='{DUCKDB_TMP}'")

    select_cols = [
        ("is_enh", "enhancer"),
        ("is_sil", "silencer"),
    ]
    if do_full:
        select_cols.extend([("is_prom", "promoter"), ("is_genic", "genic")])

    select_sql = ",\n            ".join(
        f"CAST(list_contains(all_labels, '{tag}') AS UTINYINT) AS {col}"
        for col, tag in select_cols
    )
    print(f"Extracting {[t for _, t in select_cols]} from parquet...")
    t0 = time.time()
    out = con.execute(f"""
        SELECT
            {select_sql}
        FROM read_parquet('{PEAKS_PARQUET}')
        ORDER BY chrom, peak_start
    """).fetchnumpy()
    is_enh = np.ascontiguousarray(out["is_enh"], dtype=np.uint8)
    is_sil = np.ascontiguousarray(out["is_sil"], dtype=np.uint8)
    is_prom = (np.ascontiguousarray(out["is_prom"], dtype=np.uint8)
               if do_full else None)
    is_genic = (np.ascontiguousarray(out["is_genic"], dtype=np.uint8)
                if do_full else None)
    con.close()

    if len(is_enh) != n_total:
        sys.exit(f"row count mismatch: parquet returned {len(is_enh)}, "
                 f"expected {n_total}")
    print(f"  done in {time.time()-t0:.1f}s")

    # Stats — always show 2-label slice (back-compat); add 4-label slice when asked
    n_enh = int(is_enh.sum())
    n_sil = int(is_sil.sum())
    n_both = int(((is_enh == 1) & (is_sil == 1)).sum())
    n_neither = int(((is_enh == 0) & (is_sil == 0)).sum())
    print(f"\n  enhancer-positive:        {n_enh:>12,} ({100*n_enh/n_total:.4f}%)")
    print(f"  silencer-positive:        {n_sil:>12,} ({100*n_sil/n_total:.4f}%)")
    print(f"  enh ∧ sil (dual):         {n_both:>12,} ({100*n_both/n_total:.4f}%)")
    print(f"  neither enh nor sil:      {n_neither:>12,} ({100*n_neither/n_total:.4f}%)")
    if do_full:
        n_prom = int(is_prom.sum())
        n_genic = int(is_genic.sum())
        print(f"  promoter-positive:        {n_prom:>12,} ({100*n_prom/n_total:.4f}%)")
        print(f"  genic-positive:           {n_genic:>12,} ({100*n_genic/n_total:.4f}%)")

    # ---- Write lab2 (basic) ----
    if do_basic:
        print(f"\nSplitting into {n_chunks} per-chunk lab2_*.npy files...")
        skipped = written = 0
        for ci in range(n_chunks):
            offset = ci * chunk_size
            end = min(offset + chunk_size, n_total)
            n_in = end - offset
            path = os.path.join(MMAP_DIR, f"lab2_{ci:04d}.npy")
            if is_chunk_done(path, n_in, 2):
                skipped += 1; continue
            arr = np.stack([is_enh[offset:end], is_sil[offset:end]], axis=1)
            atomic_write_npy(path, np.ascontiguousarray(arr, dtype=np.uint8))
            written += 1
            if (ci + 1) % 1000 == 0:
                print(f"  ...wrote {ci + 1}/{n_chunks}")
        print(f"  lab2 — wrote {written}, skipped {skipped} (up-to-date)")

    # ---- Write lab4 (full) ----
    if do_full:
        print(f"\nSplitting into {n_chunks} per-chunk lab4_*.npy files...")
        skipped = written = 0
        for ci in range(n_chunks):
            offset = ci * chunk_size
            end = min(offset + chunk_size, n_total)
            n_in = end - offset
            path = os.path.join(MMAP_DIR, f"lab4_{ci:04d}.npy")
            if is_chunk_done(path, n_in, 4):
                skipped += 1; continue
            arr = np.stack([is_enh[offset:end], is_prom[offset:end],
                            is_genic[offset:end], is_sil[offset:end]], axis=1)
            atomic_write_npy(path, np.ascontiguousarray(arr, dtype=np.uint8))
            written += 1
            if (ci + 1) % 1000 == 0:
                print(f"  ...wrote {ci + 1}/{n_chunks}")
        print(f"  lab4 — wrote {written}, skipped {skipped} (up-to-date)")

    # Update dataset_info_mmap.json
    if do_basic:
        info["has_multilabel"] = True
        info["multilabel_classes"] = ["is_enhancer", "is_silencer"]
        info["multilabel_pos_counts"] = {
            "is_enhancer": n_enh, "is_silencer": n_sil,
            "both": n_both, "neither": n_neither,
        }
    if do_full:
        info["has_multilabel4"] = True
        info["multilabel4_classes"] = ["is_enhancer", "is_promoter",
                                       "is_genic", "is_silencer"]
        info["multilabel4_pos_counts"] = {
            "is_enhancer": n_enh,
            "is_promoter": int(is_prom.sum()),
            "is_genic":    int(is_genic.sum()),
            "is_silencer": n_sil,
        }
    out_info = DATASET_INFO_MMAP
    with open(out_info + ".tmp", "w") as f:
        json.dump(info, f, indent=2)
    os.replace(out_info + ".tmp", out_info)
    print(f"\nUpdated {out_info}.")

    marker_dir = os.path.join(BASE_DIR, "unbound_characetrize", ".markers")
    os.makedirs(marker_dir, exist_ok=True)
    marker = ("step2e_multilabel_full.done" if do_full and not do_basic
              else "step2e_multilabel.done")
    with open(os.path.join(marker_dir, marker), "w") as f:
        f.write(f"time={time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"label_set={args.label_set}\n")
        f.write(f"n_enhancer={n_enh}\nn_silencer={n_sil}\n")
        if do_full:
            f.write(f"n_promoter={int(is_prom.sum())}\n")
            f.write(f"n_genic={int(is_genic.sum())}\n")
    print(f"Marker written: {marker}")


if __name__ == "__main__":
    main()
