#!/usr/bin/env python3
"""
02_aggregate_training_data.py (DuckDB edition)
================================================
Aggregate labeled training data using DuckDB for memory-safe, out-of-core
columnar aggregation across ~2 TB of overlap CSVs.

Pipeline:
  Phase 1: For each overlap type (enhancer, regulome, promoter, gtf), DuckDB
           reads all per-cell CSVs across all 25 sample groups in a single
           query, projects relevant columns, and writes a parquet file.
  Phase 2: DuckDB loads the four parquet files, UNIONs them, groups by ATAC
           peak coordinates, aggregates TF sets + labels + score statistics,
           and writes the unique-peaks parquet.
  Phase 3: For each unique peak, extract 512bp sequence from the genome FASTA
           and save as chunked .npz files for the training script.

This replaces the previous ProcessPoolExecutor-based Python dict aggregation
that kept OOMing on large samples. DuckDB handles spilling to disk
automatically and uses vectorized columnar execution.

Usage:
    python 02_aggregate_training_data.py [--sample SAMPLE_NAME] [--max-workers 40]
"""

import argparse
import gc
import glob
import json
import os
import shutil
import sys
import zipfile

import duckdb
import numpy as np
import pandas as pd
import pysam
from numpy.lib.format import (
    read_array_header_1_0,
    read_array_header_2_0,
    read_magic,
)

BASE_DIR = "/path/to/data"
GENOME_FASTA = "/path/to/resources/refdata-cellranger-arc-GRCh38-2024-A/fasta/genome.fa"
SEQ_LENGTH = 512

OUTPUT_DIR = os.path.join(BASE_DIR, "unbound_characetrize", "training_data")
DUCKDB_TMP_DIR = os.path.join(BASE_DIR, "unbound_characetrize", "duckdb_tmp")
DUCKDB_FILE = os.path.join(OUTPUT_DIR, "_aggregation.duckdb")


# ----- Regulome class -> label mapping -----
REGULOME_CLASS_MAP = {
    "enhancer": "enhancer",
    "silencer": "silencer",
    "promoter": "promoter",
    "transcriptional_cis_regulatory_region": "cis_regulatory",
    "dnase_i_hypersensitive_site": "other_regulatory",
    "locus_control_region": "other_regulatory",
    "enhancer_blocking_element": "other_regulatory",
    "response_element": "other_regulatory",
    "replication_regulatory_region": "other_regulatory",
    "epigenetically_modified_region": "other_regulatory",
    "matrix_attachment_region": "other_regulatory",
    "conserved_region": "other_regulatory",
    "cage_cluster": "other_regulatory",
    "tata_box": "other_regulatory",
    "nucleotide_motif": "other_regulatory",
}


def ensure_clean_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DUCKDB_TMP_DIR, exist_ok=True)


def phase1_parse_overlaps(con, n_threads, mem_gb):
    """Phase 1: parse raw overlap CSVs of each type into per-type parquet files.

    Each overlap type has a different schema; we project to a common schema
    (chrom, peak_start, peak_end, tf_motif, motif_score, fp_score, label).
    """
    print("\n== Phase 1: Parse overlap CSVs by type (DuckDB) ==")

    # ----- Enhancer (CSV, 25 cols, header) -----
    enh_pq = os.path.join(OUTPUT_DIR, "_phase1_enhancer.parquet")
    if not os.path.isfile(enh_pq):
        print("  Parsing enhancer overlaps...")
        con.execute(f"""
            COPY (
                SELECT
                    CAST(Chromosome AS VARCHAR) AS chrom,
                    TRY_CAST(TRY_CAST(Extra7_b AS DOUBLE) AS BIGINT) AS peak_start,
                    TRY_CAST(TRY_CAST(Extra8_b AS DOUBLE) AS BIGINT) AS peak_end,
                    CAST(Extra3_b AS VARCHAR) AS tf_motif,
                    TRY_CAST(Extra4_b AS DOUBLE) AS motif_score,
                    TRY_CAST(Extra16 AS DOUBLE) AS fp_score,
                    'enhancer' AS label
                FROM read_csv(
                    '{BASE_DIR}/*_sorted_scBAMs_enhancer_csv/*/*.csv',
                    header=true,
                    union_by_name=true,
                    ignore_errors=true,
                    parallel=true,
                    all_varchar=true
                )
                WHERE peak_start IS NOT NULL
                  AND peak_end IS NOT NULL
                  AND tf_motif IS NOT NULL
                  AND tf_motif <> ''
            ) TO '{enh_pq}' (FORMAT PARQUET, COMPRESSION 'zstd')
        """)
        print(f"  Enhancer parquet written: {enh_pq}")
    else:
        print(f"  Enhancer parquet already exists: {enh_pq}")

    # ----- Regulome (CSV, 22 cols, header) -----
    # class is Extra3; we map it via CASE to training label (skip NULLs silently)
    reg_pq = os.path.join(OUTPUT_DIR, "_phase1_regulome.parquet")
    if not os.path.isfile(reg_pq):
        print("  Parsing regulome overlaps...")
        case_lines = []
        for cls, label in REGULOME_CLASS_MAP.items():
            case_lines.append(f"        WHEN LOWER(Extra3) = '{cls}' THEN '{label}'")
        case_expr = "CASE\n" + "\n".join(case_lines) + "\n        ELSE 'other_regulatory'\n      END"
        # TFBS from protein_bind entries is handled in the protein_bind_* fallback
        case_expr = case_expr.replace(
            "ELSE 'other_regulatory'",
            "ELSE CASE WHEN LOWER(Extra3) LIKE 'protein_bind%' THEN 'TFBS' ELSE 'other_regulatory' END"
        )
        con.execute(f"""
            COPY (
                SELECT
                    CASE WHEN Chromosome LIKE 'Chr_%' THEN 'chr' || SUBSTR(Chromosome, 5)
                         ELSE Chromosome END AS chrom,
                    TRY_CAST(TRY_CAST(Extra7 AS DOUBLE) AS BIGINT) AS peak_start,
                    TRY_CAST(TRY_CAST(Extra8 AS DOUBLE) AS BIGINT) AS peak_end,
                    CAST(Extra3_b AS VARCHAR) AS tf_motif,
                    TRY_CAST(Extra4_b AS DOUBLE) AS motif_score,
                    TRY_CAST(Extra16 AS DOUBLE) AS fp_score,
                    {case_expr} AS label
                FROM read_csv(
                    '{BASE_DIR}/*_sorted_scBAMs_REGULOME_csv/*/*.csv',
                    header=true,
                    union_by_name=true,
                    ignore_errors=true,
                    parallel=true,
                    all_varchar=true
                )
                WHERE peak_start IS NOT NULL
                  AND peak_end IS NOT NULL
                  AND tf_motif IS NOT NULL
                  AND tf_motif <> ''
                  AND Extra3 IS NOT NULL
            ) TO '{reg_pq}' (FORMAT PARQUET, COMPRESSION 'zstd')
        """)
        print(f"  Regulome parquet written: {reg_pq}")
    else:
        print(f"  Regulome parquet already exists: {reg_pq}")

    # ----- Promoter (BED, tab-delimited, NO header) -----
    # Cols: 0=chr 1=fp_start 2=fp_end 3=tf_motif 4=motif_score 5=strand
    #       6=peak_chr 7=peak_start 8=peak_end 9=peak_id 10-15=extra
    #       16=fp_score 17-24=promoter ref fields
    prom_pq = os.path.join(OUTPUT_DIR, "_phase1_promoter.parquet")
    if not os.path.isfile(prom_pq):
        print("  Parsing promoter overlaps (tab-delimited BED)...")
        con.execute(f"""
            COPY (
                SELECT
                    CAST(column00 AS VARCHAR) AS chrom,
                    TRY_CAST(TRY_CAST(column07 AS DOUBLE) AS BIGINT) AS peak_start,
                    TRY_CAST(TRY_CAST(column08 AS DOUBLE) AS BIGINT) AS peak_end,
                    CAST(column03 AS VARCHAR) AS tf_motif,
                    TRY_CAST(column04 AS DOUBLE) AS motif_score,
                    TRY_CAST(column16 AS DOUBLE) AS fp_score,
                    'promoter' AS label
                FROM read_csv(
                    '{BASE_DIR}/*_sorted_scBAMs_PROMOTER_csv/*/*.bed',
                    header=false,
                    delim='\t',
                    union_by_name=false,
                    ignore_errors=true,
                    parallel=true,
                    all_varchar=true
                )
                WHERE peak_start IS NOT NULL
                  AND peak_end IS NOT NULL
                  AND tf_motif IS NOT NULL
                  AND tf_motif <> ''
            ) TO '{prom_pq}' (FORMAT PARQUET, COMPRESSION 'zstd')
        """)
        print(f"  Promoter parquet written: {prom_pq}")
    else:
        print(f"  Promoter parquet already exists: {prom_pq}")

    # ----- GTF (CSV, 22 cols, header) -----
    gtf_pq = os.path.join(OUTPUT_DIR, "_phase1_gtf.parquet")
    if not os.path.isfile(gtf_pq):
        print("  Parsing GTF overlaps...")
        con.execute(f"""
            COPY (
                SELECT
                    CAST(Chromosome AS VARCHAR) AS chrom,
                    TRY_CAST(TRY_CAST(Extra7 AS DOUBLE) AS BIGINT) AS peak_start,
                    TRY_CAST(TRY_CAST(Extra8 AS DOUBLE) AS BIGINT) AS peak_end,
                    CAST(Extra3 AS VARCHAR) AS tf_motif,
                    TRY_CAST(Extra4 AS DOUBLE) AS motif_score,
                    TRY_CAST(Extra16 AS DOUBLE) AS fp_score,
                    'genic' AS label
                FROM read_csv(
                    '{BASE_DIR}/*_sorted_scBAMs_GTF_csv/*/*.csv',
                    header=true,
                    union_by_name=true,
                    ignore_errors=true,
                    parallel=true,
                    all_varchar=true
                )
                WHERE peak_start IS NOT NULL
                  AND peak_end IS NOT NULL
                  AND tf_motif IS NOT NULL
                  AND tf_motif <> ''
            ) TO '{gtf_pq}' (FORMAT PARQUET, COMPRESSION 'zstd')
        """)
        print(f"  GTF parquet written: {gtf_pq}")
    else:
        print(f"  GTF parquet already exists: {gtf_pq}")

    return enh_pq, reg_pq, prom_pq, gtf_pq


def phase2_aggregate(con, enh_pq, reg_pq, prom_pq, gtf_pq):
    """Phase 2: union all overlap types, groupby peak, aggregate.

    Label priority (more specific wins): enhancer=1, silencer=2, promoter=3,
    TFBS=4, cis_regulatory=5, other_regulatory=6, genic=7.
    """
    print("\n== Phase 2: Union and aggregate by peak (DuckDB) ==")

    peaks_pq = os.path.join(OUTPUT_DIR, "_unique_peaks.parquet")
    if os.path.isfile(peaks_pq):
        print(f"  Unique peaks parquet already exists: {peaks_pq}")
        return peaks_pq

    # Partition by chromosome — each chromosome is ~1/24 of the data.
    # Within a chromosome, the GROUP BY with list(DISTINCT tf_motif) fits in memory.
    # Across chromosomes, no merging needed since (chrom, peak_start, peak_end) is disjoint.
    standard_chroms = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
    all_parquets = [enh_pq, reg_pq, prom_pq, gtf_pq]
    parquet_list = ", ".join([f"'{p}'" for p in all_parquets])

    per_chrom_dir = os.path.join(OUTPUT_DIR, "_per_chrom")
    os.makedirs(per_chrom_dir, exist_ok=True)

    # Aggregate per chromosome WITHOUT list(DISTINCT tf_motif).
    # list(DISTINCT tf_motif) is astronomically expensive (OOMed at 466 GB even for chr1).
    # TF features are reconstructed per-chunk in Phase 3 instead.
    per_chrom_parquets = []
    for chrom in standard_chroms:
        chrom_pq = os.path.join(per_chrom_dir, f"{chrom}.parquet")
        if os.path.isfile(chrom_pq):
            n = con.execute(f"SELECT COUNT(*) FROM read_parquet('{chrom_pq}')").fetchone()[0]
            print(f"  {chrom}: already done ({n:,} peaks)")
        else:
            print(f"  Aggregating {chrom}...")
            con.execute(f"""
                COPY (
                    SELECT
                        chrom, peak_start, peak_end,
                        list(DISTINCT label) AS all_labels,
                        CASE
                            WHEN list_contains(list(DISTINCT label), 'enhancer')         THEN 'enhancer'
                            WHEN list_contains(list(DISTINCT label), 'silencer')         THEN 'silencer'
                            WHEN list_contains(list(DISTINCT label), 'promoter')         THEN 'promoter'
                            WHEN list_contains(list(DISTINCT label), 'TFBS')             THEN 'TFBS'
                            WHEN list_contains(list(DISTINCT label), 'cis_regulatory')   THEN 'cis_regulatory'
                            WHEN list_contains(list(DISTINCT label), 'other_regulatory') THEN 'other_regulatory'
                            ELSE 'genic'
                        END AS primary_label,
                        COUNT(DISTINCT tf_motif) AS n_tfs,
                        MAX(motif_score) AS max_motif_score,
                        MAX(fp_score) AS max_fp_score,
                        AVG(motif_score) AS mean_motif_score,
                        AVG(fp_score) AS mean_fp_score
                    FROM read_parquet([{parquet_list}])
                    WHERE chrom = '{chrom}'
                    GROUP BY chrom, peak_start, peak_end
                    ORDER BY peak_start
                ) TO '{chrom_pq}' (FORMAT PARQUET, COMPRESSION 'zstd')
            """)
            n = con.execute(f"SELECT COUNT(*) FROM read_parquet('{chrom_pq}')").fetchone()[0]
            print(f"  {chrom}: {n:,} unique peaks")
        per_chrom_parquets.append(chrom_pq)

    # Merge all per-chromosome parquets into one (simple concat, no groupby needed)
    print("  Merging per-chromosome parquets...")
    chrom_glob = os.path.join(per_chrom_dir, "chr*.parquet")
    con.execute(f"""
        COPY (
            SELECT * FROM read_parquet('{chrom_glob}')
            ORDER BY chrom, peak_start
        ) TO '{peaks_pq}' (FORMAT PARQUET, COMPRESSION 'zstd')
    """)

    n_peaks = con.execute(f"SELECT COUNT(*) FROM read_parquet('{peaks_pq}')").fetchone()[0]
    print(f"  Unique peaks after dedup on standard chromosomes: {n_peaks:,}")

    # Class distribution
    dist = con.execute(f"""
        SELECT primary_label, COUNT(*) AS n
        FROM read_parquet('{peaks_pq}')
        GROUP BY primary_label
        ORDER BY n DESC
    """).fetchall()
    print("\n  Class distribution:")
    for label, count in dist:
        print(f"    {label}: {count:,}")

    return peaks_pq


def one_hot_encode(seq, length=SEQ_LENGTH):
    mapping = {
        "A": [1, 0, 0, 0],
        "C": [0, 1, 0, 0],
        "G": [0, 0, 1, 0],
        "T": [0, 0, 0, 1],
        "N": [0.25, 0.25, 0.25, 0.25],
    }
    encoded = np.zeros((length, 4), dtype=np.float32)
    for i, base in enumerate(seq[:length]):
        encoded[i] = mapping.get(base.upper(), [0.25, 0.25, 0.25, 0.25])
    return encoded


def extract_sequence(fasta, chrom, start, end, target_len=SEQ_LENGTH):
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


def phase3_extract_sequences(con, peaks_pq, enh_pq, reg_pq, prom_pq, gtf_pq):
    """Phase 3: extract 512bp DNA sequences and build training chunks.

    Streams peaks from the parquet file in chunks of 50k.
    TF features are reconstructed per-chunk by querying the raw Phase 1 parquets
    (since Phase 2 dropped list(DISTINCT tf_motif) to avoid OOM).
    """
    print("\n== Phase 3: Extract sequences and build training chunks ==")

    parquet_list = ", ".join([f"'{p}'" for p in [enh_pq, reg_pq, prom_pq, gtf_pq]])

    # Build TF vocabulary from the raw parquets (single DISTINCT on one column — cheap).
    # Resumable: reuse existing tf_vocabulary.json if present.
    tf_vocab_path = os.path.join(OUTPUT_DIR, "tf_vocabulary.json")
    if os.path.isfile(tf_vocab_path):
        with open(tf_vocab_path) as f:
            _tv = json.load(f)
        tf_list = _tv["tf_list"]
        tf_to_idx = _tv["tf_to_idx"]
        n_tfs = len(tf_list)
        print(f"  [resume] Reusing tf_vocabulary.json ({n_tfs} TFs)")
    else:
        print("  Building TF vocabulary...")
        tfs_result = con.execute(f"""
            SELECT DISTINCT tf_motif
            FROM read_parquet([{parquet_list}])
            WHERE tf_motif IS NOT NULL AND tf_motif <> ''
            ORDER BY tf_motif
        """).fetchall()
        tf_list = [row[0] for row in tfs_result if row[0]]
        tf_to_idx = {tf: i for i, tf in enumerate(tf_list)}
        n_tfs = len(tf_list)
        print(f"  Total unique TFs: {n_tfs}")
        with open(tf_vocab_path, "w") as f:
            json.dump({"tf_list": tf_list, "tf_to_idx": tf_to_idx}, f)

    # Build label vocabulary (resumable).
    label_vocab_path = os.path.join(OUTPUT_DIR, "label_vocabulary.json")
    if os.path.isfile(label_vocab_path):
        with open(label_vocab_path) as f:
            _lv = json.load(f)
        label_list = _lv["label_list"]
        label_to_idx = _lv["label_to_idx"]
        print(f"  [resume] Reusing label_vocabulary.json: {label_to_idx}")
    else:
        labels_result = con.execute(f"""
            SELECT DISTINCT primary_label FROM read_parquet('{peaks_pq}') ORDER BY primary_label
        """).fetchall()
        label_list = [row[0] for row in labels_result]
        label_to_idx = {l: i for i, l in enumerate(label_list)}
        print(f"  Labels: {label_to_idx}")
        with open(label_vocab_path, "w") as f:
            json.dump({"label_list": label_list, "label_to_idx": label_to_idx}, f)

    # Stream peaks in chunks and extract sequences
    n_total = con.execute(f"SELECT COUNT(*) FROM read_parquet('{peaks_pq}')").fetchone()[0]
    chunk_size = 50000
    n_chunks = (n_total + chunk_size - 1) // chunk_size
    print(f"  Extracting sequences for {n_total:,} peaks in {n_chunks} chunks...")

    fasta = pysam.FastaFile(GENOME_FASTA)
    valid_chroms = set(fasta.references)

    # ---- Resume detection (header-only, no array decompression) ----
    # Each chunk_{i:04d}.npz is a zip of 4 .npy members. We read just the
    # .npy magic + header of each member to get dtype+shape WITHOUT
    # decompressing the array body. This makes validating 8k+ chunks take
    # seconds instead of hours.
    expected_tf_dim = n_tfs
    valid_existing = set()

    def _npz_shapes(path):
        """Return {member: shape} for the 4 required members without decompression."""
        need = ("sequences.npy", "labels.npy",
                "tf_features.npy", "tabular_features.npy")
        shapes = {}
        with zipfile.ZipFile(path) as zf:
            names = set(zf.namelist())
            missing = [m for m in need if m not in names]
            if missing:
                raise ValueError(f"missing zip members: {missing}")
            for m in need:
                with zf.open(m) as fh:
                    v = read_magic(fh)
                    if v == (1, 0):
                        hdr = read_array_header_1_0(fh)
                    elif v == (2, 0):
                        hdr = read_array_header_2_0(fh)
                    else:
                        raise ValueError(f"{m}: unknown npy magic {v}")
                    shapes[m] = hdr[0]  # (shape, fortran_order, dtype)
        return shapes

    _scan_t0 = __import__("time").time()
    for ci in range(n_chunks):
        cp = os.path.join(OUTPUT_DIR, f"chunk_{ci:04d}.npz")
        if not os.path.isfile(cp):
            continue
        try:
            sh = _npz_shapes(cp)
            s = sh["sequences.npy"]
            t = sh["tf_features.npy"]
            tab = sh["tabular_features.npy"]
            l = sh["labels.npy"]
            n = s[0]
            ok = (
                s == (n, SEQ_LENGTH, 4)
                and t == (n, expected_tf_dim)
                and tab == (n, 5)
                and l == (n,)
                and n > 0
            )
            if ok:
                valid_existing.add(ci)
                continue
        except Exception as e:
            print(f"  [resume] chunk_{ci:04d}.npz unreadable ({e}); will regenerate")
        # Corrupt / wrong shape — remove so the loop will rewrite it
        try:
            os.remove(cp)
            print(f"  [resume] removed bad chunk_{ci:04d}.npz")
        except OSError:
            pass
    print(f"  [resume] scan took {__import__('time').time() - _scan_t0:.1f}s")
    # Also nuke any leftover stage files from a prior killed write:
    #   chunk_NNNN.inflight.npz  — current staging name
    #   chunk_NNNN.npz.tmp       — never actually created (old suffix)
    #   chunk_NNNN.npz.tmp.npz   — legacy: np.savez_compressed auto-appended .npz
    for pat in ("chunk_*.inflight.npz", "chunk_*.npz.tmp", "chunk_*.npz.tmp.npz"):
        for tmp in glob.glob(os.path.join(OUTPUT_DIR, pat)):
            os.remove(tmp)
            print(f"  [resume] removed stale {os.path.basename(tmp)}")
    print(f"  [resume] {len(valid_existing)}/{n_chunks} chunks already valid; "
          f"{n_chunks - len(valid_existing)} to (re)build")

    for chunk_idx in range(n_chunks):
        if chunk_idx in valid_existing:
            continue
        offset = chunk_idx * chunk_size
        chunk_df = con.execute(f"""
            SELECT chrom, peak_start, peak_end, primary_label, n_tfs,
                   max_motif_score, max_fp_score, mean_motif_score, mean_fp_score, all_labels
            FROM read_parquet('{peaks_pq}')
            ORDER BY chrom, peak_start
            LIMIT {chunk_size} OFFSET {offset}
        """).df()
        chunk_n = len(chunk_df)

        # Reconstruct TF features for this chunk by querying raw parquets.
        # Register chunk peaks as a temp table, join with raw data filtered by
        # the chunk's chromosome(s), collect distinct TFs per peak.
        chunk_chroms = chunk_df["chrom"].unique().tolist()
        chrom_filter = ", ".join([f"'{c}'" for c in chunk_chroms])

        con.execute("DROP TABLE IF EXISTS _chunk_peaks")
        con.execute("CREATE TEMP TABLE _chunk_peaks (chrom VARCHAR, peak_start BIGINT, peak_end BIGINT)")
        con.executemany(
            "INSERT INTO _chunk_peaks VALUES (?, ?, ?)",
            chunk_df[["chrom", "peak_start", "peak_end"]].values.tolist()
        )

        tf_lookup = con.execute(f"""
            SELECT cp.peak_start, cp.peak_end, list(DISTINCT r.tf_motif) AS tfs
            FROM _chunk_peaks cp
            INNER JOIN read_parquet([{parquet_list}]) r
              ON cp.chrom = r.chrom AND cp.peak_start = r.peak_start AND cp.peak_end = r.peak_end
            WHERE r.chrom IN ({chrom_filter})
            GROUP BY cp.peak_start, cp.peak_end
        """).df()

        # Build a lookup dict: (peak_start, peak_end) -> list of TFs
        peak_tf_map = {}
        for _, tf_row in tf_lookup.iterrows():
            peak_tf_map[(int(tf_row["peak_start"]), int(tf_row["peak_end"]))] = tf_row["tfs"]
        del tf_lookup

        sequences = np.zeros((chunk_n, SEQ_LENGTH, 4), dtype=np.float32)
        tf_features = np.zeros((chunk_n, n_tfs), dtype=np.float32)
        tabular_features = np.zeros((chunk_n, 5), dtype=np.float32)
        labels_arr = np.zeros(chunk_n, dtype=np.int64)

        for i in range(chunk_n):
            row = chunk_df.iloc[i]
            chrom = row["chrom"]
            ps = int(row["peak_start"])
            pe = int(row["peak_end"])

            if chrom in valid_chroms:
                seq = extract_sequence(fasta, chrom, ps, pe, SEQ_LENGTH)
            else:
                seq = "N" * SEQ_LENGTH
            sequences[i] = one_hot_encode(seq, SEQ_LENGTH)

            tfs = peak_tf_map.get((ps, pe), [])
            for tf in tfs:
                if tf in tf_to_idx:
                    tf_features[i, tf_to_idx[tf]] = 1.0

            tabular_features[i, 0] = row["max_motif_score"] or 0.0
            tabular_features[i, 1] = row["max_fp_score"] or 0.0
            tabular_features[i, 2] = row["mean_motif_score"] or 0.0
            tabular_features[i, 3] = row["mean_fp_score"] or 0.0
            tabular_features[i, 4] = row["n_tfs"]

            labels_arr[i] = label_to_idx[row["primary_label"]]

        del peak_tf_map

        chunk_path = os.path.join(OUTPUT_DIR, f"chunk_{chunk_idx:04d}.npz")
        # NOTE: np.savez_compressed auto-appends ".npz" if the given path does
        # not already end in ".npz", so the staging name MUST already end in
        # ".npz" or the file will land at <path>.npz and os.replace will fail.
        tmp_path = os.path.join(OUTPUT_DIR, f"chunk_{chunk_idx:04d}.inflight.npz")
        np.savez_compressed(
            tmp_path,
            sequences=sequences,
            tf_features=tf_features,
            tabular_features=tabular_features,
            labels=labels_arr,
        )
        os.replace(tmp_path, chunk_path)  # atomic on same FS
        print(f"  Saved {chunk_path} ({chunk_n:,} regions)")

        del sequences, tf_features, tabular_features, labels_arr, chunk_df
        gc.collect()

    fasta.close()

    # ---- Build metadata.csv + dataset_info.json from parquet (resumable) ----
    # Streamed straight from _unique_peaks.parquet via DuckDB — no chunk reads.
    # Downstream (03_model.py RegulomeDataset) only uses the `coord` column
    # for chromosome filtering; gc_content is never read. Row ordering
    # matches the chunk loop exactly (same ORDER BY chrom, peak_start).
    print("\n  Building metadata.csv from peaks parquet (streamed)...")
    meta_path = os.path.join(OUTPUT_DIR, "metadata.csv")
    meta_tmp = meta_path + ".tmp"
    # One-shot COPY ... TO stdout would also work, but using COPY to a file is
    # simpler and keeps the all_labels list flattened to a pipe-separated string.
    con.execute(f"""
        COPY (
            SELECT
                chrom || ':' || CAST(peak_start AS VARCHAR) || '-' || CAST(peak_end AS VARCHAR) AS coord,
                primary_label AS label,
                list_aggregate(all_labels, 'string_agg', '|') AS all_labels,
                n_tfs
            FROM read_parquet('{peaks_pq}')
            ORDER BY chrom, peak_start
        ) TO '{meta_tmp}' (HEADER, DELIMITER ',')
    """)
    os.replace(meta_tmp, meta_path)
    print(f"  Metadata saved to {meta_path}")

    # class_counts: cheap aggregate on the parquet, no row-by-row work.
    _cc_rows = con.execute(f"""
        SELECT primary_label, COUNT(*) AS n
        FROM read_parquet('{peaks_pq}')
        GROUP BY primary_label
    """).fetchall()
    class_counts = {l: 0 for l in label_list}
    for lbl, cnt in _cc_rows:
        class_counts[lbl] = int(cnt)

    # Save dataset info
    info = {
        "n_regions": n_total,
        "n_chunks": n_chunks,
        "chunk_size": chunk_size,
        "seq_length": SEQ_LENGTH,
        "n_tfs": n_tfs,
        "n_classes": len(label_list),
        "label_list": label_list,
        "label_to_idx": label_to_idx,
        "class_counts": class_counts,
    }
    info_path = os.path.join(OUTPUT_DIR, "dataset_info.json")
    info_tmp = info_path + ".tmp"
    with open(info_tmp, "w") as f:
        json.dump(info, f, indent=2)
    os.replace(info_tmp, info_path)
    print(f"  Dataset info saved to {info_path}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate labeled training data via DuckDB")
    parser.add_argument("--sample", type=str, default=None,
                        help="(unused — DuckDB processes all samples globally)")
    parser.add_argument("--max-workers", type=int, default=40,
                        help="DuckDB thread count (default 40)")
    args = parser.parse_args()

    ensure_clean_dirs()

    # Clean up legacy per-sample CSVs from the previous Python pipeline
    for old_csv in glob.glob(os.path.join(OUTPUT_DIR, "_peaks_*.csv")):
        os.remove(old_csv)
        print(f"  Removed legacy {old_csv}")

    # Memory-safe DuckDB config.
    # Use in-memory connection (no file-backed mmap that compounds NFS page cache pressure).
    # Keep memory_limit conservative to leave room for OS page cache from NFS reads.
    mem_gb = 500
    con = duckdb.connect(":memory:")
    con.execute(f"PRAGMA threads={min(args.max_workers, 20)}")
    con.execute(f"PRAGMA memory_limit='{mem_gb}GB'")
    con.execute(f"PRAGMA temp_directory='{DUCKDB_TMP_DIR}'")
    con.execute("PRAGMA preserve_insertion_order=false")

    print(f"DuckDB config: threads={args.max_workers}, memory_limit={mem_gb}GB, temp_dir={DUCKDB_TMP_DIR}")

    # Phase 1: parse overlaps to parquet
    enh_pq, reg_pq, prom_pq, gtf_pq = phase1_parse_overlaps(con, args.max_workers, mem_gb)

    # Phase 2: union + groupby
    peaks_pq = phase2_aggregate(con, enh_pq, reg_pq, prom_pq, gtf_pq)

    # Phase 3: extract sequences + write npz chunks
    phase3_extract_sequences(con, peaks_pq, enh_pq, reg_pq, prom_pq, gtf_pq)

    con.close()

    # Clean up the intermediate DuckDB file (optional — keep if you want to inspect)
    print("\nDone. Intermediate files retained in training_data/:")
    for f in ["_phase1_enhancer.parquet", "_phase1_regulome.parquet",
              "_phase1_promoter.parquet", "_phase1_gtf.parquet", "_unique_peaks.parquet"]:
        fp = os.path.join(OUTPUT_DIR, f)
        if os.path.isfile(fp):
            size_mb = os.path.getsize(fp) / 1024 / 1024
            print(f"  {f}: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
