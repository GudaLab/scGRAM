#!/usr/bin/env python3
"""
01_derive_uncharacterized.py
============================
For each cell's bound merged BED, derive "uncharacterized" regions by subtracting
all footprint coordinates that overlap with known regulatory annotations (enhancer,
regulome, promoter, GTF).

Uncharacterized = bound footprint coords NOT found in ANY overlap file.

Usage:
    python 01_derive_uncharacterized.py [--sample SAMPLE_NAME] [--dry-run]

If --sample is omitted, processes ALL sample groups.
"""

import argparse
import csv
import glob
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


BASE_DIR = "/path/to/data"

# Discover all sample groups that have merged_beds directories
def discover_sample_groups():
    """Find all *_sorted_scBAMs_merged_beds directories."""
    pattern = os.path.join(BASE_DIR, "*_sorted_scBAMs_merged_beds")
    groups = []
    for d in sorted(glob.glob(pattern)):
        if os.path.isdir(d):
            # Extract sample name: e.g., Dopaminergic_sorted_scBAMs_merged_beds -> Dopaminergic
            basename = os.path.basename(d)
            sample = basename.replace("_sorted_scBAMs_merged_beds", "")
            groups.append(sample)
    return groups


def get_paths_for_sample(sample):
    """Return directory paths for a given sample group."""
    return {
        "merged_beds": os.path.join(BASE_DIR, f"{sample}_sorted_scBAMs_merged_beds"),
        "enhancer_csv": os.path.join(BASE_DIR, f"{sample}_sorted_scBAMs_enhancer_csv"),
        "regulome_csv": os.path.join(BASE_DIR, f"{sample}_sorted_scBAMs_REGULOME_csv"),
        "promoter_csv": os.path.join(BASE_DIR, f"{sample}_sorted_scBAMs_PROMOTER_csv"),
        "gtf_csv": os.path.join(BASE_DIR, f"{sample}_sorted_scBAMs_GTF_csv"),
        "output": os.path.join(BASE_DIR, f"{sample}_sorted_scBAMs_UNKNOWN_TFBS"),
    }


def parse_bound_bed_coords(bed_path):
    """Extract unique footprint coordinates (chr, start, end) from bound merged BED.

    Bound BED is tab-delimited with 17 columns:
    cols 0-2: footprint chr, start, end
    col 3: TF motif ID
    col 4: motif score
    col 5: strand
    cols 6-9: ATAC peak chr, start, end, peak_id
    ...
    """
    coords = set()
    with open(bed_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            chrom = parts[0]
            try:
                start = int(parts[1])
                end = int(parts[2])
            except ValueError:
                continue
            coords.add((chrom, start, end))
    return coords


def parse_overlap_csv_coords(csv_path):
    """Extract footprint coordinates from overlap CSV files.

    These CSVs have headers. The footprint coordinates are in Start_b/End_b columns.
    The chromosome is in the Chromosome column (same for ref and footprint).
    """
    coords = set()
    if not os.path.isfile(csv_path):
        return coords

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            chrom = row.get("Chromosome", "")
            start_b = row.get("Start_b", "")
            end_b = row.get("End_b", "")
            if not chrom or not start_b or not end_b:
                continue
            try:
                coords.add((chrom, int(float(start_b)), int(float(end_b))))
            except (ValueError, TypeError):
                continue
    return coords


def parse_overlap_bed_coords(bed_path):
    """Extract footprint coordinates from PROMOTER overlap BED (tab-delimited, no header).

    The PROMOTER overlap BED has 25 tab-delimited columns.
    Footprint coordinates are cols 0-2 (chr, start, end) — these are the footprint coords.
    """
    coords = set()
    if not os.path.isfile(bed_path):
        return coords

    with open(bed_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            chrom = parts[0]
            try:
                start = int(parts[1])
                end = int(parts[2])
            except ValueError:
                continue
            coords.add((chrom, start, end))
    return coords


def find_overlap_file(overlap_dir, celltype, barcode_stem, overlap_type):
    """Find the overlap file for a given cell.

    Naming conventions vary slightly:
    - enhancer: {barcode_stem}_merged_overlaps_enhancer.csv
    - regulome: {barcode_stem}_merged_overlaps_REGULOME.csv
    - promoter: {barcode_stem}_merged_overlaps_PROMOTER.bed
    - GTF: {barcode_stem}__merged_overlaps_GTF.csv  (note double underscore sometimes)
    """
    if not os.path.isdir(overlap_dir):
        return None

    celltype_dir = os.path.join(overlap_dir, celltype)
    if not os.path.isdir(celltype_dir):
        return None

    suffixes = {
        "enhancer": ["_merged_overlaps_enhancer.csv", "_BD_merged_overlaps_enhancer.csv"],
        "regulome": ["_merged_overlaps_REGULOME.csv", "_BD_merged_overlaps_REGULOME.csv"],
        "promoter": ["_merged_overlaps_PROMOTER.bed", "_BD_merged_overlaps_PROMOTER.bed"],
        "gtf": [
            "__merged_overlaps_GTF.csv",   # double underscore variant
            "_merged_overlaps_GTF.csv",
            "_BD__merged_overlaps_GTF.csv",
            "_BD_merged_overlaps_GTF.csv",
        ],
    }

    for suffix in suffixes.get(overlap_type, []):
        candidate = os.path.join(celltype_dir, barcode_stem + suffix)
        if os.path.isfile(candidate):
            return candidate

    # Fallback: glob for any file containing the barcode stem and overlap type
    patterns = {
        "enhancer": f"*{barcode_stem}*enhancer*",
        "regulome": f"*{barcode_stem}*REGULOME*",
        "promoter": f"*{barcode_stem}*PROMOTER*",
        "gtf": f"*{barcode_stem}*GTF*",
    }
    matches = glob.glob(os.path.join(celltype_dir, patterns.get(overlap_type, "")))
    if matches:
        return matches[0]

    return None


def process_single_cell(args):
    """Process a single cell: derive uncharacterized regions.

    Returns (cell_id, n_total, n_overlapping, n_uncharacterized) or None on error.
    """
    bed_path, celltype, barcode_stem, paths, output_dir = args

    # Parse all bound footprint coordinates
    bound_coords = parse_bound_bed_coords(bed_path)
    if not bound_coords:
        return None

    # Collect all overlapping coordinates from the 4 overlap sources
    overlapping_coords = set()

    # 1. Enhancer overlaps (CSV)
    enh_file = find_overlap_file(paths["enhancer_csv"], celltype, barcode_stem, "enhancer")
    if enh_file:
        overlapping_coords |= parse_overlap_csv_coords(enh_file)

    # 2. Regulome overlaps (CSV)
    reg_file = find_overlap_file(paths["regulome_csv"], celltype, barcode_stem, "regulome")
    if reg_file:
        overlapping_coords |= parse_overlap_csv_coords(reg_file)

    # 3. Promoter overlaps (BED, tab-delimited)
    prom_file = find_overlap_file(paths["promoter_csv"], celltype, barcode_stem, "promoter")
    if prom_file:
        overlapping_coords |= parse_overlap_bed_coords(prom_file)

    # 4. GTF overlaps (CSV)
    gtf_file = find_overlap_file(paths["gtf_csv"], celltype, barcode_stem, "gtf")
    if gtf_file:
        overlapping_coords |= parse_overlap_csv_coords(gtf_file)

    # Uncharacterized = bound - overlapping
    uncharacterized = bound_coords - overlapping_coords

    # Write uncharacterized regions
    if uncharacterized:
        ct_out_dir = os.path.join(output_dir, celltype)
        os.makedirs(ct_out_dir, exist_ok=True)

        out_file = os.path.join(ct_out_dir, f"{barcode_stem}_uncharacterized.bed")

        # Also write the full bound BED rows for uncharacterized footprints
        # (preserving TF info for downstream feature extraction)
        unchar_rows = []
        with open(bed_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) < 3:
                    continue
                try:
                    coord = (parts[0], int(parts[1]), int(parts[2]))
                except ValueError:
                    continue
                if coord in uncharacterized:
                    unchar_rows.append(line)

        with open(out_file, "w") as f:
            for row in unchar_rows:
                f.write(row + "\n")

    cell_id = f"{celltype}/{barcode_stem}"
    return (cell_id, len(bound_coords), len(overlapping_coords & bound_coords), len(uncharacterized))


def process_sample_group(sample, max_workers=30, dry_run=False):
    """Process all cells in a sample group."""
    paths = get_paths_for_sample(sample)
    merged_beds_dir = paths["merged_beds"]
    output_dir = paths["output"]

    if not os.path.isdir(merged_beds_dir):
        print(f"  Merged BEDs directory not found: {merged_beds_dir}")
        return

    # Discover all cells (BED files organized by celltype subdirs)
    tasks = []
    for celltype in sorted(os.listdir(merged_beds_dir)):
        ct_dir = os.path.join(merged_beds_dir, celltype)
        if not os.path.isdir(ct_dir):
            continue

        for bed_file in sorted(os.listdir(ct_dir)):
            if not bed_file.endswith(".bed"):
                continue

            bed_path = os.path.join(ct_dir, bed_file)
            # Derive barcode stem: remove _merged.bed suffix
            barcode_stem = bed_file.replace("_merged.bed", "").replace("_BD_merged.bed", "")

            # Check if output already exists
            out_check = os.path.join(output_dir, celltype, f"{barcode_stem}_uncharacterized.bed")
            if os.path.isfile(out_check) and os.path.getsize(out_check) > 0:
                continue

            tasks.append((bed_path, celltype, barcode_stem, paths, output_dir))

    if not tasks:
        print(f"  No cells to process (all done or no BED files found).")
        return

    print(f"  Found {len(tasks)} cells to process.")

    if dry_run:
        print(f"  [DRY RUN] Would process {len(tasks)} cells.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Process in parallel
    stats = {"total": 0, "with_unchar": 0, "total_bound": 0, "total_unchar": 0}

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_cell, t): t for t in tasks}

        done_count = 0
        for future in as_completed(futures):
            done_count += 1
            result = future.result()
            if result is None:
                continue

            cell_id, n_total, n_overlap, n_unchar = result
            stats["total"] += 1
            stats["total_bound"] += n_total
            stats["total_unchar"] += n_unchar
            if n_unchar > 0:
                stats["with_unchar"] += 1

            if done_count % 500 == 0:
                print(f"    Processed {done_count}/{len(tasks)} cells...")

    print(f"\n  Summary for {sample}:")
    print(f"    Cells processed: {stats['total']}")
    print(f"    Cells with uncharacterized regions: {stats['with_unchar']}")
    print(f"    Total unique bound footprints: {stats['total_bound']}")
    print(f"    Total uncharacterized footprints: {stats['total_unchar']}")


def main():
    parser = argparse.ArgumentParser(description="Derive uncharacterized regions per cell")
    parser.add_argument("--sample", type=str, default=None,
                        help="Process only this sample group (e.g., Dopaminergic). Default: all.")
    parser.add_argument("--max-workers", type=int, default=30,
                        help="Max parallel workers per sample group.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Count cells without processing.")
    args = parser.parse_args()

    if args.sample:
        samples = [args.sample]
    else:
        samples = discover_sample_groups()

    print(f"Sample groups to process: {samples}\n")

    for sample in samples:
        print(f"Processing sample group: {sample}")
        process_sample_group(sample, max_workers=args.max_workers, dry_run=args.dry_run)
        print()

    print("Done.")


if __name__ == "__main__":
    main()
