#!/bin/bash
#SBATCH --partition=guest,batch,unmc_gudalab,unmc_cbsb
#SBATCH --time=165:00:00
#SBATCH --mem=200G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=45
#SBATCH --job-name=10_enhancer_overlap_summary
#SBATCH --output=10_enhancer_overlap_summary_%j.out
#SBATCH --error=10_enhancer_overlap_summary_%j.err

set -euo pipefail
trap 'echo "❌ ERROR on line $LINENO. Exiting." >&2' ERR

# ───────────────────── ENVIRONMENT ─────────────────────

# Fix for set -u + conda.sh expecting PS1
export PS1=${PS1:-"noninteractive"}

# Load conda and activate snapatac2 (has pandas, etc.)
set +u
source /util/opt/anaconda/4.9.2/etc/profile.d/conda.sh
set -u
conda activate snapatac2

echo "Using Python: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"

# ───────────────────── PYTHON PART ─────────────────────

python << 'EOF'
import os
import pandas as pd
import pyranges as pr
import concurrent.futures
import traceback

# ========= CONFIG (tailored to Dopaminergic pipeline) =========

# Enhancer BED file (update this path if yours lives elsewhere)
ENHANCER_PATH = "/work/avinash/user/CA/CA/Enhancer_Database_Final.bed"

# Input from Step 9:
#   /work/avinash/user/CA/CA/Dopaminergic_sorted_scBAMs_merged_beds/<celltype>/<base>_merged.bed
MERGED_BEDS_ROOT = "/work/avinash/user/CA/CA/Dopaminergic_sorted_scBAMs_merged_beds"

# Output root for enhancer overlaps & summaries:
#   /work/avinash/user/CA/CA/Dopaminergic_sorted_scBAMs_enhancer_csv/<celltype>/
ENHANCER_OUT_ROOT = "/work/avinash/user/CA/CA/Dopaminergic_sorted_scBAMs_enhancer_csv"

os.makedirs(ENHANCER_OUT_ROOT, exist_ok=True)

# Parallelism (we have 45 CPUs; 40 workers is reasonable)
MAX_WORKERS = 40

print("ENHANCER_PATH     =", ENHANCER_PATH)
print("MERGED_BEDS_ROOT  =", MERGED_BEDS_ROOT)
print("ENHANCER_OUT_ROOT =", ENHANCER_OUT_ROOT)
print("MAX_WORKERS       =", MAX_WORKERS)
print()

# ========= GLOBAL enhancer PyRanges (one per worker process) =========

enhancer_gr = None

def init_worker():
    """Initializer for each worker process: load enhancer.bed into a global PyRanges."""
    global enhancer_gr
    print("[worker] Loading enhancer BED:", ENHANCER_PATH)
    enhancer_df = pd.read_csv(ENHANCER_PATH, sep='\t', header=None)
    enhancer_df.columns = ['Chromosome', 'Start', 'End'] + [
        f'Extra{i}' for i in range(3, enhancer_df.shape[1])
    ]
    enhancer_df['Start'] = enhancer_df['Start'].astype(int)
    enhancer_df['End']   = enhancer_df['End'].astype(int)
    enhancer_gr = pr.PyRanges(enhancer_df)
    print("[worker] Enhancer BED loaded with", len(enhancer_df), "rows.")

# ========= Enhancer summary logic (from your second script) =========

def process_enhancer_data(file_path):
    """
    Read one *_merged_overlaps_enhancer.csv and aggregate enhancer → genes & TFs
    exactly like your original second script.
    """
    data = pd.read_csv(file_path, low_memory=False)
    enhancer_summaries = []
    current_enhancer = None
    enhancer_info = {}

    for _, row in data.iterrows():
        chrom, start, end = row['Chromosome'], row['Start'], row['End']

        # Same enhancer as previous row?
        if (
            current_enhancer
            and chrom == current_enhancer['Chromosome']
            and start == current_enhancer['Start']
            and end == current_enhancer['End']
        ):
            # Add gene symbols
            enhancer_info['GENE_symbols'].update(row[3:6].dropna().tolist())
            if 'Extra7' in row and pd.notna(row['Extra7']):
                enhancer_info['GENE_symbols'].add(row['Extra7'])

            # Add transcription factors (from bed, suffixed _b)
            if all(k in row for k in ['Extra3_b','Extra5_b']) and pd.notna(row['Extra3_b']) and pd.notna(row['Extra5_b']):
                enhancer_info['Transcription_factors'].add((row['Extra3_b'], row['Extra5_b']))
        else:
            # Finalize previous enhancer group
            if current_enhancer:
                enhancer_summaries.append(enhancer_info)

            # Start new enhancer group
            current_enhancer = {'Chromosome': chrom, 'Start': start, 'End': end}
            enhancer_info = {
                'Chromosome': chrom,
                'Start': start,
                'End': end,
                'GENE_symbols': set(row[3:6].dropna().tolist()),
                'Cell_line': row['Extra6'] if 'Extra6' in row else 'Unknown',
                'Distance_to_nearest_gene': row['Extra8'] if 'Extra8' in row else 'Unknown',
                'Transcription_factors': set(
                    [(row['Extra3_b'], row['Extra5_b'])]
                ) if all(k in row for k in ['Extra3_b','Extra5_b']) and pd.notna(row['Extra3_b']) and pd.notna(row['Extra5_b']) else set(),
            }
            if 'Extra7' in row and pd.notna(row['Extra7']):
                enhancer_info['GENE_symbols'].add(row['Extra7'])

    # Flush last enhancer
    if enhancer_info:
        enhancer_summaries.append(enhancer_info)

    summary_df = pd.DataFrame(enhancer_summaries)

    if not summary_df.empty:
        # Deterministic string formatting
        summary_df['GENE_symbols'] = summary_df['GENE_symbols'].apply(
            lambda x: ','.join(sorted(x))
        )
        summary_df['Transcription_factors'] = summary_df['Transcription_factors'].apply(
            lambda x: ';'.join([f"{tf}({strand})" for tf, strand in sorted(x)])
        )

    return summary_df

# ========= Per-merged-BED worker =========

def process_one(args):
    """
    For one merged BED:
      - compute overlaps with enhancer
      - write *_merged_overlaps_enhancer.csv
      - write *_merged_overlaps_enhancer_summary.csv
    """
    global enhancer_gr

    celltype, bed_path = args
    bed_file = os.path.basename(bed_path)

    # Only .bed or .bed.gz
    if not (bed_file.endswith('.bed') or bed_file.endswith('.bed.gz')):
        return

    if bed_file.endswith('.bed'):
        base = bed_file[:-4]   # strip ".bed"
    else:  # .bed.gz
        base = bed_file[:-7]

    # Ensure we are dealing with merged beds (from Step 9)
    # They should end with "_merged"
    if not base.endswith("_merged"):
        # If you want to force only merged files, you can uncomment:
        # print(f"[SKIP] {bed_file} (does not end with _merged)")
        pass

    ct_out_dir = os.path.join(ENHANCER_OUT_ROOT, celltype)
    os.makedirs(ct_out_dir, exist_ok=True)

    overlap_csv = os.path.join(ct_out_dir, f"{base}_overlaps_enhancer.csv")
    summary_csv = overlap_csv.replace(
        "_overlaps_enhancer.csv",
        "_overlaps_enhancer_summary.csv"
    )

    # Idempotency: if summary already exists, skip
    if os.path.exists(summary_csv):
        print(f"[SKIP] {summary_csv} already exists")
        return

    try:
        # ---- Read merged BED ----
        bed_df = pd.read_csv(bed_path, sep='\t', header=None)
        bed_df.columns = ['Chromosome', 'Start', 'End'] + [
            f'Extra{i}' for i in range(3, bed_df.shape[1])
        ]
        bed_df['Start'] = pd.to_numeric(bed_df['Start'], errors='coerce')
        bed_df['End']   = pd.to_numeric(bed_df['End'], errors='coerce')
        bed_df.dropna(subset=['Chromosome', 'Start', 'End'], inplace=True)
        bed_df['Start'] = bed_df['Start'].astype(int)
        bed_df['End']   = bed_df['End'].astype(int)
    except Exception as e:
        print(f"[WARN] ({celltype}) error reading {bed_path}: {e}")
        traceback.print_exc()
        return

    try:
        # ---- Join with enhancer (PyRanges) ----
        bed_gr = pr.PyRanges(bed_df)
        overlaps = enhancer_gr.join(bed_gr)
    except Exception as e:
        print(f"[WARN] ({celltype}) PyRanges join failed for {bed_path}: {e}")
        traceback.print_exc()
        return

    if overlaps.df.empty:
        print(f"[INFO] ({celltype}) No enhancer overlaps: {bed_file}")
        return

    # ---- Write overlaps CSV ----
    try:
        overlaps.df.to_csv(overlap_csv, index=False)
    except Exception as e:
        print(f"[WARN] ({celltype}) Error writing overlaps CSV {overlap_csv}: {e}")
        traceback.print_exc()
        return

    # ---- Summarize enhancer per file (using your second script's logic) ----
    try:
        summary_df = process_enhancer_data(overlap_csv)
        if summary_df is None or summary_df.empty:
            print(f"[INFO] ({celltype}) No summary rows for {overlap_csv}")
            return
        summary_df.to_csv(summary_csv, index=False)
        print(f"[OK] ({celltype}) {bed_file} → {os.path.basename(overlap_csv)}, {os.path.basename(summary_csv)}")
    except Exception as e:
        print(f"[WARN] ({celltype}) Error summarizing {overlap_csv}: {e}")
        traceback.print_exc()
        return

# ========= Build task list from Step 9 outputs =========

tasks = []

if not os.path.isdir(MERGED_BEDS_ROOT):
    raise SystemExit(f"❌ MERGED_BEDS_ROOT not found: {MERGED_BEDS_ROOT}")

for celltype in sorted(os.listdir(MERGED_BEDS_ROOT)):
    ct_dir = os.path.join(MERGED_BEDS_ROOT, celltype)
    if not os.path.isdir(ct_dir):
        continue

    for bed_file in sorted(os.listdir(ct_dir)):
        if bed_file.endswith(".bed") or bed_file.endswith(".bed.gz"):
            bed_path = os.path.join(ct_dir, bed_file)
            tasks.append((celltype, bed_path))

print(f"Found {len(tasks)} merged BED files to process.")
if not tasks:
    raise SystemExit("❗️ No merged BEDs found. Check MERGED_BEDS_ROOT.")

# ========= Run in parallel =========

print("\n🚀 Starting parallel enhancer overlap + summary...")
with concurrent.futures.ProcessPoolExecutor(
    max_workers=MAX_WORKERS,
    initializer=init_worker
) as executor:
    list(executor.map(process_one, tasks))

print("\n🎉 STEP 10 COMPLETE:")
print("  Overlap CSVs + summaries in:")
print("  ", ENHANCER_OUT_ROOT, "/<celltype>/*_merged_overlaps_enhancer*.csv", sep="")
EOF