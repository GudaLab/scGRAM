#!/bin/bash
#SBATCH --partition=guest,batch,unmc_gudalab,unmc_cbsb
#SBATCH --time=165:00:00
#SBATCH --mem=200G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=45
#SBATCH --job-name=10_REGULOME_overlap_summary
#SBATCH --output=10_REGULOME_overlap_summary_%j.out
#SBATCH --error=10_REGULOME_overlap_summary_%j.err

set -euo pipefail
trap 'echo "❌ ERROR on line $LINENO. Exiting." >&2' ERR

# ───────────────────── ENVIRONMENT ─────────────────────

# Fix for set -u + conda.sh expecting PS1
export PS1=${PS1:-"noninteractive"}

# Load conda and activate snapatac2 (has pandas, numpy, pyranges, etc.)
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

# ========= CONFIG (tailored to FOXP2 REGULOME pipeline) =========

# REGULOME BED file (simple 6-column BED: Chr_10  start  end  type  score  strand)
REGULOME_PATH = "/work/avinash/user/CA/CA/regulome_merged_updated.BED"

# Input from Step 9:
#   /work/avinash/user/CA/CA/FOXP2_sorted_scBAMs_merged_beds/<celltype>/<base>_merged.bed
MERGED_BEDS_ROOT = "/work/avinash/user/CA/CA/FOXP2_sorted_scBAMs_merged_beds"

# Output root for REGULOME overlaps & summaries:
#   /work/avinash/user/CA/CA/FOXP2_sorted_scBAMs_REGULOME_csv/<celltype>/
REGULOME_OUT_ROOT = "/work/avinash/user/CA/CA/FOXP2_sorted_scBAMs_REGULOME_csv"

os.makedirs(REGULOME_OUT_ROOT, exist_ok=True)

# Parallelism (we have 45 CPUs; 40 workers is reasonable)
MAX_WORKERS = 40

print("REGULOME_PATH     =", REGULOME_PATH)
print("MERGED_BEDS_ROOT  =", MERGED_BEDS_ROOT)
print("REGULOME_OUT_ROOT =", REGULOME_OUT_ROOT)
print("MAX_WORKERS       =", MAX_WORKERS)
print()

# ========= GLOBAL REGULOME PyRanges (one per worker process) =========

REGULOME_gr = None

def init_worker():
    """
    Initializer for each worker process:
    - loads REGULOME BED into a global PyRanges
    - normalizes chromosome names: 'Chr_10' -> 'chr10'
    """
    global REGULOME_gr
    print("[worker] Loading REGULOME BED:", REGULOME_PATH)
    REGULOME_df = pd.read_csv(REGULOME_PATH, sep='\t', header=None)

    # Original columns: 0:Chr_10, 1:Start, 2:End, 3:type, 4:score, 5:strand
    REGULOME_df.columns = ['Chromosome', 'Start', 'End'] + [
        f'Extra{i}' for i in range(3, REGULOME_df.shape[1])
    ]

    # Normalize chromosome names: Chr_10 → chr10 (adjust if your BAM uses Chr_*)
    REGULOME_df['Chromosome'] = (
        REGULOME_df['Chromosome']
        .astype(str)
        .str.replace(r'^Chr_', 'chr', regex=True)
    )

    REGULOME_df['Start'] = REGULOME_df['Start'].astype(int)
    REGULOME_df['End']   = REGULOME_df['End'].astype(int)

    REGULOME_gr = pr.PyRanges(REGULOME_df)
    print("[worker] REGULOME BED loaded with", len(REGULOME_df), "rows.")

# ========= REGULOME summary logic (adapted to simple BED) =========

def process_REGULOME_data(file_path):
    """
    Read one *_merged_overlaps_REGULOME.csv and aggregate per REGULOME region:
      - Chromosome, Start, End, Regulome_type (enhancer/silencer/etc.)
      - Num_overlaps (how many TFBS/footprints overlap this region in this cell)
      - TFs (unique motif names from the merged BED side, if available)
    """
    df = pd.read_csv(file_path, low_memory=False)
    if df.empty:
        return pd.DataFrame()

    # After join, columns from REGULOME side:
    #   Chromosome, Start, End, Extra3 (type), Extra4, Extra5
    # and from merged BED side suffixed with _b (e.g. Extra3_b = motif name)
    if 'Extra3' not in df.columns:
        print(f"[WARN] No 'Extra3' column (REGULOME type) in {file_path}")
        return pd.DataFrame()

    summaries = []

    # Group by REGULOME region + type
    grouped = df.groupby(['Chromosome', 'Start', 'End', 'Extra3'], sort=False)

    for (chrom, start, end, reg_type), sub in grouped:
        tfs = set()

        # Motif / TF names from merged BED side (if present)
        # Adapt this if motif name lives in a different column.
        if 'Extra3_b' in sub.columns:
            tfs.update(sub['Extra3_b'].dropna().astype(str))

        summaries.append({
            'Chromosome': chrom,
            'Start': int(start),
            'End': int(end),
            'Regulome_type': reg_type,     # enhancer / silencer / etc.
            'Num_overlaps': len(sub),
            'TFs': ','.join(sorted(tfs)) if tfs else ''
        })

    return pd.DataFrame(summaries)

# ========= Per-merged-BED worker =========

def process_one(args):
    """
    For one merged BED:
      - compute overlaps with REGULOME
      - write *_merged_overlaps_REGULOME.csv
      - write *_merged_overlaps_REGULOME_summary.csv
    """
    global REGULOME_gr

    celltype, bed_path = args
    bed_file = os.path.basename(bed_path)

    # Only .bed or .bed.gz
    if not (bed_file.endswith('.bed') or bed_file.endswith('.bed.gz')):
        return

    if bed_file.endswith('.bed'):
        base = bed_file[:-4]   # strip ".bed"
    else:  # .bed.gz
        base = bed_file[:-7]

    # We expect Step 9 outputs to end with "_merged"
    # but we won't hard-fail if they don't.
    # If you want to enforce strictly, uncomment next lines:
    # if not base.endswith("_merged"):
    #     print(f"[SKIP] {bed_file} (does not end with _merged)")
    #     return

    ct_out_dir = os.path.join(REGULOME_OUT_ROOT, celltype)
    os.makedirs(ct_out_dir, exist_ok=True)

    overlap_csv = os.path.join(ct_out_dir, f"{base}_overlaps_REGULOME.csv")
    summary_csv = overlap_csv.replace(
        "_overlaps_REGULOME.csv",
        "_overlaps_REGULOME_summary.csv"
    )

    # Idempotency: if summary already exists, skip
    if os.path.exists(summary_csv):
        print(f"[SKIP] {summary_csv} already exists")
        return

    # ---- Read merged BED ----
    try:
        bed_df = pd.read_csv(bed_path, sep='\t', header=None)
        bed_df.columns = ['Chromosome', 'Start', 'End'] + [
            f'Extra{i}' for i in range(3, bed_df.shape[1])
        ]
        bed_df['Chromosome'] = bed_df['Chromosome'].astype(str)
        bed_df['Start'] = pd.to_numeric(bed_df['Start'], errors='coerce')
        bed_df['End']   = pd.to_numeric(bed_df['End'], errors='coerce')
        bed_df.dropna(subset=['Chromosome', 'Start', 'End'], inplace=True)
        bed_df['Start'] = bed_df['Start'].astype(int)
        bed_df['End']   = bed_df['End'].astype(int)
    except Exception as e:
        print(f"[WARN] ({celltype}) error reading {bed_path}: {e}")
        traceback.print_exc()
        return

    # ---- Join with REGULOME (PyRanges) ----
    try:
        bed_gr = pr.PyRanges(bed_df)
        overlaps = REGULOME_gr.join(bed_gr)
    except Exception as e:
        print(f"[WARN] ({celltype}) PyRanges join failed for {bed_path}: {e}")
        traceback.print_exc()
        return

    if overlaps.df.empty:
        print(f"[INFO] ({celltype}) No REGULOME overlaps: {bed_file}")
        return

    # ---- Write overlaps CSV ----
    try:
        overlaps.df.to_csv(overlap_csv, index=False)
    except Exception as e:
        print(f"[WARN] ({celltype}) Error writing overlaps CSV {overlap_csv}: {e}")
        traceback.print_exc()
        return

    # ---- Summarize REGULOME per file ----
    try:
        summary_df = process_REGULOME_data(overlap_csv)
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

print("\n🚀 Starting parallel REGULOME overlap + summary...")
with concurrent.futures.ProcessPoolExecutor(
    max_workers=MAX_WORKERS,
    initializer=init_worker
) as executor:
    list(executor.map(process_one, tasks))

print("\n🎉 STEP 10 REGULOME COMPLETE:")
print("  Overlap CSVs + summaries in:")
print("  ", REGULOME_OUT_ROOT, "/<celltype>/*_merged_overlaps_REGULOME*.csv", sep="")
EOF
