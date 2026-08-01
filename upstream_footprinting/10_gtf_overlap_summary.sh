#!/bin/bash
#SBATCH --partition=guest,batch,unmc_gudalab,unmc_cbsb
#SBATCH --time=165:00:00
#SBATCH --mem=200G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=10_GTF_overlap_summary
#SBATCH --output=10_GTF_overlap_summary_%j.out
#SBATCH --error=10_GTF_overlap_summary_%j.err

set -euo pipefail
trap 'echo "❌ ERROR on line $LINENO. Exiting." >&2' ERR

# ---- Fix for set -u + conda.sh using PS1 ----
export PS1=${PS1:-"noninteractive"}

# ---- Env: use snapatac2 (has pandas, numpy, pyranges, etc.) ----
set +u
source /util/opt/anaconda/4.9.2/etc/profile.d/conda.sh
set -u
conda activate snapatac2

echo "Using Python: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"

# ---- Paths (adjust only if needed) ----
export MERGED_BEDS_ROOT="/work/avinash/user/CA/CA/Dopaminergic_sorted_scBAMs_merged_beds"
export GTF_PATH="/work/avinash/user/CA/CA/gencode.v49.chr_patch_hapl_scaff.annotation.gtf"
export GTF_OUT_ROOT="/work/avinash/user/CA/CA/Dopaminergic_sorted_scBAMs_GTF_csv"

python << 'EOF'
import os
import re
import pandas as pd
import pyranges as pr

MERGED_BEDS_ROOT = os.environ["MERGED_BEDS_ROOT"]
GTF_PATH         = os.environ["GTF_PATH"]
OUT_ROOT         = os.environ["GTF_OUT_ROOT"]

os.makedirs(OUT_ROOT, exist_ok=True)

print("MERGED_BEDS_ROOT:", MERGED_BEDS_ROOT)
print("GTF_PATH        :", GTF_PATH)
print("OUT_ROOT        :", OUT_ROOT)

# -------------------------------------------------------------------
# 1) Load and parse GTF -> gene-level PyRanges
# -------------------------------------------------------------------
if not os.path.isfile(GTF_PATH):
    raise SystemExit(f"❌ GTF file not found: {GTF_PATH}")

# GTF columns
gtf_cols = [
    "Chromosome", "Source", "Feature", "Start", "End",
    "Score", "Strand", "Frame", "Attribute"
]

print("🔍 Reading GTF (this may take a bit)...")
gtf = pd.read_csv(
    GTF_PATH,
    sep="\t",
    comment="#",
    header=None,
    names=gtf_cols,
    low_memory=False
)

# Keep only 'gene' features
genes = gtf[gtf["Feature"] == "gene"].copy()
print(f"Found {len(genes)} 'gene' entries in GTF.")

# Parse attributes: gene_id, gene_name
def extract_attr(attr_string, key):
    # key is something like 'gene_id' or 'gene_name'
    # GTF attributes look like: gene_id "ENSG..."; gene_name "XYZ";
    if not isinstance(attr_string, str):
        return ""
    pattern = rf'{key}\s+"([^"]+)"'
    m = re.search(pattern, attr_string)
    return m.group(1) if m else ""

genes["gene_id"]   = genes["Attribute"].apply(lambda s: extract_attr(s, "gene_id"))
genes["gene_name"] = genes["Attribute"].apply(lambda s: extract_attr(s, "gene_name"))

# Convert to 0-based, half-open to better match BEDs:
# GTF is 1-based inclusive; BED is typically 0-based start, half-open end.
genes["Start"] = genes["Start"].astype(int) - 1
genes["End"]   = genes["End"].astype(int)

gene_df = genes[["Chromosome", "Start", "End", "Strand", "gene_id", "gene_name"]].copy()
gene_gr = pr.PyRanges(gene_df)

print("✅ GTF gene PyRanges created.")

# -------------------------------------------------------------------
# 2) Iterate over merged BEDs and compute overlaps + summary
# -------------------------------------------------------------------

def read_bed_as_pyranges(bed_path):
    """Read a BED file (unknown extra columns) into PyRanges."""
    df = pd.read_csv(bed_path, sep="\t", header=None)
    if df.shape[1] < 3:
        raise ValueError(f"{bed_path} has fewer than 3 columns; cannot treat as BED")

    df.columns = ["Chromosome", "Start", "End"] + [
        f"Extra{i}" for i in range(3, df.shape[1])
    ]

    # numeric coords
    df["Start"] = pd.to_numeric(df["Start"], errors="coerce")
    df["End"]   = pd.to_numeric(df["End"], errors="coerce")
    df = df.dropna(subset=["Chromosome", "Start", "End"])
    df["Start"] = df["Start"].astype(int)
    df["End"]   = df["End"].astype(int)

    return pr.PyRanges(df)

def summarize_gene_overlaps(overlap_df):
    """
    Given the overlap DataFrame (gene + BED columns with _a/_b suffixes),
    group by gene and count overlaps. Also attempt to aggregate TF motifs
    if Extra3_b / Extra5_b exist (same logic as enhancer/regulome summary).
    """
    # Gene-side columns will typically be unsuffixed or with "_a" depending on PyRanges version.
    # After join, PyRanges usually suffixes duplicates with "_a"/"_b".
    # Let's normalize:
    cols = list(overlap_df.columns)

    # Try to identify gene columns
    chrom_cols = [c for c in cols if c.startswith("Chromosome")]
    start_cols = [c for c in cols if c.startswith("Start")]
    end_cols   = [c for c in cols if c.startswith("End")]
    gid_cols   = [c for c in cols if c.startswith("gene_id")]
    gname_cols = [c for c in cols if c.startswith("gene_name")]
    strand_cols= [c for c in cols if c.startswith("Strand")]

    if not chrom_cols or not start_cols or not end_cols:
        print("⚠️ Overlap DF missing Chromosome/Start/End columns for gene side; returning empty summary.")
        return pd.DataFrame()

    chrom_col  = chrom_cols[0]
    start_col  = start_cols[0]
    end_col    = end_cols[0]
    gene_id_col   = gid_cols[0] if gid_cols else None
    gene_name_col = gname_cols[0] if gname_cols else None
    strand_col    = strand_cols[0] if strand_cols else None

    group_cols = [chrom_col, start_col, end_col]
    if gene_id_col:
        group_cols.append(gene_id_col)
    if gene_name_col:
        group_cols.append(gene_name_col)
    if strand_col:
        group_cols.append(strand_col)

    # optional TF info if present
    has_extra3_b = "Extra3_b" in overlap_df.columns
    has_extra5_b = "Extra5_b" in overlap_df.columns

    rows = []
    for _, group in overlap_df.groupby(group_cols, dropna=False):
        first = group.iloc[0]
        row = {
            "Chromosome": first[chrom_col],
            "Start":      int(first[start_col]),
            "End":        int(first[end_col]),
        }
        if gene_id_col:
            row["gene_id"] = first[gene_id_col]
        if gene_name_col:
            row["gene_name"] = first[gene_name_col]
        if strand_col:
            row["Strand"] = first[strand_col]

        row["n_overlaps"] = len(group)

        # Aggregate TF motifs if columns exist (same logic as enhancer/regulome)
        tf_set = set()
        if has_extra3_b and has_extra5_b:
            for _, r in group.iterrows():
                tf = r["Extra3_b"]
                strand = r["Extra5_b"]
                if pd.notna(tf) and pd.notna(strand):
                    tf_set.add(f"{tf}({strand})")
        row["TFs"] = ";".join(sorted(tf_set)) if tf_set else ""

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    summary = pd.DataFrame(rows)
    summary = summary.sort_values(by="n_overlaps", ascending=False)
    return summary

# -------------------------------------------------------------------
# Traverse merged BED structure:
#   Dopaminergic_sorted_scBAMs_merged_beds/
#       D12NAC/
#         MM_568_D12NAC_<barcode>_BD_merged.bed
# -------------------------------------------------------------------

if not os.path.isdir(MERGED_BEDS_ROOT):
    raise SystemExit(f"❌ MERGED_BEDS_ROOT does not exist: {MERGED_BEDS_ROOT}")

for celltype in sorted(os.listdir(MERGED_BEDS_ROOT)):
    ct_dir = os.path.join(MERGED_BEDS_ROOT, celltype)
    if not os.path.isdir(ct_dir):
        continue

    # Input BEDs for this celltype
    bed_files = sorted(
        f for f in os.listdir(ct_dir)
        if f.endswith("_merged.bed")
    )
    if not bed_files:
        continue

    out_ct_dir = os.path.join(OUT_ROOT, celltype)
    os.makedirs(out_ct_dir, exist_ok=True)

    print(f"\n📂 Celltype: {celltype}")
    print(f"   Input dir : {ct_dir}")
    print(f"   Output dir: {out_ct_dir}")
    print(f"   Found {len(bed_files)} merged BED files.")

    for bed_file in bed_files:
        bed_path = os.path.join(ct_dir, bed_file)
        base = bed_file[:-10] if bed_file.endswith("_merged.bed") else os.path.splitext(bed_file)[0]

        out_overlap = os.path.join(out_ct_dir, f"{base}_merged_overlaps_GTF.csv")
        out_summary = os.path.join(out_ct_dir, f"{base}_merged_overlaps_GTF_summary.csv")

        # Skip if already done
        if os.path.exists(out_summary):
            print(f"  ⏩ Skipping {bed_file} (summary already exists).")
            continue

        print(f"  🔄 Processing {bed_file} ...")

        try:
            bed_gr = read_bed_as_pyranges(bed_path)
        except Exception as e:
            print(f"    ⚠️ Error reading BED {bed_path}: {e}")
            continue

        # Join gene ranges with BED peaks
        overlaps = gene_gr.join(bed_gr)
        df_overlap = overlaps.df

        if df_overlap.empty:
            print(f"    ℹ️ No gene overlaps found for {bed_file}.")
            # still write an empty file for bookkeeping
            df_overlap.to_csv(out_overlap, index=False)
            pd.DataFrame().to_csv(out_summary, index=False)
            continue

        # Save full overlap table
        df_overlap.to_csv(out_overlap, index=False)

        # Build and save summary
        summary_df = summarize_gene_overlaps(df_overlap)
        summary_df.to_csv(out_summary, index=False)

        print(f"    ✅ Overlap:  {out_overlap}")
        print(f"    ✅ Summary:  {out_summary}")

print("\n🎉 All merged BEDs processed against GTF. Outputs in:")
print(f"  {OUT_ROOT}/<celltype>/*_merged_overlaps_GTF*.csv")
EOF