# Data

This directory is intentionally empty in the git repository (see `.gitignore`). scGRAM's
inputs and large outputs are hosted externally.

## Inputs (public)

| Dataset | Source |
|---|---|
| Single-nucleus ATAC-seq (whole human brain) | BRAIN Initiative Cell Census Network (BICCN) — *[accession / URL: TODO]* |
| Single-nucleus RNA-seq (validation) | Human Brain Cell Atlas, Siletti et al. — *[accession / URL: TODO]* |
| Reference genome | GRCh38 (`genome.fa`) |
| Gene annotation | GENCODE v44 (`gene.gtf`) |

Place (or symlink) the reference genome and GTF under `/path/to/resources/`.

## Outputs (hosted)

| Product | Location |
|---|---|
| Trained model weights (`model_output_v2/`, `model_output_v2_hybrid_v2/`) | scGRAM portal + Zenodo DOI *[TODO]* |
| Full per-cell prediction catalogue | scGRAM portal: https://www.gudalab-rtools.net/scgram/ |
| Known + novel regulatory-element tables (per group/subtype) | scGRAM portal (genome-browser / atlas / map views) |

## Expected local layout (after download)

```
/path/to/data/
├── resources/                genome.fa, gene.gtf
├── training_data/            labelled corpus, mmap chunks
├── model_output_v2/          trained ensemble weights
├── predictions/              per-cell prediction CSVs
└── Joint_Differential/       known+novel master tables
```
