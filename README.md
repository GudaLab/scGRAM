# scGRAM

**scGRAM** — **s**ingle-**c**ell **G**enomic **R**egulatory **A**nnotation **M**odel.

> A multi-modal, multi-label deep-learning framework that fuses transcription-factor (TF)
> footprints with open-chromatin sequence context to annotate **enhancer**, **promoter**,
> **genic** and **silencer** elements at true single-cell resolution.

scGRAM takes per-cell TF footprint peaks from single-nucleus ATAC-seq, represents each peak
by its **TF binding profile**, **6-mer composition**, **tabular peak statistics** and a
**512-bp DNA sequence**, and predicts four regulatory categories through **independent
sigmoid heads** — so a single element can be, for example, both enhancer- and silencer-active,
a state that mutually exclusive (softmax) schemes cannot represent. Applied genome-wide to
previously uncharacterised peaks, scGRAM nominates a cell-type-resolved catalogue of candidate
regulatory elements and ranks those whose activity diverges most sharply across cell classes.

- **Interactive catalogue & genome browser:** https://www.gudalab-rtools.net/scgram/
- **Source code:** https://github.com/gudalab/scgram

---

## Highlights

- **Multi-modal fusion** of TF footprints + sequence + tabular features, learned end-to-end.
- **Multi-label** prediction (enhancer / promoter / genic / silencer) via independent heads.
- **True single-cell resolution** — each cell is treated as an independent sample.
- Outperforms eight independently trained baselines, including all five publicly available
  DNA foundation models (NT-500M, HyenaDNA, Caduceus, DNABERT-2, Sei), under a uniform
  probing protocol.
- Produces a genome-wide, subtype-resolved catalogue of **known and novel** regulatory
  elements, with independent transcriptomic (RNA) cross-validation.

---

## Repository structure

```
scgram/
├── environment.yml            Conda environment (main; see README for the benchmark env)
├── requirements.txt           pip fallback
├── setup_env.sh               convenience environment bootstrap
├── run_pipeline.sh            top-level driver (edit paths first)
│
├── pipeline/                  footprint → features → train → predict
│   ├── 01_derive_uncharacterized.py     derive uncharacterised (bound-but-unannotated) peaks
│   ├── 02_aggregate_training_data.py    build the labelled training corpus (DuckDB)
│   ├── 02b_convert_to_mmap.py           compressed chunks → memory-mapped .npy
│   ├── 02d_repack_seq_uint8.py          pack one-hot sequence to uint8 (16× smaller)
│   ├── 02e_build_multilabel.py          build the 4-label targets
│   ├── 02f_repack_tf_bits.py            pack the TF binding profile
│   ├── 03_model.py                      model + lazy mmap Dataset definitions
│   ├── 04_train.py                      DDP training (CNN / Hybrid / ensemble)
│   ├── 05_predict_uncharacterized.py    genome-wide inference
│   └── master_pipeline_v2.sh, run_v2_retrain.sh, run_hybrid_warmup.sh
│
├── evaluation/                metrics, cross-validation, figures
│   ├── 06_visualize_results.py, 07_comprehensive_plots.py, 07_calibrate_predictions.py
│   ├── 08_workflow_diagram.py, 09_tf_enrichment_heatmap.py
│   ├── 10_loco_cv_folds.py, 11_aggregate_loco.py, run_loco_cv.sh   leave-one-chromosome-out CV
│   ├── 12_eval_reg_vs_nonreg.py         reg-vs-non-reg re-bucketing (probabilistic-OR)
│   ├── 13_compute_f1_at_best.py
│   └── plot_v2_training_curves.py
│
├── benchmarks/                comparison vs 8 baselines
│   ├── run_benchmarks.sh                orchestrates the full suite
│   ├── run_classical.py                 logistic-tf / logistic-kmer / (xgboost)
│   ├── run_deep.py                      Basset + frozen DNA foundation models + MLP head
│   ├── aggregate.py                     unified comparison table + plot
│   └── sei_src/                         vendored Sei architecture (FunctionLab) — see NOTE below
│
├── downstream/                known-vs-novel bucketing, cross-group divergence, FOXP2 evidence
│   ├── 14_*, 15_*, 16_*, 17_*, 18_*     per-subtype master tables, TF divergence/permutation
│   ├── 22_joint_analysis_known_novel.py, 23_joint_differential.py, 24_joint_crossgroup.py
│   ├── 25_joint_tf_divergence.py, 26_joint_tf_cooccur.py, 27_*, 28_*
│   ├── 33_replot_C5b_subset.py          cross-group divergence heatmaps
│   ├── 40_foxp2_evidence.py, 41_foxp2_percell_novel.py, 42_foxp2_final_figure.py
│   └── aggregate_predictions.*, build_neural_table.py, build_paths_table.py
│
├── rna_validation/            independent transcriptomic cross-validation
│   ├── 62_DA_vs_nonDA.py, 63_DA_vs_nonDA_TF.py, 64_annotate_DA_regions.py
│   ├── 65_enhancer_rna_validation.py, 66_rna_subtype_resolution.py
│   ├── 67_concordant_subtype_heatmap.py, 68_silencer_subtype_heatmap.py
│   ├── 70_novel_subtype_heatmap.py, 71_novel_foxp2_subtype_heatmap.py, 72_known_novel_concordant.py
│   └── annotate_case_studies.py         map elements to candidate target genes (GENCODE)
│
├── upstream_footprinting/     prior-framework overlap/differential shell scripts (optional)
├── templates/                 example SLURM submission scripts
└── data/                      (git-ignored) inputs + a README pointing to hosted data
```

> **NOTE — `benchmarks/sei_src/`** is vendored third-party code (the Sei architecture from
> FunctionLab). It is included unmodified for reproducibility and retains its own upstream
> licence; it is *not* covered by this repository's licence.

---

## Installation

```bash
git clone https://github.com/gudalab/scgram.git
cd scgram
conda env create -f environment.yml      # creates 'scgram-env'
conda activate scgram-env
```

Adjust the PyTorch CUDA build in `environment.yml` to match your GPU/driver
(see https://pytorch.org/get-started/locally/).

### Benchmark environment (DNA foundation models)

The five DNA foundation models need an older, pinned stack (kept separate from the main env):

```bash
conda create -n dnabert2-env python=3.10
conda activate dnabert2-env
pip install "torch==2.0.1" --extra-index-url https://download.pytorch.org/whl/cu118
pip install "transformers==4.29" "triton==2.0.0" einops "mamba_ssm==1.2.2"
```

Model-specific notes (also documented inline in `benchmarks/run_deep.py`):
- **DNABERT-2:** its triton flash-attention kernel is patched to
  `torch.nn.functional.scaled_dot_product_attention`.
- **NT-500M:** load `AutoModelForMaskedLM` and extract the `.esm` encoder (a naïve
  `AutoModel.from_pretrained` reinitialises encoder layers).
- **Caduceus-PH:** requires `mamba_ssm==1.2.2` and an explicit `tie_weights()` after loading.
- **HyenaDNA:** run with bf16 (fp16 triggers a cuDNN error); data-shard for throughput.

---

## Configuration

All paths are set in **one place** — `config.sh`. Edit it once, then apply it everywhere:

```bash
#   1. edit config.sh with your real paths
nano config.sh
#   2. fill those values into every script
bash apply_config.sh
```

`config.sh` defines:

| Variable          | Meaning                                                  |
|-------------------|----------------------------------------------------------|
| `SCGRAM_ROOT`     | this repository / working directory (`/path/to/scgram`)  |
| `DATA_ROOT`       | root of the input & output data tree (`/path/to/data`)   |
| `RESOURCES_DIR`   | reference genome (GRCh38 FASTA) + GENCODE v44 GTF         |
| `CONDA_BASE`      | your conda installation (contains `etc/profile.d/`)       |
| `CONDA_ENV`       | conda environment name (default `scgram-env`)            |
| `HF_HOME`         | HuggingFace cache (foundation-model benchmarks)          |
| `SLURM_PARTITION`, `COMPUTE_NODE` | SLURM settings for the `.sbatch` templates |

`apply_config.sh` replaces the `/path/to/...`, `scgram-env`, `<partition>` and `<node>`
placeholders across all scripts with your values (the vendored `benchmarks/sei_src/` is left
untouched). It refuses to run until you have edited `config.sh` off its defaults. Shell scripts
can alternatively `source config.sh` at runtime instead of applying it.

---

## Typical workflow

```bash
# 1. Derive uncharacterised peaks, build the labelled corpus, convert to mmap
bash run_pipeline.sh                       # or step through pipeline/01..02f

# 2. Train (DDP across GPUs) and predict genome-wide
python pipeline/04_train.py  --task multilabel --model ensemble ...
python pipeline/05_predict_uncharacterized.py --sample <GROUP> --ensemble ...

# 3. Evaluate + cross-validate
bash evaluation/run_loco_cv.sh
python evaluation/12_eval_reg_vs_nonreg.py

# 4. Benchmark against baselines + foundation models
sbatch benchmarks/run_benchmarks.sh

# 5. Downstream: known-vs-novel bucketing, cross-group divergence, FOXP2 evidence
python downstream/23_joint_differential.py
python downstream/42_foxp2_final_figure.py

# 6. Independent RNA validation
python rna_validation/72_known_novel_concordant.py
```

---

## Data availability

Input datasets are public:
- **snATAC-seq** — BRAIN Initiative Cell Census Network (BICCN) whole-human-brain resource.
- **snRNA-seq** — Human Brain Cell Atlas (Siletti et al.), used for independent validation.
- **Reference** — GRCh38 genome and GENCODE v44 gene annotation.

Because they are large, the following are hosted externally rather than in git:
- **Trained model weights** and the **full per-cell prediction catalogue** — via the scGRAM
  portal (https://www.gudalab-rtools.net/scgram/) and a versioned archive (Zenodo DOI: *TODO*).
- **Known + novel regulatory-element tables** (per cell-type group/subtype) — browsable at the
  portal with genome-browser, atlas and map views.

See `data/README.md` for exact download locations and expected directory layout.

---

## Citation

If you use scGRAM, please cite:

> *[Author list]*. scGRAM: multi-modal deep learning of single-cell transcription-factor
> footprints for cell-type-resolved regulatory-element annotation in the human brain.
> *[Journal]*, *[year]*. DOI: *[TODO]*

---

## License

This repository is released under the *[LICENSE — e.g. MIT / GPL-3.0]* license (see `LICENSE`).
Vendored third-party code under `benchmarks/sei_src/` retains its original upstream license.
