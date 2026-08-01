#!/usr/bin/env python3
"""
08_workflow_diagram.py
======================
Landscape, presentation-quality workflow diagram for the enhancer/silencer
characterization pipeline. Pure matplotlib. Concise blocks, large fonts,
drop shadows, scientific palette.

Output: figures/workflow_diagram.png / .pdf
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon

FIG_DIR = "/path/to/scgram/figures"
os.makedirs(FIG_DIR, exist_ok=True)

# Scientific palette (header color, light fill)
PAL = {
    "input":  ("#1565c0", "#e3f0fb"),
    "derive": ("#00838f", "#e0f3f5"),
    "agg":    ("#2e7d32", "#e6f4e7"),
    "encode": ("#558b2f", "#eef6e3"),
    "train":  ("#e65100", "#fdecdb"),
    "predict":("#ad1457", "#fbe4ee"),
    "viz":    ("#6a1b9a", "#f0e6f6"),
    "result": ("#b71c1c", "#fde6e6"),
    "model":  ("#37474f", "#eceff1"),
    "metric": ("#00695c", "#e0f2f0"),
}

fig, ax = plt.subplots(figsize=(26, 14))
ax.set_xlim(0, 260)
ax.set_ylim(0, 140)
ax.axis("off")
fig.patch.set_facecolor("white")


def shadow_box(x, y, w, h, fill, edge, radius=2.2, lw=2.0):
    # drop shadow
    ax.add_patch(FancyBboxPatch((x + 0.7, y - 0.7), w, h,
                 boxstyle=f"round,pad=0.3,rounding_size={radius}",
                 linewidth=0, facecolor="#00000018", zorder=1))
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle=f"round,pad=0.3,rounding_size={radius}",
                 linewidth=lw, edgecolor=edge, facecolor=fill, zorder=2))


def header_block(x, y, w, h, key, title, subtitle, tsize=15, ssize=11):
    edge, fill = PAL[key]
    shadow_box(x, y, w, h, fill, edge)
    # colored header strip
    hh = h * 0.32
    ax.add_patch(FancyBboxPatch((x, y + h - hh), w, hh,
                 boxstyle="round,pad=0.3,rounding_size=2.2",
                 linewidth=0, facecolor=edge, zorder=3))
    ax.text(x + w / 2, y + h - hh / 2, title, ha="center", va="center",
            fontsize=tsize, fontweight="bold", color="white", zorder=4)
    ax.text(x + w / 2, y + (h - hh) / 2, subtitle, ha="center", va="center",
            fontsize=ssize, color="#1a1a1a", zorder=4, linespacing=1.45)


def card(x, y, w, h, key, title, bullets, tsize=15, bsize=12):
    edge, fill = PAL[key]
    shadow_box(x, y, w, h, fill, edge)
    ax.add_patch(FancyBboxPatch((x, y + h - 7), w, 7,
                 boxstyle="round,pad=0.3,rounding_size=2.2",
                 linewidth=0, facecolor=edge, zorder=3))
    ax.text(x + w / 2, y + h - 3.5, title, ha="center", va="center",
            fontsize=tsize, fontweight="bold", color="white", zorder=4)
    ax.text(x + 2.5, y + h - 10, "\n".join(f"•  {b}" for b in bullets),
            ha="left", va="top", fontsize=bsize, color="#1a1a1a",
            zorder=4, linespacing=1.7)


def chevron(x, y, color):
    ax.add_patch(Polygon([(x, y + 3.2), (x + 3.2, y), (x, y - 3.2)],
                 closed=True, facecolor=color, edgecolor="none", zorder=5))


# ---------------- Title ----------------
ax.text(130, 136, "Unbound TF-Footprint Regulatory Characterization",
        ha="center", va="center", fontsize=27, fontweight="bold", color="#0f172a")
ax.text(130, 130.5,
        "Multi-label deep-learning classification of uncharacterized TF footprints  →  "
        "enhancer · silencer · dual · neither",
        ha="center", va="center", fontsize=14, color="#475569")

# ---------------- Top pipeline ribbon (6 stages) ----------------
stages = [
    ("input",  "INPUT",      "68,299 cells\nscATAC-seq · GRCh38"),
    ("derive", "DERIVE",     "bound − known\n= uncharacterized"),
    ("agg",    "AGGREGATE",  "529M peaks\nDuckDB"),
    ("encode", "ENCODE",     "seq + TF mmap\nuint8 / bit-packed"),
    ("train",  "TRAIN",      "CNN + Hybrid\nmulti-label"),
    ("predict","PREDICT",    "ensemble\n68k cells"),
]
sx, sw, gap, sy, sh = 9, 35, 6.5, 104, 18
for i, (key, t, sub) in enumerate(stages):
    x = sx + i * (sw + gap)
    header_block(x, sy, sw, sh, key, t, sub, tsize=16, ssize=12)
    if i < len(stages) - 1:
        chevron(x + sw + 1.2, sy + sh / 2, PAL[stages[i + 1][0]][0])

# ---------------- Detail cards (3) ----------------
cy, ch, cw = 58, 38, 80
card(6, cy, cw, ch, "agg", "DATA ENGINEERING", [
    "44 scATAC-seq groups → uncharacterized footprints",
    "DuckDB → 529M peaks (10,581 chunks)",
    "Features: 512-bp seq + 879-TF vector + 5 tabular",
    "mmap: uint8 seq + bit-packed TF → 322 GB",
    "Split (by chromosome, no leakage):",
    "    train 87.7%  ·  val 7.3%  ·  test 5.0%",
], tsize=16, bsize=12)

card(90, cy, cw, ch, "train", "MODEL TRAINING & VALIDATION", [
    "2 sigmoid heads: is_enhancer · is_silencer",
    "CNN (1.26M) + Hybrid (1.22M) · BCE pos_weight",
    "bf16 · 8-GPU DDP · AdamW · cosine LR",
    "Validation: single chromosome holdout",
    "    (no k-fold cross-validation)",
    "≤50 epochs, patience 10 → CNN 16, Hybrid 11",
], tsize=15, bsize=12)

card(174, cy, cw, ch, "predict", "INFERENCE & OUTPUT", [
    "Parallel 8-GPU ensemble (~1,000 cells/min)",
    "Per peak → enhancer / silencer / dual / neither",
    "+ p_enhancer, p_silencer, n_tfs per peak",
    "126.7M peaks across all 68,299 cells",
    "Raw probs saved → re-threshold freely",
], tsize=16, bsize=12)

# arrows linking ribbon stages down to the cards (under AGGREGATE, TRAIN, PREDICT)
ribbon_x = {i: sx + i * (sw + gap) + sw / 2 for i in range(len(stages))}
for ri, cx in [(2, 6 + cw / 2), (4, 90 + cw / 2), (5, 174 + cw / 2)]:
    ax.add_patch(FancyArrowPatch((ribbon_x[ri], sy), (cx, cy + ch),
                 arrowstyle="-|>", mutation_scale=16, lw=1.5, color="#94a3b8",
                 connectionstyle="arc3,rad=0.0", zorder=1))

# ---------------- Models flow + metrics (bottom band) ----------------
my, mh = 26, 26
# CNN + Hybrid -> Ensemble
header_block(6, my, 33, mh, "model", "CNN", "RegulatoryClassifier\n1.26M params\ntest AUROC 0.839", tsize=15, ssize=11)
header_block(45, my, 33, mh, "model", "HYBRID", "CNN→128 + 2× attn\n1.22M params\ntest AUROC 0.848", tsize=15, ssize=11)
header_block(96, my, 38, mh, "metric", "ENSEMBLE", "weighted CNN+Hybrid\n(0.42 / 0.58)\ntest AUROC 0.854", tsize=15, ssize=11)
chevron(40.2, my + mh / 2, PAL["metric"][0])
chevron(79.2, my + mh / 2, PAL["metric"][0])

# Metrics callout
card(141, my, 51, mh, "metric", "KEY METRICS (test)", [
    "Macro AUROC 0.854 · AUPRC 0.673 · F1 0.637",
    "is_silencer AUROC 0.940  (4.75% prevalence)",
    "is_enhancer AUROC 0.767  (the ceiling)",
    "Metric for selection: macro-AUROC",
], tsize=14, bsize=11.5)

# Results callout
card(199, my, 55, mh, "result", "PREDICTED CATEGORIES", [
    "enhancer            25.22 %",
    "enhancer_silencer    1.56 %  (dual)",
    "silencer             0.006 %",
    "neither             73.21 %",
], tsize=14, bsize=12)

# ---------------- Footer ----------------
ax.text(130, 8,
        "Pipeline: 01_derive → 02_aggregate → 02b/2d/2e/2f encode → 04_train (03_model) "
        "→ 05_predict → 06/07 visualize   ·   orchestrated by run_pipeline.sh   ·   "
        "see WORKFLOW.md",
        ha="center", va="center", fontsize=11.5, color="#64748b", style="italic")

for ext in ("png", "pdf"):
    fig.savefig(os.path.join(FIG_DIR, f"workflow_diagram.{ext}"),
                dpi=170, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Wrote {FIG_DIR}/workflow_diagram.png and .pdf")
