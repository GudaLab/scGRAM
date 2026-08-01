#!/usr/bin/env python3
"""
benchmarks/aggregate.py
=======================
Collect benchmarks/results/*/results.json + our own model_output_v2/{cnn,
hybrid,ensemble}/training_results.json, write a unified table, and produce
the comparison bar plot (figures/17_benchmark_comparison.png).

Each row of the output table holds per-label and macro AUROC/AUPRC/F1.
"""

import glob
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/path/to/scgram"
BENCH = os.path.join(BASE, "benchmarks", "results")
FIG_DIR = os.path.join(BASE, "figures")
LABELS = ["is_enhancer", "is_promoter", "is_genic", "is_silencer"]


def load_benchmark_results():
    rows = []
    for p in sorted(glob.glob(os.path.join(BENCH, "*", "results.json"))):
        name = os.path.basename(os.path.dirname(p))
        with open(p) as f:
            r = json.load(f)
        row = {"model": name, **flatten(r)}
        _merge_f1_at_best(row, os.path.dirname(p))
        # Reg-vs-non-reg summary uses "results/<name>" (2-level basename);
        # fall back to "benchmarks/results/<name>" or bare name.
        _merge_reg_vs_nonreg(row, f"results/{name}",
                             f"benchmarks/results/{name}", name)
        rows.append(row)
    return rows


def _merge_f1_at_best(row, model_dir):
    """If <model_dir>/f1_at_best.json exists (written by 13_compute_f1_at_best.py),
    merge its summary fields into the row."""
    p = os.path.join(model_dir, "f1_at_best.json")
    if not os.path.isfile(p):
        return
    s = json.load(open(p))
    row["macro_f1_at_best"] = s.get("macro_f1_at_best_all")
    row["macro_f1_at_best_excl_genic"] = s.get("macro_f1_at_best_excl_genic")
    row["macro_f1_at_best_shared"] = s.get("macro_f1_at_best_shared")


def _merge_reg_vs_nonreg(row, *candidate_keys):
    """Look up this model in reg_vs_nonreg_summary.json and merge AUROC/AUPRC.

    Accepts multiple candidate keys because the naming convention in
    12_eval_reg_vs_nonreg.py uses `<parent_dir_basename>/<model>` whereas
    aggregate.py historically used `<full_root>/<model>`. Try each in order.
    """
    p = os.path.join(BASE, "reg_vs_nonreg_summary.json")
    if not os.path.isfile(p):
        return
    summary = json.load(open(p))
    for key in candidate_keys:
        for r in summary:
            if r.get("model") == key:
                row["reg_auroc"] = r.get("auroc")
                row["reg_auprc"] = r.get("auprc")
                row["reg_f1_at_0p5"] = r.get("f1_at_0p5")
                return


def load_our_models():
    rows = []
    variants = [
        (os.path.join(BASE, "model_output_v2"), "ours-v2", "model_output_v2"),
        (os.path.join(BASE, "model_output_v2_hybrid_v2"), "ours-v2-planA", "model_output_v2_hybrid_v2"),
        (os.path.join(BASE, "model_output"), "ours-v1", "model_output"),
    ]
    for variant_root, tag, reg_root in variants:
        for sub in ("cnn", "hybrid", "ensemble"):
            tr = os.path.join(variant_root, sub, "training_results.json")
            if not os.path.isfile(tr):
                continue
            with open(tr) as f:
                r = json.load(f)
            row = {"model": f"{tag}/{sub}", **flatten(r)}
            _merge_f1_at_best(row, os.path.dirname(tr))
            _merge_reg_vs_nonreg(row, f"{reg_root}/{sub}", f"{tag}/{sub}")
            rows.append(row)
    return rows


def flatten(r):
    out = {"macro_auroc": r.get("test_macro_auroc", r.get("macro_auroc")),
           "macro_auprc": r.get("test_macro_auprc", r.get("macro_auprc")),
           "macro_f1_at_0p5": r.get("test_macro_f1", r.get("macro_f1"))}
    per = r.get("test_per_label", r.get("per_label", {}))
    for L in LABELS:
        v = per.get(L, {})
        out[f"{L}_auroc"] = v.get("auroc")
        out[f"{L}_auprc"] = v.get("auprc")
    # Like-for-like Macro AUROC over shared labels with v1 (enh + sil).
    # Same denominator as v1 reported → fair comparison.
    shared_auroc = [out.get(f"{L}_auroc") for L in ("is_enhancer", "is_silencer")
                    if out.get(f"{L}_auroc") is not None]
    out["macro_auroc_shared"] = (sum(shared_auroc) / len(shared_auroc)
                                 if len(shared_auroc) == 2 else None)
    shared_auprc = [out.get(f"{L}_auprc") for L in ("is_enhancer", "is_silencer")
                    if out.get(f"{L}_auprc") is not None]
    out["macro_auprc_shared"] = (sum(shared_auprc) / len(shared_auprc)
                                 if len(shared_auprc) == 2 else None)
    return out


def main():
    rows = load_our_models() + load_benchmark_results()
    if not rows:
        print("No results found yet.")
        return

    # Sort by macro AUROC desc
    # Sort by reg-vs-non-reg AUROC (the reviewer-aligned headline) when
    # available; fall back to macro_auroc (4-way) otherwise.
    rows.sort(key=lambda r: (r.get("reg_auroc") or r.get("macro_auroc") or 0),
              reverse=True)

    def _fmt(v):
        if isinstance(v, float):
            return f"{v:.4f}"
        if v is None:
            return "—"
        return str(v)

    # CSV — every column, machine-readable
    csv_keys = ["model",
                "reg_auroc", "reg_auprc", "reg_f1_at_0p5",
                "macro_auroc_shared", "macro_auprc_shared",
                "macro_auroc", "macro_auprc",
                "macro_f1_at_0p5", "macro_f1_at_best",
                "macro_f1_at_best_excl_genic", "macro_f1_at_best_shared",
                "is_enhancer_auroc", "is_promoter_auroc",
                "is_genic_auroc", "is_silencer_auroc",
                "is_enhancer_auprc", "is_promoter_auprc",
                "is_genic_auprc", "is_silencer_auprc"]
    print("\n  " + "\t".join(csv_keys[:7]))
    for r in rows:
        print("  " + "\t".join(_fmt(r.get(k)) for k in csv_keys[:7]))

    os.makedirs(BENCH, exist_ok=True)
    with open(os.path.join(BENCH, "comparison.json"), "w") as f:
        json.dump(rows, f, indent=2)

    csv_path = os.path.join(BENCH, "comparison.csv")
    with open(csv_path, "w") as f:
        f.write(",".join(csv_keys) + "\n")
        for r in rows:
            f.write(",".join(
                f"{r[k]:.6f}" if isinstance(r.get(k), float)
                else (str(r.get(k)) if r.get(k) is not None else "")
                for k in csv_keys) + "\n")
    print(f"Wrote {csv_path}")

    # Markdown table — TWO TABLES, ordered by what the reviewer actually
    # asked about. Reg-vs-non-reg first (literal reviewer comment #2),
    # then like-for-like macros, then the per-label breakdown.
    md_path = os.path.join(BENCH, "comparison.md")
    head_keys = ["model", "reg_auroc", "reg_auprc", "reg_f1_at_0p5",
                 "macro_auroc_shared", "macro_auprc_shared"]
    head_titles = ["Model", "Reg AUROC", "Reg AUPRC", "Reg F1@0.5",
                   "Macro AUROC (enh+sil)", "Macro AUPRC (enh+sil)"]
    detail_keys = ["model", "macro_auroc", "macro_auprc",
                   "macro_f1_at_0p5", "macro_f1_at_best",
                   "macro_f1_at_best_excl_genic", "macro_f1_at_best_shared",
                   "is_enhancer_auroc", "is_promoter_auroc",
                   "is_genic_auroc", "is_silencer_auroc"]
    detail_titles = ["Model", "Macro AUROC (all)", "Macro AUPRC (all)",
                     "F1@0.5", "F1@best (all)",
                     "F1@best (excl genic)", "F1@best (enh+sil)",
                     "Enh AUROC", "Prom AUROC", "Genic AUROC", "Sil AUROC"]
    # Conditional footer: if model_output_v2_hybrid_v2/ doesn't have a
    # done.marker yet, the v2/Hybrid rows in this table reflect the
    # pre-warmup (epoch-1-best) model — surface that prominently so a
    # reviewer reading the table without WORKFLOW.md doesn't miss the
    # asterisk.
    hybrid_v2_done = os.path.isfile(os.path.join(
        BASE, "model_output_v2_hybrid_v2", "hybrid", "done.marker"))
    hybrid_caveat = (
        "" if hybrid_v2_done else
        "\n> **Note on v2/hybrid rows:** these come from the pre-warmup "
        "model that diagnosed an optimization-instability pattern "
        "(epoch-1-best val loss). The fixed Hybrid retrain "
        "(`run_hybrid_warmup.sh`, Plan A — LR warmup + grad clip + "
        "conservative LR) is queued post-benchmark and will regenerate "
        "these rows. See WORKFLOW.md § 7A for the diagnosis and "
        "Plan A / A.5 / B remediation ladder.\n"
    )
    with open(md_path, "w") as f:
        f.write("## Headline — reviewer-aligned (reg-vs-non-reg + like-for-like)\n\n")
        f.write("| " + " | ".join(head_titles) + " |\n")
        f.write("|" + "|".join(["---"] * len(head_titles)) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(_fmt(r.get(k)) for k in head_keys) + " |\n")
        f.write(hybrid_caveat)
        f.write("\n## Detail — 4-way macros + per-label AUROC + F1 family\n\n")
        f.write("| " + " | ".join(detail_titles) + " |\n")
        f.write("|" + "|".join(["---"] * len(detail_titles)) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(_fmt(r.get(k)) for k in detail_keys) + " |\n")
        f.write(hybrid_caveat)
    print(f"Wrote {md_path}")

    # Bar plot: macro AUROC, AUPRC, F1 + per-label AUROC for ALL 4 labels.
    # 7 sub-panels so reviewers can scan every metric in one figure.
    os.makedirs(FIG_DIR, exist_ok=True)
    n_models = len(rows)
    metrics = [("macro_auroc", "Macro AUROC"),
               ("is_enhancer_auroc", "Enh AUROC"),
               ("is_promoter_auroc", "Prom AUROC"),
               ("is_genic_auroc",    "Genic AUROC"),
               ("is_silencer_auroc", "Sil AUROC"),
               ("macro_auprc", "Macro AUPRC"),
               ("macro_f1", "Macro F1@0.5")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 7),
                             sharey=False)
    palette = plt.get_cmap("tab20")(np.linspace(0, 1, n_models))
    names = [r["model"] for r in rows]
    for ax, (mk, mt) in zip(axes, metrics):
        vals = [r.get(mk) or 0 for r in rows]
        ax.barh(range(n_models), vals, color=palette,
                edgecolor="black", linewidth=0.5)
        ax.set_yticks(range(n_models)); ax.set_yticklabels(names, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.0); ax.set_title(mt, fontsize=12, fontweight="bold")
        ax.grid(alpha=0.3, axis="x")
        for i, v in enumerate(vals):
            if v: ax.text(min(v + 0.01, 0.98), i, f"{v:.3f}",
                          va="center", fontsize=8)
    fig.suptitle("Benchmark comparison — test chr21/22/X "
                 "(higher is better)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG_DIR, f"17_benchmark_comparison.{ext}"),
                    dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {FIG_DIR}/17_benchmark_comparison.png")


if __name__ == "__main__":
    main()
