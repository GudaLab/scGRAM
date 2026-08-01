#!/usr/bin/env python3
"""
10_loco_cv_folds.py
===================
Define a 10-fold chromosome-rotation cross-validation over chr1-22 + chrX.
Chromosomes are round-robined by size rank so each fold's held-out test set
is a balanced mix of large and small chromosomes (balances peak counts).

For fold k:
  TEST  = fold-k chromosomes (held out)
  VAL   = fold-(k+1) chromosomes (for early stopping; disjoint from train & test)
  TRAIN = all remaining chromosomes

Prints one line:   <train_csv>|<val_csv>|<test_csv>

Usage:
    python 10_loco_cv_folds.py --fold 0
    python 10_loco_cv_folds.py --list        # show full fold table
"""

import argparse

# size-sorted (largest → smallest, approximate by chromosome length)
SIZE_SORTED = ["chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr7", "chrX",
               "chr8", "chr9", "chr10", "chr11", "chr12", "chr13", "chr14",
               "chr15", "chr16", "chr17", "chr18", "chr19", "chr20", "chr21", "chr22"]
N_FOLDS = 10


def folds():
    f = [[] for _ in range(N_FOLDS)]
    for i, c in enumerate(SIZE_SORTED):
        f[i % N_FOLDS].append(c)
    return f


def fold_split(k):
    f = folds()
    test = f[k]
    val = f[(k + 1) % N_FOLDS]
    train = [c for c in SIZE_SORTED if c not in test and c not in val]
    return train, val, test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--n-folds", action="store_true", help="print N_FOLDS and exit")
    args = ap.parse_args()
    if args.n_folds:
        print(N_FOLDS); return
    if args.list:
        for k in range(N_FOLDS):
            tr, va, te = fold_split(k)
            print(f"fold {k}: TEST={te}  VAL={va}  (train={len(tr)} chroms)")
        return
    tr, va, te = fold_split(args.fold)
    print(f"{','.join(tr)}|{','.join(va)}|{','.join(te)}")


if __name__ == "__main__":
    main()
