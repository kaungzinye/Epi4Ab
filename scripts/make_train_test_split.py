#!/usr/bin/env python3
"""
Build train/test PDB lists with no antigen-structure leakage.

Modes (--split_mode):
  pdb_hash     — legacy: hash-sort pdbID, take fraction as test (may split same pdb code)
  antigen_group — hold out whole antigen groups (metadata 'pdb' 4-letter code): all
                  complexes with the same antigen structure entry stay on one side.

Seeds (42, 123, 456) pick different held-out antigen groups when test_fraction=0.2.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd


def _pdb_id_col(df: pd.DataFrame) -> str:
    for c in ("pdbID", "pdbId", "pdb_id"):
        if c in df.columns:
            return c
    return str(df.columns[0])


def _antigen_group_col(df: pd.DataFrame) -> str:
    if "pdb" in df.columns:
        return "pdb"
    if "antigen" in df.columns:
        return "antigen"
    raise ValueError("metadata needs 'pdb' or 'antigen' column for antigen_group split")


def split_pdb_hash(pdbs: list[str], seed: int, test_fraction: float) -> tuple[list[str], list[str]]:
    def key(pdb_id: str) -> int:
        h = hashlib.md5(f"{seed}:{pdb_id}".encode("utf-8")).hexdigest()
        return int(h, 16)

    pdbs_sorted = sorted(pdbs, key=key)
    n_test = max(1, int(round(len(pdbs_sorted) * test_fraction)))
    test = pdbs_sorted[:n_test]
    train = pdbs_sorted[n_test:]
    return train, test


def split_antigen_group(
    df: pd.DataFrame, pdb_col: str, ag_col: str, seed: int, test_fraction: float
) -> tuple[list[str], list[str], pd.DataFrame]:
    """Hold out entire antigen groups (by metadata pdb code)."""
    groups = sorted(df[ag_col].astype(str).unique())

    def gkey(ag: str) -> int:
        h = hashlib.md5(f"{seed}:{ag}".encode("utf-8")).hexdigest()
        return int(h, 16)

    groups_sorted = sorted(groups, key=gkey)
    n_test_g = max(1, int(round(len(groups_sorted) * test_fraction)))
    test_groups = set(groups_sorted[:n_test_g])

    rows = []
    train_ids: list[str] = []
    test_ids: list[str] = []
    for _, row in df.iterrows():
        pid = str(row[pdb_col])
        ag = str(row[ag_col])
        split = "test" if ag in test_groups else "train"
        rows.append({pdb_col: pid, ag_col: ag, "split": split})
        if split == "test":
            test_ids.append(pid)
        else:
            train_ids.append(pid)

    report = pd.DataFrame(rows)
    return train_ids, test_ids, report


def main() -> int:
    ap = argparse.ArgumentParser(description="Train/test split without antigen leakage.")
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--out_train", required=True)
    ap.add_argument("--out_test", required=True)
    ap.add_argument("--split_mode", choices=("pdb_hash", "antigen_group"), default="antigen_group")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_fraction", type=float, default=0.2)
    ap.add_argument("--report", default=None, help="Optional CSV: pdbID, antigen, split")
    args = ap.parse_args()

    df = pd.read_csv(args.metadata, comment="#")
    pdb_col = _pdb_id_col(df)

    if args.split_mode == "pdb_hash":
        pdbs = df[pdb_col].astype(str).tolist()
        train, test = split_pdb_hash(pdbs, args.seed, args.test_fraction)
        report = None
    else:
        ag_col = _antigen_group_col(df)
        train, test, report = split_antigen_group(
            df, pdb_col, ag_col, args.seed, args.test_fraction
        )

    Path(args.out_train).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({pdb_col: train}).to_csv(args.out_train, index=False)
    pd.DataFrame({pdb_col: test}).to_csv(args.out_test, index=False)
    if args.report and report is not None:
        report.to_csv(args.report, index=False)

    print(f"split_mode={args.split_mode} seed={args.seed} test_fraction={args.test_fraction}")
    print(f"train={len(train)} test={len(test)}")
    if report is not None:
        n_leak = 0
        for ag, g in report.groupby(ag_col):
            if g["split"].nunique() > 1:
                n_leak += 1
        print(f"antigen groups with mixed split (should be 0): {n_leak}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
