#!/usr/bin/env python3
"""
Per-complex AUROC: continuous scores vs binary literature epitope prior on a filtered subset.

Typical manifest: output of scripts/build_iedb_epitope_prior.py with columns including
pdb_id and prior_source. Default filter keeps rows where prior_source == iedb_sifts.

Optional CoV-AbDab manifest (scripts/epitope_ground_truth/build_covabdab_epitope_prior.py):
when provided, only PDBs whose covabdab_prior_status is in --covabdab_keep are kept
(intersection with the primary manifest filter).

Inputs mirror scripts/visualize_dasa_backend_comparison.py:
  run_id, backend, split_seed, nodes_edges_dir, test_record_dir, label_file, epitope_prior_file

For each run, for each PDB in the filtered set, merges inference TSV with the epitope
parquet on res_id / resId and computes AUROC(epitope_prior, score) when both classes
of epitope_prior exist. Writes mean ± std of per-PDB AUROCs across the subset.

Examples:
  python scripts/eval_dasa_vs_literature_subset.py \\
    --manifest reports/iedb_prior_manifest.csv \\
    --runs_csv runs.csv --score_col true_score \\
    --out_prefix reports/lit_subset_eval

  python scripts/eval_dasa_vs_literature_subset.py \\
    --manifest reports/iedb_prior_manifest.csv \\
    --covabdab_manifest reports/covabdab_prior_manifest.csv \\
    --runs_csv runs.csv --score_col pred_score

  # Presets (set epitope_prior_file in runs_csv to match):
  python scripts/eval_dasa_vs_literature_subset.py \\
    --manifest reports/covabdab_prior_manifest.csv --truth_tier covabdab \\
    --runs_csv runs_covabdab.csv --score_col true_score --out_prefix reports/eval_cov

  python scripts/eval_dasa_vs_literature_subset.py \\
    --manifest reports/iedb_pdb_prior_manifest.csv --truth_tier iedb_pdb \\
    --runs_csv runs_iedb_pdb.csv --score_col true_score --out_prefix reports/eval_iedb_pdb
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from epi4ab_plot_paths import resolve_run_id, run_csv_dir
from typing import List, Set

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    roc_auc_score = None  # type: ignore


def _read_manifest_filters(
    manifest_path: Path,
    col: str,
    values: Set[str],
) -> Set[str]:
    df = pd.read_csv(manifest_path, comment="#")
    if "pdb_id" not in df.columns:
        raise ValueError(f"Manifest missing pdb_id: {manifest_path}")
    if col not in df.columns:
        raise ValueError(f"Manifest missing column {col!r}: {manifest_path}")
    sub = df[df[col].astype(str).isin(values)]
    return set(sub["pdb_id"].astype(str))


def _pdb_auroc_for_file(
    result_tsv: Path,
    epitope_path: Path,
    score_col: str,
) -> float:
    if roc_auc_score is None:
        return float("nan")
    df = pd.read_csv(result_tsv, sep="\t")
    if score_col not in df.columns or "res_id" not in df.columns:
        return float("nan")
    edf = pd.read_parquet(epitope_path)
    if "epitope_prior" not in edf.columns or "resId" not in edf.columns:
        return float("nan")
    m = df.merge(
        edf[["resId", "epitope_prior"]],
        left_on="res_id",
        right_on="resId",
        how="inner",
    )
    y = m["epitope_prior"].fillna(0).astype(int).to_numpy()
    s = pd.to_numeric(m[score_col], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(s)
    y, s = y[mask], s[mask]
    if len(y) < 2 or len(np.unique(y)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y, s))
    except ValueError:
        return float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description="AUROC vs literature epitope subset (per PDB, then mean±std per run).")
    ap.add_argument("--manifest", required=True, help="IEDB (or merged) prior manifest CSV")
    ap.add_argument(
        "--truth_tier",
        choices=("none", "iedb_uniprot", "iedb_pdb", "covabdab"),
        default="none",
        help="Preset for manifest filter columns/values (default none: use explicit --manifest_prior_*).",
    )
    ap.add_argument(
        "--manifest_prior_source_col",
        default=None,
        help="Column to filter (default prior_source, or set by --truth_tier)",
    )
    ap.add_argument(
        "--manifest_prior_values",
        default=None,
        help="Comma-separated keep values (default iedb_sifts, or set by --truth_tier)",
    )
    ap.add_argument("--covabdab_manifest", default=None, help="Optional second manifest for CoV-AbDab QC")
    ap.add_argument(
        "--covabdab_status_col",
        default="covabdab_prior_status",
    )
    ap.add_argument(
        "--covabdab_keep",
        default="ok",
        help="Comma-separated covabdab_prior_status values to keep when second manifest is set",
    )
    ap.add_argument("--runs_csv", required=True)
    ap.add_argument(
        "--score_col",
        choices=("true_score", "pred_score"),
        default="true_score",
        help="Score used as AUROC prediction (default true_score = ΔASA vs epitope prior)",
    )
    ap.add_argument(
        "--out_prefix",
        default=None,
        help="Writes <out_prefix>_per_pdb.csv and <out_prefix>_per_run.csv "
        "(default: plots/runs/<run_id>/csv/lit_eval_<score_col>)",
    )
    ap.add_argument("--run_id", default=None, help="Hub run id (default: first run_id in runs_csv)")
    args = ap.parse_args()

    if roc_auc_score is None:
        print("sklearn is required for roc_auc_score", file=sys.stderr)
        return 1

    mcol = args.manifest_prior_source_col
    mvals = args.manifest_prior_values
    if args.truth_tier == "iedb_uniprot":
        mcol = mcol or "prior_source"
        mvals = mvals or "iedb_sifts"
    elif args.truth_tier == "iedb_pdb":
        mcol = mcol or "prior_source"
        mvals = mvals or "iedb_pdb_sifts"
    elif args.truth_tier == "covabdab":
        mcol = mcol or "covabdab_prior_source"
        mvals = mvals or "covabdab_sifts"
    else:
        mcol = mcol or "prior_source"
        mvals = mvals or "iedb_sifts"

    keep_vals = {v.strip() for v in mvals.split(",") if v.strip()}
    pdb_set = _read_manifest_filters(Path(args.manifest), mcol, keep_vals)
    if args.covabdab_manifest:
        cov_keep = {v.strip() for v in args.covabdab_keep.split(",") if v.strip()}
        cov_pdbs = _read_manifest_filters(
            Path(args.covabdab_manifest),
            args.covabdab_status_col,
            cov_keep,
        )
        pdb_set &= cov_pdbs

    if not pdb_set:
        print("No PDBs left after manifest filters.", file=sys.stderr)
        return 2

    runs = pd.read_csv(args.runs_csv, comment="#")
    required = ("run_id", "nodes_edges_dir", "test_record_dir", "epitope_prior_file")
    for c in required:
        if c not in runs.columns:
            raise SystemExit(f"runs_csv missing column {c}")

    per_pdb_rows: List[dict] = []
    per_run_rows: List[dict] = []

    for _, run in runs.iterrows():
        run_id = str(run["run_id"])
        nodes_edges = Path(str(run["nodes_edges_dir"]))
        test_record_dir = Path(str(run["test_record_dir"]))
        epi_name = str(run["epitope_prior_file"])
        aucs: List[float] = []
        for pdb_id in sorted(pdb_set):
            res_file = test_record_dir / f"{pdb_id}_final_result.txt"
            ep_path = nodes_edges / pdb_id / epi_name
            if not res_file.is_file() or not ep_path.is_file():
                per_pdb_rows.append(
                    {
                        "run_id": run_id,
                        "pdb_id": pdb_id,
                        "auroc": float("nan"),
                        "status": "missing_file",
                    }
                )
                continue
            a = _pdb_auroc_for_file(res_file, ep_path, args.score_col)
            per_pdb_rows.append(
                {
                    "run_id": run_id,
                    "pdb_id": pdb_id,
                    "auroc": a,
                    "status": "ok" if np.isfinite(a) else "no_signal",
                }
            )
            if np.isfinite(a):
                aucs.append(a)
        if aucs:
            per_run_rows.append(
                {
                    "run_id": run_id,
                    "mean_auroc": float(np.mean(aucs)),
                    "std_auroc": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0,
                    "n_pdbs_auroc": len(aucs),
                    "n_pdbs_manifest": len(pdb_set),
                    "score_col": args.score_col,
                }
            )
        else:
            per_run_rows.append(
                {
                    "run_id": run_id,
                    "mean_auroc": float("nan"),
                    "std_auroc": float("nan"),
                    "n_pdbs_auroc": 0,
                    "n_pdbs_manifest": len(pdb_set),
                    "score_col": args.score_col,
                }
            )

    if args.out_prefix:
        prefix = Path(args.out_prefix)
    else:
        rid = resolve_run_id(args.run_id) or str(runs.iloc[0]["run_id"])
        prefix = run_csv_dir(rid) / f"lit_eval_{args.score_col}"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(per_pdb_rows).to_csv(f"{prefix}_per_pdb.csv", index=False)
    pd.DataFrame(per_run_rows).to_csv(f"{prefix}_per_run.csv", index=False)
    print(f"Wrote {prefix}_per_pdb.csv and {prefix}_per_run.csv  n_pdbs={len(pdb_set)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
