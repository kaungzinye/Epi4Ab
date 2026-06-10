#!/usr/bin/env python3
"""
Aggregate comparison dashboard for multiple DASA training/inference runs (ASA backend × seed).

Input: a CSV (--runs_csv) with columns:
  - run_id: short label (e.g. dssp_s42)
  - backend: dssp | freesasa | biopython_sr (informational)
  - split_seed: int (informational)
  - nodes_edges_dir: path to nodes_edges root (same across runs is OK)
  - test_record_dir: directory containing <pdb_id>_final_result.txt from regression inference
  - label_file: DASA target parquet name (e.g. node_label_dasa.parquet)
  - epitope_prior_file: optional; e.g. node_iedb_epitope_prior.parquet for AUROC vs literature prior

Outputs:
  - comparison_summary.csv next to --output
  - Single Plotly HTML at --output

Example runs CSV:
  run_id,backend,split_seed,nodes_edges_dir,test_record_dir,label_file,epitope_prior_file
  dssp_s42,dssp,42,/path/nodes_edges,/path/run1/test_record,node_label_dasa.parquet,node_iedb_epitope_prior.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from epi4ab_plot_paths import (
    cohort_csv_dir,
    default_cohort_html,
    ensure_dir,
    maybe_build_index,
    out_base_tag,
)

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError as e:
    raise SystemExit("plotly is required: pip install plotly") from e

try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    roc_auc_score = None  # type: ignore


def _load_run_metrics(
    test_record_dir: Path,
    nodes_edges: Path,
    label_file: str,
    epitope_file: str | None,
) -> dict:
    files = sorted(test_record_dir.glob("*_final_result.txt"))
    pred_all: list[np.ndarray] = []
    true_all: list[np.ndarray] = []
    epi_all: list[np.ndarray] = []
    n_pdb = 0
    for f in files:
        stem = f.stem
        if not stem.endswith("_final_result"):
            continue
        pdb_id = stem.replace("_final_result", "")
        df = pd.read_csv(f, sep="\t")
        if "pred_score" not in df.columns:
            continue
        pred = df["pred_score"].to_numpy(dtype=float)
        n_pdb += 1
        pred_all.append(pred)
        if "true_score" in df.columns and df["true_score"].notna().any():
            true_all.append(df["true_score"].to_numpy(dtype=float))
        if epitope_file and nodes_edges:
            ep_path = nodes_edges / pdb_id / epitope_file
            if ep_path.is_file():
                edf = pd.read_parquet(ep_path)
                if (
                    "epitope_prior" in edf.columns
                    and "resId" in edf.columns
                    and "res_id" in df.columns
                ):
                    m = df.merge(
                        edf[["resId", "epitope_prior"]],
                        left_on="res_id",
                        right_on="resId",
                        how="left",
                    )
                    epi_all.append(m["epitope_prior"].fillna(0).astype(int).to_numpy())

    out: dict = {"n_pdbs": n_pdb}
    if pred_all and true_all and len(pred_all) == len(true_all):
        p = np.concatenate(pred_all)
        t = np.concatenate(true_all)
        out["mse_mean"] = float(np.mean((p - t) ** 2))
        out["pearson"] = (
            float(np.corrcoef(p, t)[0, 1]) if len(p) > 1 and np.std(p) > 0 and np.std(t) > 0 else float("nan")
        )
    else:
        out["mse_mean"] = float("nan")
        out["pearson"] = float("nan")

    if epi_all and pred_all and len(epi_all) == len(pred_all):
        y = np.concatenate(epi_all)
        p = np.concatenate(pred_all)
        if roc_auc_score and len(np.unique(y)) > 1:
            try:
                out["auroc_vs_iedb"] = float(roc_auc_score(y, p))
            except ValueError:
                out["auroc_vs_iedb"] = float("nan")
        else:
            out["auroc_vs_iedb"] = float("nan")
    else:
        out["auroc_vs_iedb"] = float("nan")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare DASA backend runs (metrics + optional epitope AUROC).")
    ap.add_argument("--runs_csv", required=True, help="Table of runs (see module docstring)")
    ap.add_argument(
        "--output",
        default=None,
        help="Output HTML path (default: plots/cohort/<tag>/html/dasa_backend_comparison.html)",
    )
    ap.add_argument("--out_base_tag", default=None)
    ap.add_argument("--build-index", action="store_true")
    args = ap.parse_args()

    tag = args.out_base_tag or out_base_tag()
    out_html = Path(args.output) if args.output else default_cohort_html("dasa_backend_comparison", tag)

    runs = pd.read_csv(args.runs_csv, comment="#")
    required = {"run_id", "nodes_edges_dir", "test_record_dir", "label_file"}
    miss = required - set(runs.columns)
    if miss:
        print(f"runs_csv missing columns: {miss}", file=sys.stderr)
        return 2

    rows = []
    for _, row in runs.iterrows():
        rid = str(row["run_id"])
        ne = Path(str(row["nodes_edges_dir"]))
        tr = Path(str(row["test_record_dir"]))
        lf = str(row["label_file"])
        epi = str(row["epitope_prior_file"]) if "epitope_prior_file" in runs.columns and pd.notna(row.get("epitope_prior_file")) else None
        m = _load_run_metrics(tr, ne, lf, epi)
        rec = {
            "run_id": rid,
            "backend": row.get("backend", ""),
            "split_seed": row.get("split_seed", ""),
            **m,
        }
        rows.append(rec)

    summary = pd.DataFrame(rows)
    ensure_dir(out_html.parent)
    csv_path = cohort_csv_dir(tag) / "dasa_backend_comparison.csv"
    ensure_dir(csv_path.parent)
    summary.to_csv(csv_path, index=False)

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Pearson(pred, true DASA) by run", "AUROC(pred, IEDB epitope prior) by run"),
    )
    fig.add_trace(
        go.Bar(x=summary["run_id"], y=summary["pearson"], name="pearson", showlegend=False),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(x=summary["run_id"], y=summary["auroc_vs_iedb"], name="auroc", showlegend=False),
        row=1,
        col=2,
    )
    fig.update_layout(title_text="DASA backend × seed comparison", height=480)
    plot_html = fig.to_html(full_html=False, include_plotlyjs="cdn")

    summary_html = summary.to_html(index=False, float_format=lambda x: f"{x:.4f}")
    body = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/><title>DASA backend comparison</title></head>
<body>
<h1>DASA backend comparison</h1>
<p>runs_csv: {args.runs_csv}</p>
<p>Summary CSV: {csv_path.resolve()}</p>
{summary_html}
{plot_html}
</body></html>"""
    out_html.write_text(body, encoding="utf-8")
    print(f"Wrote {out_html.resolve()} and {csv_path.resolve()}")
    maybe_build_index(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
