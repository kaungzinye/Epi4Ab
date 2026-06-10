#!/usr/bin/env python3
"""
3×3 grid: predicted vs true DASA (test_record) for each run in a runs CSV.

Optional IEDB (or any epitope_prior_file): two marker colors for epitope_prior 0 vs 1.

Example:
  python scripts/visualize_dasa_pred_vs_true_grid.py \\
    --runs_csv /path/reports/dasa_runs_augmented_iedb_pdb.csv \\
    --output /path/reports/dasa_pred_vs_true_augmented_iedb_pdb_20260410.html
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from epi4ab_plot_paths import default_cohort_html, ensure_dir, maybe_build_index, out_base_tag

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError as e:
    raise SystemExit("plotly is required: pip install plotly") from e

BACKENDS_ORDER = ["dssp", "freesasa", "biopython_sr"]
SEEDS_ORDER = [42, 123, 456]


def _pool_run(
    test_record_dir: Path,
    nodes_edges: Path,
    epitope_file: str | None,
    max_points: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Return (true, pred, epitope_or_none) concatenated over test PDBs."""
    t_list: list[np.ndarray] = []
    p_list: list[np.ndarray] = []
    e_list: list[np.ndarray] = []
    for f in sorted(test_record_dir.glob("*_final_result.txt")):
        stem = f.stem
        if not stem.endswith("_final_result"):
            continue
        pdb_id = stem.replace("_final_result", "")
        df = pd.read_csv(f, sep="\t")
        if not {"pred_score", "true_score", "res_id"}.issubset(df.columns):
            continue
        df = df.dropna(subset=["pred_score", "true_score"])
        if df.empty:
            continue
        pred = df["pred_score"].to_numpy(dtype=float)
        true = df["true_score"].to_numpy(dtype=float)
        epi: np.ndarray | None = None
        if epitope_file and nodes_edges:
            ep_path = nodes_edges / pdb_id / epitope_file
            if ep_path.is_file():
                edf = pd.read_parquet(ep_path)
                if "epitope_prior" in edf.columns and "resId" in edf.columns:
                    m = df.merge(
                        edf[["resId", "epitope_prior"]],
                        left_on="res_id",
                        right_on="resId",
                        how="left",
                    )
                    epi = m["epitope_prior"].fillna(0).astype(int).to_numpy()
        t_list.append(true)
        p_list.append(pred)
        if epi is not None:
            e_list.append(epi)

    if not t_list:
        return np.array([]), np.array([]), None
    t = np.concatenate(t_list)
    p = np.concatenate(p_list)
    e_cat: np.ndarray | None
    if epitope_file and len(e_list) == len(t_list):
        e_cat = np.concatenate(e_list)
    else:
        e_cat = None

    n = t.size
    if n > max_points:
        idx = rng.choice(n, size=max_points, replace=False)
        t, p = t[idx], p[idx]
        if e_cat is not None:
            e_cat = e_cat[idx]
    return t, p, e_cat


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main() -> int:
    ap = argparse.ArgumentParser(description="3×3 pred vs true DASA scatter grid (+ optional IEDB coloring).")
    ap.add_argument("--runs_csv", required=True)
    ap.add_argument(
        "--output",
        default=None,
        help="Output Plotly HTML (default: plots/cohort/<tag>/html/dasa_pred_vs_true_grid.html)",
    )
    ap.add_argument("--out_base_tag", default=None)
    ap.add_argument("--build-index", action="store_true")
    ap.add_argument("--max_points_per_panel", type=int, default=12_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    runs = pd.read_csv(args.runs_csv, comment="#")
    need = {"run_id", "backend", "split_seed", "nodes_edges_dir", "test_record_dir"}
    miss = need - set(runs.columns)
    if miss:
        print(f"runs_csv missing columns: {miss}", file=sys.stderr)
        return 2

    rng = np.random.default_rng(args.seed)
    epitope_col = "epitope_prior_file" if "epitope_prior_file" in runs.columns else None

    def lookup2(backend: str, seed: int) -> pd.Series | None:
        b = backend.lower()
        for _, row in runs.iterrows():
            rb = str(row["backend"]).lower().strip()
            if rb in ("biopython", "opensasa"):
                rb = "biopython_sr"
            if rb != b:
                continue
            if int(row["split_seed"]) != int(seed):
                continue
            return row
        return None

    fig = make_subplots(
        rows=3,
        cols=3,
        vertical_spacing=0.06,
        horizontal_spacing=0.05,
        subplot_titles=[
            f"{be} seed {sd}" for be in BACKENDS_ORDER for sd in SEEDS_ORDER
        ],
    )

    for i, be in enumerate(BACKENDS_ORDER):
        for j, sd in enumerate(SEEDS_ORDER):
            row = lookup2(be, sd)
            col_idx = j + 1
            row_idx = i + 1
            if row is None:
                fig.add_annotation(
                    x=0.5,
                    y=0.5,
                    xref="x domain",
                    yref="y domain",
                    text="missing run",
                    showarrow=False,
                    row=row_idx,
                    col=col_idx,
                )
                continue

            ne = Path(str(row["nodes_edges_dir"]))
            tr = Path(str(row["test_record_dir"])) / "test_record"
            if not tr.is_dir():
                tr = Path(str(row["test_record_dir"]))
            epi_file = str(row[epitope_col]) if epitope_col and pd.notna(row.get(epitope_col)) else None

            t, p, epi = _pool_run(tr, ne, epi_file, args.max_points_per_panel, rng)

            if t.size == 0:
                fig.add_annotation(
                    x=0.5,
                    y=0.5,
                    xref="x domain",
                    yref="y domain",
                    text="no data",
                    showarrow=False,
                    row=row_idx,
                    col=col_idx,
                )
                continue

            r = _pearson(t, p)
            lo = float(min(t.min(), p.min()))
            hi = float(max(t.max(), p.max()))

            # y=x reference
            fig.add_trace(
                go.Scatter(
                    x=[lo, hi],
                    y=[lo, hi],
                    mode="lines",
                    line=dict(color="black", width=1, dash="dash"),
                    name="y=x",
                    showlegend=(i == 0 and j == 0),
                    legendgroup="diag",
                ),
                row=row_idx,
                col=col_idx,
            )

            if epi is not None and epi.size == t.size:
                mask0 = epi == 0
                mask1 = epi == 1
                if mask0.any():
                    fig.add_trace(
                        go.Scatter(
                            x=t[mask0],
                            y=p[mask0],
                            mode="markers",
                            name="epitope prior 0",
                            legendgroup="e0",
                            showlegend=(i == 0 and j == 0),
                            marker=dict(size=4, opacity=0.22, color="#6b7280"),
                        ),
                        row=row_idx,
                        col=col_idx,
                    )
                if mask1.any():
                    fig.add_trace(
                        go.Scatter(
                            x=t[mask1],
                            y=p[mask1],
                            mode="markers",
                            name="epitope prior 1",
                            legendgroup="e1",
                            showlegend=(i == 0 and j == 0),
                            marker=dict(size=5, opacity=0.55, color="#dc2626"),
                        ),
                        row=row_idx,
                        col=col_idx,
                    )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=t,
                        y=p,
                        mode="markers",
                        name="residues",
                        showlegend=False,
                        marker=dict(size=4, opacity=0.25, color="#2563eb"),
                    ),
                    row=row_idx,
                    col=col_idx,
                )

            fig.layout.annotations[(i * 3) + j].text = f"{be} seed {sd}<br>r={r:.3f} n={t.size}"

    fig.update_xaxes(title_text="true DASA", range=[-0.02, 1.02])
    fig.update_yaxes(title_text="pred DASA", range=[-0.02, 1.02])
    fig.update_layout(
        title="Test set: predicted vs true DASA (colored by epitope_prior when runs_csv provides epitope_prior_file)",
        height=980,
        width=1100,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    tag = args.out_base_tag or out_base_tag()
    out = Path(args.output) if args.output else default_cohort_html("dasa_pred_vs_true_grid", tag)
    ensure_dir(out.parent)
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"Wrote {out.resolve()}")
    maybe_build_index(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
