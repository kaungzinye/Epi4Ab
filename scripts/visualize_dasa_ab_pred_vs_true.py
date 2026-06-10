#!/usr/bin/env python3
"""Plot pred-vs-true DASA for arbitrary A/B regression runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from epi4ab_plot_paths import cohort_html_dir, ensure_dir, maybe_build_index, out_base_tag

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _load_run(test_record_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    true_parts: list[np.ndarray] = []
    pred_parts: list[np.ndarray] = []
    for path in sorted(test_record_dir.glob("*_final_result.txt")):
        df = pd.read_csv(path, sep="\t")
        if {"true_score", "pred_score"}.issubset(df.columns):
            sub = df.dropna(subset=["true_score", "pred_score"])
            true_parts.append(sub["true_score"].to_numpy(dtype=float))
            pred_parts.append(sub["pred_score"].to_numpy(dtype=float))
    if not true_parts:
        return np.array([]), np.array([])
    return np.concatenate(true_parts), np.concatenate(pred_parts)


def _pearson(true: np.ndarray, pred: np.ndarray) -> float:
    if true.size < 2 or np.std(true) == 0 or np.std(pred) == 0:
        return float("nan")
    return float(np.corrcoef(true, pred)[0, 1])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs_csv", required=True)
    ap.add_argument(
        "--output",
        default=None,
        help="Output HTML (default: plots/cohort/<tag>/html/dasa_ab_pred_vs_true.html)",
    )
    ap.add_argument("--out_base_tag", default=None)
    ap.add_argument("--build-index", action="store_true")
    ap.add_argument("--max_points_per_run", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    runs = pd.read_csv(args.runs_csv, comment="#")
    fig = make_subplots(rows=1, cols=len(runs), subplot_titles=list(runs["run_id"]))
    rng = np.random.default_rng(args.seed)

    summary = []
    for idx, row in runs.reset_index(drop=True).iterrows():
        run_id = str(row["run_id"])
        test_record_dir = Path(str(row["test_record_dir"]))
        true, pred = _load_run(test_record_dir)
        if true.size == 0:
            fig.add_annotation(text="no data", showarrow=False, row=1, col=idx + 1)
            summary.append({"run_id": run_id, "n": 0, "pearson": np.nan, "mse": np.nan})
            continue

        if true.size > args.max_points_per_run:
            keep = rng.choice(true.size, size=args.max_points_per_run, replace=False)
            true_plot = true[keep]
            pred_plot = pred[keep]
        else:
            true_plot = true
            pred_plot = pred

        r = _pearson(true, pred)
        mse = float(np.mean((pred - true) ** 2))
        summary.append({"run_id": run_id, "n": int(true.size), "pearson": r, "mse": mse})

        fig.add_trace(
            go.Scattergl(
                x=true_plot,
                y=pred_plot,
                mode="markers",
                marker={"size": 4, "opacity": 0.35},
                name=run_id,
                showlegend=False,
            ),
            row=1,
            col=idx + 1,
        )
        fig.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line={"dash": "dash"}, showlegend=False),
            row=1,
            col=idx + 1,
        )
        fig.update_xaxes(title_text="true DASA", range=[0, 1], row=1, col=idx + 1)
        fig.update_yaxes(title_text="pred DASA", range=[0, 1], row=1, col=idx + 1)

    summary_df = pd.DataFrame(summary)
    tag = args.out_base_tag or out_base_tag()
    out = (
        Path(args.output)
        if args.output
        else cohort_html_dir(tag) / "dasa_ab_pred_vs_true.html"
    )
    ensure_dir(out.parent)
    from epi4ab_plot_paths import cohort_csv_dir

    summary_path = cohort_csv_dir(tag) / "dasa_ab_pred_vs_true.csv"
    ensure_dir(summary_path.parent)
    summary_df.to_csv(summary_path, index=False)

    fig.update_layout(title_text="DASA A/B pred-vs-true", height=520, width=max(600, 520 * len(runs)))
    body = f"""<!doctype html>
<html><head><meta charset="utf-8"/><title>DASA A/B pred-vs-true</title></head>
<body>
<h1>DASA A/B pred-vs-true</h1>
<p>runs_csv: {args.runs_csv}</p>
<p>summary_csv: {summary_path.resolve()}</p>
{summary_df.to_html(index=False, float_format=lambda x: f"{x:.4f}")}
{fig.to_html(full_html=False, include_plotlyjs="cdn")}
</body></html>"""
    out.write_text(body, encoding="utf-8")
    print(f"Wrote {out.resolve()} and {summary_path.resolve()}")
    maybe_build_index(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
