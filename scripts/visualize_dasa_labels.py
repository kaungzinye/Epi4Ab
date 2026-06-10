#!/usr/bin/env python3
"""
Build an HTML dashboard from node_label_dasa.parquet files.

Reads each pdbID from a metadata CSV, loads nodes_edges/<pdbID>/<label_file> (default node_label_dasa.parquet),
writes a summary table, distribution plots, and per-residue traces for a sample of complexes.

Usage:
  python scripts/visualize_dasa_labels.py \\
    --nodes_edges_dir <OUT_BASE>/nodes_edges \\
    --metadata <OUT_BASE>/gates/fill_edge/metadata.ok.fill_edge.csv \\
    --output <OUT_BASE>/reports/dasa_label_dashboard.html

Optional:
  --sample_detailed 12   # number of PDBs to plot along the sequence (default 12)
  --label_file           # parquet name under each pdb dir (default node_label_dasa.parquet)
  --epitope_prior_file   # e.g. node_iedb_epitope_prior.parquet — overlays epitope residues on traces
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from epi4ab_plot_paths import default_cohort_html, ensure_dir, maybe_build_index, out_base_tag
from typing import List, Optional

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError as e:
    raise SystemExit("plotly is required: pip install plotly") from e


def _read_pdb_ids(metadata_path: str) -> List[str]:
    df = pd.read_csv(metadata_path, comment="#")
    for c in ("pdbID", "pdbId", "pdb_id"):
        if c in df.columns:
            return df[c].astype(str).tolist()
    return df.iloc[:, 0].astype(str).tolist()


def main() -> int:
    ap = argparse.ArgumentParser(description="Visualize DASA label parquets.")
    ap.add_argument("--nodes_edges_dir", required=True)
    ap.add_argument("--metadata", required=True, help="CSV with pdbID column")
    ap.add_argument(
        "--label_file",
        default="node_label_dasa.parquet",
        help="Label parquet filename under each pdb dir",
    )
    ap.add_argument(
        "--epitope_prior_file",
        default=None,
        help="Optional per-PDB parquet with resId + epitope_prior (0/1), e.g. node_iedb_epitope_prior.parquet",
    )
    ap.add_argument(
        "--dasa_epitope_threshold",
        type=float,
        default=0.85,
        help="For metrics: binarize DASA score at this cutoff vs epitope_prior",
    )
    ap.add_argument(
        "--output",
        default=None,
        help="Output HTML (default: plots/cohort/<tag>/html/dasa_label_dashboard.html)",
    )
    ap.add_argument("--out_base_tag", default=None)
    ap.add_argument("--build-index", action="store_true")
    ap.add_argument(
        "--sample_detailed",
        type=int,
        default=12,
        help="How many PDBs get full resId vs score subplots (default 12)",
    )
    ap.add_argument("--seed", type=int, default=42, help="RNG seed for sampling PDBs")
    args = ap.parse_args()

    root = Path(args.nodes_edges_dir)
    pdb_ids = _read_pdb_ids(args.metadata)
    rows = []
    missing = []

    label_name = args.label_file
    epi_name = args.epitope_prior_file

    for pid in pdb_ids:
        p = root / pid / label_name
        if not p.exists():
            missing.append(pid)
            continue
        df = pd.read_parquet(p)
        score = df["score"].to_numpy(dtype=float)
        dclip = df["delta_asa_clipped"].to_numpy(dtype=float)
        d = df["delta_asa"].to_numpy(dtype=float)
        row_d = {
            "pdb_id": pid,
            "n_res": len(df),
            "score_mean": float(np.mean(score)),
            "score_std": float(np.std(score)),
            "score_min": float(np.min(score)),
            "score_max": float(np.max(score)),
            "delta_pos_frac": float(np.mean(d > 0)),
            "delta_clip_sum": float(np.sum(dclip)),
        }
        if epi_name:
            ep = root / pid / epi_name
            if ep.exists():
                edf = pd.read_parquet(ep)
                if (
                    "resId" in edf.columns
                    and "epitope_prior" in edf.columns
                    and len(edf) == len(df)
                ):
                    mer = df.merge(
                        edf[["resId", "epitope_prior"]], on="resId", how="left"
                    )
                    y = mer["epitope_prior"].fillna(0).astype(int).to_numpy()
                    dasa_bin = (mer["score"].to_numpy(dtype=float) >= args.dasa_epitope_threshold).astype(int)
                    tp = int(np.sum((dasa_bin == 1) & (y == 1)))
                    fp = int(np.sum((dasa_bin == 1) & (y == 0)))
                    fn = int(np.sum((dasa_bin == 0) & (y == 1)))
                    tn = int(np.sum((dasa_bin == 0) & (y == 0)))
                    denom = (tp + fp + fn + tn) or 1
                    row_d["iedb_epitope_frac"] = float(np.mean(y))
                    row_d["dasa_vs_prior_acc"] = float((tp + tn) / denom)
                    d_mcc = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
                    row_d["dasa_vs_prior_mcc"] = (
                        float((tp * tn - fp * fn) / d_mcc) if d_mcc > 0 else 0.0
                    )
                else:
                    row_d["iedb_epitope_frac"] = float("nan")
            else:
                row_d["iedb_epitope_frac"] = float("nan")
        rows.append(row_d)

    if not rows:
        print(f"No {label_name} files found for listed PDBs.", file=sys.stderr)
        return 1

    summary = pd.DataFrame(rows).sort_values("pdb_id")
    tag = args.out_base_tag or out_base_tag()
    out_path = Path(args.output) if args.output else default_cohort_html("dasa_label_dashboard", tag)
    ensure_dir(out_path.parent)

    # Sample PDBs for detailed multi-panel plot
    rng = np.random.default_rng(args.seed)
    pool = summary["pdb_id"].tolist()
    k = min(args.sample_detailed, len(pool))
    sample = list(rng.choice(pool, size=k, replace=False))
    sample.sort()

    fig_overview = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Per-complex mean score", "Fraction of residues with delta_asa > 0"),
    )
    fig_overview.add_trace(
        go.Histogram(x=summary["score_mean"], nbinsx=30, name="mean score"),
        row=1,
        col=1,
    )
    fig_overview.add_trace(
        go.Histogram(x=summary["delta_pos_frac"], nbinsx=30, name="frac dASA>0"),
        row=1,
        col=2,
    )
    fig_overview.update_layout(
        title_text=f"DASA labels overview (n={len(summary)} complexes)",
        showlegend=False,
        height=400,
    )

    n_cols = 3
    n_rows = (len(sample) + n_cols - 1) // n_cols
    fig_detail = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=sample,
        vertical_spacing=0.09,
        horizontal_spacing=0.06,
    )
    for i, pid in enumerate(sample):
        r = i // n_cols + 1
        c = i % n_cols + 1
        df = pd.read_parquet(root / pid / label_name)
        fig_detail.add_trace(
            go.Scatter(
                x=df["resId"],
                y=df["score"],
                mode="lines",
                name=pid,
                line=dict(width=1),
                hovertemplate="resId=%{x}<br>score=%{y:.3f}<extra></extra>",
            ),
            row=r,
            col=c,
        )
        if epi_name:
            ep = root / pid / epi_name
            if ep.exists():
                edf = pd.read_parquet(ep)
                if "epitope_prior" in edf.columns and "resId" in edf.columns:
                    mer = df.merge(edf[["resId", "epitope_prior"]], on="resId", how="left")
                    mask = mer["epitope_prior"].fillna(0).astype(int).to_numpy() == 1
                    if mask.any():
                        mx = mer[mask]
                        fig_detail.add_trace(
                            go.Scatter(
                                x=mx["resId"],
                                y=mx["score"],
                                mode="markers",
                                name=f"{pid}_epi",
                                marker=dict(size=5, color="red", symbol="circle-open"),
                                hovertemplate="IEDB prior resId=%{x}<br>score=%{y:.3f}<extra></extra>",
                            ),
                            row=r,
                            col=c,
                        )
        fig_detail.update_yaxes(title_text="score", row=r, col=c, range=[-0.05, 1.05])
        fig_detail.update_xaxes(title_text="resId", row=r, col=c)

    fig_detail.update_layout(
        title_text="Per-residue normalized DASA score (sample)",
        height=220 * n_rows,
        showlegend=False,
    )

    summary_html = summary.to_html(index=False, float_format=lambda x: f"{x:.4f}")

    miss_line = ""
    if missing:
        miss_line = f"<p><b>Missing labels ({len(missing)}):</b> {', '.join(missing[:40])}"
        if len(missing) > 40:
            miss_line += f" … and {len(missing) - 40} more"
        miss_line += "</p>"

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/><title>DASA label dashboard</title></head>
<body>
<h1>DASA label verification</h1>
<p>nodes_edges_dir: {root}</p>
<p>metadata: {args.metadata}</p>
<p>Complexes with labels: {len(summary)} / {len(pdb_ids)} listed in metadata.</p>
{miss_line}
<h2>Summary table</h2>
{summary_html}
{fig_overview.to_html(full_html=False, include_plotlyjs="cdn")}
{fig_detail.to_html(full_html=False, include_plotlyjs=False)}
</body></html>"""

    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote {out_path.resolve()}")
    maybe_build_index(args)
    if missing:
        print(f"Warning: {len(missing)} PDBs had no {label_name}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
