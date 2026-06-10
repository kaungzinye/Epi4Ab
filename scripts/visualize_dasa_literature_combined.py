#!/usr/bin/env python3
"""
Per-residue plot: DASA (true / label / pred) and literature epitope on one shared axis.

X-axis: residue index along the graph (1..N) with hover auth_seq_id from lig.pdb.
Y-axis: [0, 1] — DASA as lines; literature epitope as semi-transparent bars.

Inputs (test inference + priors), same layout as visualize_dasa_backend_comparison.py runs_csv:
  run_id, nodes_edges_dir, test_record_dir, label_file, epitope_prior_file

Example:
  python scripts/visualize_dasa_literature_combined.py \\
    --runs_csv reports/dasa_runs_augmented_iedb_pdb.csv \\
    --run_id aug_biopy_s456 \\
    --processed_dir .../processed_data \\
    --output reports/dasa_literature_combined_aug_biopy_s456.html
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

from epi4ab_plot_paths import default_run_html, ensure_dir, maybe_build_index
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError as e:
    raise SystemExit("plotly is required: pip install plotly") from e


def _parse_lig_pdb_order(lig_path: Path) -> List[Dict]:
    seen = set()
    rows: List[Dict] = []
    with lig_path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            chain = (line[21:22].strip() or "A")
            raw = line[22:26].strip()
            icode = line[26:27].strip()
            if not raw:
                continue
            try:
                resseq = int(raw)
            except ValueError:
                continue
            key = (chain, resseq, icode)
            if key in seen:
                continue
            seen.add(key)
            rows.append({"auth_asym_id": chain, "auth_seq_id": resseq, "pdbx_PDB_ins_code": icode})
    return rows


def _occurrence_map(keys: List[int]) -> Dict[Tuple[int, int], int]:
    """Map (resId, k-th occurrence) -> index in keys list."""
    by_res: Dict[int, List[int]] = defaultdict(list)
    for i, k in enumerate(keys):
        by_res[k].append(i)
    out: Dict[Tuple[int, int], int] = {}
    for res_id, idxs in by_res.items():
        for occ, i in enumerate(idxs):
            out[(res_id, occ)] = i
    return out


def _load_case_frame(
    pdb_id: str,
    nodes_edges: Path,
    test_record: Path,
    label_file: str,
    epitope_file: str,
    cov_epitope_file: Optional[str],
    lig_path: Path,
    epitope_assay_file: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    pred_path = test_record / f"{pdb_id}_final_result.txt"
    lab_path = nodes_edges / pdb_id / label_file
    epi_path = nodes_edges / pdb_id / epitope_file
    if not pred_path.is_file() or not lab_path.is_file() or not epi_path.is_file():
        return None

    pred = pd.read_csv(pred_path, sep="\t")
    lab = pd.read_parquet(lab_path)
    epi = pd.read_parquet(epi_path)
    if "res_id" not in pred.columns:
        return None

    mer = pred.merge(lab[["resId", "score"]], left_on="res_id", right_on="resId", how="left")
    mer = mer.merge(epi[["resId", "epitope_prior"]], on="resId", how="left", suffixes=("", "_epi"))
    mer["literature_epitope_union"] = mer["epitope_prior"].fillna(0).astype(int)
    mer["literature_epitope"] = mer["literature_epitope_union"]

    if epitope_assay_file:
        apath = nodes_edges / pdb_id / epitope_assay_file
        if apath.is_file():
            aep = pd.read_parquet(apath)
            if "resId" in aep.columns and "epitope_prior" in aep.columns:
                mer = mer.merge(
                    aep[["resId", "epitope_prior"]].rename(
                        columns={"epitope_prior": "epitope_prior_assay"}
                    ),
                    on="resId",
                    how="left",
                )
                mer["literature_epitope_assay"] = mer["epitope_prior_assay"].fillna(0).astype(int)
            else:
                mer["literature_epitope_assay"] = 0
        else:
            mer["literature_epitope_assay"] = 0
    else:
        mer["literature_epitope_assay"] = mer["literature_epitope_union"]

    if cov_epitope_file:
        cov_path = nodes_edges / pdb_id / cov_epitope_file
        if cov_path.is_file():
            cov = pd.read_parquet(cov_path)
            if "resId" in cov.columns and "epitope_prior" in cov.columns:
                mer = mer.merge(
                    cov[["resId", "epitope_prior"]].rename(columns={"epitope_prior": "cov_prior"}),
                    on="resId",
                    how="left",
                )
                cov_b = mer["cov_prior"].fillna(0).astype(int)
                lit = mer["literature_epitope"]
                mer["literature_epitope"] = np.maximum(lit, cov_b)
                mer["epitope_source"] = np.where(
                    (lit == 1) & (cov_b == 1),
                    "IEDB+CoV",
                    np.where(lit == 1, "IEDB", np.where(cov_b == 1, "CoV", "")),
                )
            else:
                mer["epitope_source"] = np.where(mer["literature_epitope"] == 1, "IEDB", "")
        else:
            mer["epitope_source"] = np.where(mer["literature_epitope"] == 1, "IEDB", "")
    else:
        mer["epitope_source"] = np.where(mer["literature_epitope"] == 1, "IEDB", "")

    lig_rows = _parse_lig_pdb_order(lig_path) if lig_path.is_file() else []
    lig_keys = [r["auth_seq_id"] for r in lig_rows]
    lig_occ = _occurrence_map(lig_keys)

    seen_count: Dict[int, int] = defaultdict(int)
    auth_seq = []
    auth_chain = []
    auth_ins = []
    for _, row in mer.iterrows():
        rid = int(row["res_id"])
        occ = seen_count[rid]
        seen_count[rid] += 1
        li = lig_occ.get((rid, occ))
        if li is not None and li < len(lig_rows):
            auth_seq.append(int(lig_rows[li]["auth_seq_id"]))
            auth_chain.append(str(lig_rows[li]["auth_asym_id"]))
            auth_ins.append(str(lig_rows[li].get("pdbx_PDB_ins_code") or ""))
        else:
            auth_seq.append(rid)
            auth_chain.append(lig_rows[0]["auth_asym_id"] if lig_rows else "?")
            auth_ins.append("")

    mer["auth_seq_id"] = auth_seq
    mer["auth_asym_id"] = auth_chain
    mer["pdbx_PDB_ins_code"] = auth_ins
    mer["residue_index"] = np.arange(1, len(mer) + 1)

    if "true_score" in mer.columns:
        mer["dasa_true"] = pd.to_numeric(mer["true_score"], errors="coerce")
    else:
        mer["dasa_true"] = np.nan
    mer["dasa_label"] = pd.to_numeric(mer["score"], errors="coerce")
    mer["dasa_pred"] = pd.to_numeric(mer["pred_score"], errors="coerce")
    return mer


def _label_matches_true(df: pd.DataFrame) -> bool:
    if "dasa_label" not in df.columns or "dasa_true" not in df.columns:
        return False
    a = df["dasa_label"].to_numpy(dtype=float)
    b = df["dasa_true"].to_numpy(dtype=float)
    if not (np.isfinite(a).any() and np.isfinite(b).any()):
        return False
    m = np.isfinite(a) & np.isfinite(b)
    if not m.any():
        return False
    return float(np.max(np.abs(a[m] - b[m]))) < 1e-5


def _add_combined_trace(
    fig: go.Figure,
    df: pd.DataFrame,
    row: int,
    col: int,
    pdb_id: str,
    dual_track: bool = False,
) -> None:
    x = df["residue_index"]
    lit_union = df["literature_epitope_union"].astype(int)
    lit_assay = df["literature_epitope_assay"].astype(int)

    # Union track (loose): lighter red bars
    mask_u = lit_union.to_numpy() == 1
    if mask_u.any():
        fig.add_trace(
            go.Bar(
                x=x[mask_u],
                y=np.full(mask_u.sum(), 0.55),
                name="literature union (any assay)",
                marker=dict(color="rgba(248, 113, 113, 0.35)", line=dict(width=0)),
                width=0.9,
                hovertemplate="idx=%{x}<br>lit union<extra></extra>",
                legendgroup="lit_union",
                showlegend=(row == 1 and col == 1),
            ),
            row=row,
            col=col,
        )

    # Assay-matched track (tight): solid red bars
    mask_a = lit_assay.to_numpy() == 1
    if mask_a.any():
        fig.add_trace(
            go.Bar(
                x=x[mask_a],
                y=np.ones(mask_a.sum(), dtype=float),
                name="literature matched assay",
                marker=dict(color="rgba(220, 38, 38, 0.75)", line=dict(width=0)),
                width=0.65,
                hovertemplate=(
                    "idx=%{x}<br>lit assay<br>auth_seq=%{customdata[0]}<br>"
                    "chain=%{customdata[1]}<extra></extra>"
                ),
                customdata=np.stack(
                    [df.loc[mask_a, "auth_seq_id"], df.loc[mask_a, "auth_asym_id"]], axis=1
                ),
                legendgroup="lit_assay",
                showlegend=(row == 1 and col == 1),
            ),
            row=row,
            col=col,
        )

    # Blue = actual DASA on test (true_score). Orange = model prediction.
    # Training label parquet (score) is omitted when identical to true_score.
    if df["dasa_true"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=x,
                y=df["dasa_true"],
                mode="lines+markers",
                name="DASA actual (true_score)",
                line=dict(color="#2563eb", width=2),
                marker=dict(size=3),
                hovertemplate="idx=%{x}<br>actual=%{y:.3f}<extra></extra>",
                legendgroup="actual",
                showlegend=(row == 1 and col == 1),
            ),
            row=row,
            col=col,
        )
    elif df["dasa_label"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=x,
                y=df["dasa_label"],
                mode="lines+markers",
                name="DASA actual (label parquet)",
                line=dict(color="#2563eb", width=2),
                marker=dict(size=3),
                hovertemplate="idx=%{x}<br>actual=%{y:.3f}<extra></extra>",
                legendgroup="actual",
                showlegend=(row == 1 and col == 1),
            ),
            row=row,
            col=col,
        )

    if df["dasa_pred"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=x,
                y=df["dasa_pred"],
                mode="lines",
                name="DASA predicted (pred_score)",
                line=dict(color="#f59e0b", width=1.5, dash="dash"),
                hovertemplate="idx=%{x}<br>pred=%{y:.3f}<extra></extra>",
                legendgroup="pred",
                showlegend=(row == 1 and col == 1),
            ),
            row=row,
            col=col,
        )

    fig.update_xaxes(title_text="residue index (graph order)", row=row, col=col)
    fig.update_yaxes(title_text="DASA score 0–1 (lines); lit bar height=1", range=[-0.02, 1.08], row=row, col=col)


def main() -> int:
    ap = argparse.ArgumentParser(description="DASA + literature epitope on one axis per PDB.")
    ap.add_argument("--runs_csv", required=True)
    ap.add_argument("--run_id", required=True, help="Which row in runs_csv to plot")
    ap.add_argument("--processed_dir", required=True, help="processed_data root for lig.pdb")
    ap.add_argument(
        "--output",
        default=None,
        help="Output HTML (default: plots/runs/<run_id>/html/dasa_literature_combined.html)",
    )
    ap.add_argument("--out_base_tag", default=None)
    ap.add_argument("--build-index", action="store_true")
    ap.add_argument(
        "--cov_epitope_prior_file",
        default=None,
        help="Optional second prior, e.g. node_covabdab_epitope_prior.parquet (max with IEDB)",
    )
    ap.add_argument(
        "--epitope_assay_prior_file",
        default=None,
        help="Tier B per-assay prior, e.g. node_iedb_pdb_epitope_prior_assay.parquet",
    )
    ap.add_argument(
        "--dual_track",
        action="store_true",
        help="Show union (light) + matched assay (solid) bars; requires --epitope_assay_prior_file",
    )
    ap.add_argument("--cols", type=int, default=2, help="Subplot columns (default 2)")
    ap.add_argument("--width", type=int, default=1400, help="Figure width in px")
    ap.add_argument(
        "--height_per_row",
        type=int,
        default=480,
        help="Figure height per subplot row in px (taller = easier to read 0–1 DASA axis)",
    )
    args = ap.parse_args()

    runs = pd.read_csv(args.runs_csv, comment="#")
    sub = runs.loc[runs["run_id"].astype(str) == str(args.run_id)]
    if sub.empty:
        print(f"run_id {args.run_id!r} not found in {args.runs_csv}", file=sys.stderr)
        return 2
    run = sub.iloc[0]
    backend = str(run["backend"]) if "backend" in run.index else "?"
    split_seed = str(run["split_seed"]) if "split_seed" in run.index else "?"
    nodes_edges = Path(str(run["nodes_edges_dir"]))
    test_record = Path(str(run["test_record_dir"]))
    label_file = str(run["label_file"])
    epitope_file = str(run["epitope_prior_file"])
    assay_file = args.epitope_assay_prior_file
    if args.dual_track and not assay_file:
        assay_file = "node_iedb_pdb_epitope_prior_assay.parquet"
    processed = Path(args.processed_dir)

    cases: List[Tuple[str, pd.DataFrame]] = []
    missing: List[str] = []
    for pred_path in sorted(test_record.glob("*_final_result.txt")):
        pdb_id = pred_path.stem.replace("_final_result", "")
        lig = processed / pdb_id / "lig.pdb"
        df = _load_case_frame(
            pdb_id,
            nodes_edges,
            test_record,
            label_file,
            epitope_file,
            args.cov_epitope_prior_file,
            lig,
            assay_file,
        )
        if df is None or df.empty:
            missing.append(pdb_id)
            continue
        cases.append((pdb_id, df))

    if not cases:
        print("No cases loaded.", file=sys.stderr)
        return 1

    n = len(cases)
    n_cols = max(1, args.cols)
    n_rows = (n + n_cols - 1) // n_cols
    titles = []
    for pid, df in cases:
        if args.dual_track or assay_file:
            titles.append(
                f"{pid}<br>lit assay={int(df['literature_epitope_assay'].sum())}/"
                f"{len(df)} union={int(df['literature_epitope_union'].sum())}"
            )
        else:
            titles.append(
                f"{pid}<br>lit+={int(df['literature_epitope_union'].sum())}/{len(df)}"
            )
    # Plotly requires vertical_spacing <= 1/(n_rows-1); cap so 2-column grids with many PDBs work.
    if n_rows > 1:
        vspace = min(0.06, 0.90 / (n_rows - 1))
    else:
        vspace = 0.08
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=titles,
        vertical_spacing=vspace,
        horizontal_spacing=0.08,
    )

    for i, (pdb_id, df) in enumerate(cases):
        r = i // n_cols + 1
        c = i % n_cols + 1
        _add_combined_trace(fig, df, r, c, pdb_id, dual_track=bool(args.dual_track or assay_file))

    lit_note = (
        "light red = union (any assay on PDB+chain); solid red = matched single assay."
        if (args.dual_track or assay_file)
        else "red bars = literature epitope."
    )
    dasa_note = (
        f"DASA labels: {label_file} (ASA backend={backend}, train/test seed={split_seed}). "
        "Blue = test true_score; orange = model pred_score."
    )
    fig_height = max(500, args.height_per_row * n_rows)
    fig.update_layout(
        title=(
            f"DASA + literature — {args.run_id}<br>"
            f"<sup>{dasa_note} {lit_note}</sup>"
        ),
        height=fig_height,
        width=args.width,
        barmode="overlay",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    # Keep subplot titles readable without stealing vertical space from the 0–1 y-axis.
    fig.update_annotations(font_size=11)

    out = (
        Path(args.output)
        if args.output
        else default_run_html(str(args.run_id), "dasa_literature_combined")
    )
    ensure_dir(out.parent)
    fig.write_html(out, include_plotlyjs="cdn")
    maybe_build_index(args)
    print(f"Wrote {out.resolve()} ({len(cases)} PDBs, missing {len(missing)})")
    if missing:
        print(f"Missing: {', '.join(missing)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
