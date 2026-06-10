#!/usr/bin/env python3
"""Epi4Ab Regression Results Dashboard.

This replaces the old classification visualization.

Expected input files in a `test_record/` folder:
- `<pdb_id>_final_result.txt` with columns:
  - res_id, res_name, pred_score
  - optional: pred_prob (if sigmoid), true_score

Outputs:
- `<test_record_dir>/regression_summary.csv`
- `<parent_of_test_record_dir>/dashboard.html`
- Optional per-PDB HTML: `<parent>/visualizations/<pdb_id>.html`

Usage:
  python scripts/visualize_results.py --test_record_dir <...>/test_record
  python scripts/visualize_results.py --test_record_dir <...>/test_record --pdb_id 1n8z_BAC
"""

import argparse
from pathlib import Path

from epi4ab_plot_paths import (
    default_run_html,
    ensure_dir,
    maybe_build_index,
    register_run,
    resolve_run_id,
    run_csv_dir,
    run_html_dir,
)

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError as e:
    raise SystemExit('plotly is required: pip install plotly') from e


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1])


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = pd.Series(a).rank(method='average').to_numpy(dtype=float)
    rb = pd.Series(b).rank(method='average').to_numpy(dtype=float)
    return float(np.corrcoef(ra, rb)[0, 1])


def load_final_results(test_record_dir: Path, pdb_id: str) -> pd.DataFrame:
    path = test_record_dir / f'{pdb_id}_final_result.txt'
    if not path.exists():
        raise FileNotFoundError(str(path))
    return pd.read_csv(path, sep='\t')


def load_all_pdb_results(test_record_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    files = sorted(test_record_dir.glob('*_final_result.txt'))
    pdb_ids = [f.stem.replace('_final_result', '') for f in files]
    if not pdb_ids:
        raise FileNotFoundError(f'No *_final_result.txt found in {test_record_dir}')

    dfs = []
    for pdb_id in pdb_ids:
        df = load_final_results(test_record_dir, pdb_id)
        df['pdb_id'] = pdb_id
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True), pdb_ids


def per_pdb_metrics(df_all: pd.DataFrame) -> pd.DataFrame:
    rows = []
    true_kind = df_all.attrs.get('true_kind', 'unknown_or_prob')
    pred_kind = df_all.attrs.get('pred_kind', 'unit_interval')
    for pdb_id, g in df_all.groupby('pdb_id'):
        pred = g['pred_score'].to_numpy(dtype=float)
        has_true = 'true_score' in g.columns and g['true_score'].notna().any()

        if has_true:
            if true_kind == 'proteinmpnn_nll':
                # If the model predicts raw NLL, compare against raw NLL.
                # If the model predicts 0-1 (sigmoid), compare against normalized NLL.
                if pred_kind == 'raw_nll':
                    true = g['true_score'].to_numpy(dtype=float)
                else:
                    true = g['true_nll_norm'].to_numpy(dtype=float) if 'true_nll_norm' in g.columns else g['true_score'].to_numpy(dtype=float)
            else:
                true = g['true_score'].to_numpy(dtype=float)
            mse = float(np.mean((pred - true) ** 2))
            mae = float(np.mean(np.abs(pred - true)))
            pear = _pearson(pred, true) if len(pred) > 1 else float('nan')
            spear = _spearman(pred, true) if len(pred) > 1 else float('nan')
        else:
            mse = float('nan')
            mae = float('nan')
            pear = float('nan')
            spear = float('nan')

        row = {
            'pdb_id': pdb_id,
            'n_res': int(len(g)),
            'mse': mse,
            'mae': mae,
            'pearson': pear,
            'spearman': spear,
            'pred_mean': float(np.mean(pred)),
            'pred_std': float(np.std(pred)),
        }
        if true_kind == 'proteinmpnn_nll' and has_true:
            row['true_nll_mean'] = float(np.mean(g['true_score'].to_numpy(dtype=float)))
            row['true_nll_std'] = float(np.std(g['true_score'].to_numpy(dtype=float)))
        rows.append(row)

    return pd.DataFrame(rows).sort_values('pdb_id')


def _is_nll_like_true_score(df_all: pd.DataFrame) -> bool:
    # ProteinMPNN per-residue "score" in our Phase 1 pipeline is a negative
    # log-probability (NLL) for the native amino acid. Typical ranges are > 1.
    if 'true_score' not in df_all.columns:
        return False
    try:
        return float(df_all['true_score'].max()) > 1.5
    except Exception:
        return False


def _is_nll_like_pred_score(df_all: pd.DataFrame) -> bool:
    if 'pred_score' not in df_all.columns:
        return False
    try:
        return float(df_all['pred_score'].max()) > 1.5
    except Exception:
        return False


def add_derived_columns(df_all: pd.DataFrame) -> pd.DataFrame:
    df_all = df_all.copy()
    if 'true_score' in df_all.columns:
        df_all['true_score'] = df_all['true_score'].astype(float)
    df_all['pred_score'] = df_all['pred_score'].astype(float)

    if _is_nll_like_true_score(df_all):
        # Map NLL into [0,1] so regression plots/errors are interpretable.
        # Lower NLL = easier/more probable; higher NLL = harder/less probable.
        nll_min = float(df_all['true_score'].min())
        nll_max = float(df_all['true_score'].max())
        denom = (nll_max - nll_min) if (nll_max > nll_min) else 1.0
        df_all['true_nll'] = df_all['true_score']
        df_all['true_nll_norm'] = (df_all['true_nll'] - nll_min) / denom
        df_all['true_native_prob'] = np.exp(-df_all['true_nll'].to_numpy(dtype=float))
        df_all.attrs['true_kind'] = 'proteinmpnn_nll'
        df_all.attrs['nll_min'] = nll_min
        df_all.attrs['nll_max'] = nll_max
    else:
        df_all.attrs['true_kind'] = 'unknown_or_prob'

    # Predicted score kind (Phase 1 can be either sigmoid 0-1 or raw NLL)
    df_all.attrs['pred_kind'] = 'raw_nll' if _is_nll_like_pred_score(df_all) else 'unit_interval'
    return df_all


def make_pdb_plot(df: pd.DataFrame, pdb_id: str, out_html: Path):
    has_true = 'true_score' in df.columns and df['true_score'].notna().any()
    x = np.arange(len(df))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=df['pred_score'], mode='lines', name='pred_score'))
    if has_true:
        fig.add_trace(go.Scatter(x=x, y=df['true_score'], mode='lines', name='true_score'))

    fig.update_layout(
        title=f'{pdb_id} per-residue scores',
        xaxis_title='Residue index (0-based order in node_feature)',
        yaxis_title='Score (pred_score); true_score as provided',
        height=520,
    )
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html), include_plotlyjs='cdn')


def make_dashboard(df_all: pd.DataFrame, metrics: pd.DataFrame, out_html: Path):
    has_true = 'true_score' in df_all.columns and df_all['true_score'].notna().any()

    true_kind = df_all.attrs.get('true_kind', 'unknown_or_prob')
    pred_kind = df_all.attrs.get('pred_kind', 'unit_interval')
    nll_min = df_all.attrs.get('nll_min', None)
    nll_max = df_all.attrs.get('nll_max', None)

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            'Predicted score distribution',
            'Predicted vs true (all residues)' if has_true else 'Predicted score (true_score not present)',
            'Per-PDB error' if has_true else 'Per-PDB predicted mean/std',
            'Per-PDB summary table',
        ),
        specs=[[{'type': 'histogram'}, {'type': 'scatter'}], [{'type': 'bar'}, {'type': 'table'}]],
    )

    if true_kind == 'proteinmpnn_nll' and pred_kind == 'raw_nll':
        fig.add_trace(go.Histogram(x=df_all['pred_score'], nbinsx=60, name='pred_nll'), row=1, col=1)
        fig.add_trace(go.Histogram(x=df_all['true_score'], nbinsx=60, name='true_nll', opacity=0.6), row=1, col=1)
    elif true_kind == 'proteinmpnn_nll':
        fig.add_trace(go.Histogram(x=df_all['pred_score'], nbinsx=60, name='pred_score (0-1)'), row=1, col=1)
        fig.add_trace(go.Histogram(x=df_all['true_nll_norm'], nbinsx=60, name='true_nll_norm (0-1)', opacity=0.6), row=1, col=1)
    else:
        fig.add_trace(go.Histogram(x=df_all['pred_score'], nbinsx=60, name='pred_score'), row=1, col=1)
    if has_true:
        if true_kind == 'proteinmpnn_nll' and pred_kind == 'raw_nll':
            x_true = df_all['true_score']
            x_label = 'True ProteinMPNN NLL (raw; higher=less probable native AA)'
            y_label = 'Predicted NLL (raw)'
        elif true_kind == 'proteinmpnn_nll':
            x_true = df_all['true_nll_norm']
            x_label = 'True ProteinMPNN NLL (normalized to 0-1; higher=less probable native AA)'
            y_label = 'Predicted score (0-1; trained against ProteinMPNN target)'
        else:
            x_true = df_all['true_score']
            x_label = 'True score'
            y_label = 'Predicted score'

        fig.add_trace(
            go.Scatter(
                x=x_true,
                y=df_all['pred_score'],
                mode='markers',
                marker={'size': 4, 'opacity': 0.35},
                name='residues'
            ),
            row=1, col=2
        )

        if true_kind == 'proteinmpnn_nll' and pred_kind == 'raw_nll':
            fig.add_trace(go.Bar(x=metrics['pdb_id'], y=metrics['mse'], name='mse (pred_nll vs true_nll)'), row=2, col=1)
            fig.add_trace(go.Bar(x=metrics['pdb_id'], y=metrics['mae'], name='mae (pred_nll vs true_nll)'), row=2, col=1)
            table_cols = ['pdb_id', 'n_res', 'mse', 'mae', 'pearson', 'spearman', 'pred_mean', 'pred_std', 'true_nll_mean']
        elif true_kind == 'proteinmpnn_nll':
            fig.add_trace(go.Bar(x=metrics['pdb_id'], y=metrics['mse'], name='mse (pred vs true_nll_norm)'), row=2, col=1)
            fig.add_trace(go.Bar(x=metrics['pdb_id'], y=metrics['mae'], name='mae (pred vs true_nll_norm)'), row=2, col=1)
            table_cols = ['pdb_id', 'n_res', 'mse', 'mae', 'pearson', 'spearman', 'pred_mean', 'pred_std']
            if 'true_nll_mean' in metrics.columns:
                table_cols += ['true_nll_mean']
        else:
            fig.add_trace(go.Bar(x=metrics['pdb_id'], y=metrics['mse'], name='mse'), row=2, col=1)
            fig.add_trace(go.Bar(x=metrics['pdb_id'], y=metrics['mae'], name='mae'), row=2, col=1)
            table_cols = ['pdb_id', 'n_res', 'mse', 'mae', 'pearson', 'spearman', 'pred_mean', 'pred_std']
    else:
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(df_all)),
                y=df_all['pred_score'],
                mode='markers',
                marker={'size': 3, 'opacity': 0.35},
                name='pred_score'
            ),
            row=1, col=2
        )
        fig.add_trace(go.Bar(x=metrics['pdb_id'], y=metrics['pred_mean'], name='pred_mean'), row=2, col=1)
        fig.add_trace(go.Bar(x=metrics['pdb_id'], y=metrics['pred_std'], name='pred_std'), row=2, col=1)
        table_cols = ['pdb_id', 'n_res', 'pred_mean', 'pred_std']

    metrics_disp = metrics.copy()
    for c in metrics_disp.columns:
        if c == 'pdb_id':
            continue
        if metrics_disp[c].dtype.kind in 'fc':
            metrics_disp[c] = metrics_disp[c].round(4)
    # Sort table by mse then mae if present
    if 'mse' in metrics_disp.columns:
        metrics_disp = metrics_disp.sort_values(['mse', 'mae'], na_position='last')

    fig.add_trace(
        go.Table(
            header={'values': table_cols},
            cells={'values': [metrics_disp[c].tolist() for c in table_cols]},
        ),
        row=2, col=2
    )

    if true_kind == 'proteinmpnn_nll' and pred_kind == 'raw_nll':
        hist_x = 'NLL'
    elif true_kind == 'proteinmpnn_nll':
        hist_x = 'Score (0-1)'
    else:
        hist_x = 'Predicted score'
    fig.update_xaxes(title_text=hist_x, row=1, col=1)
    fig.update_yaxes(title_text='Count', row=1, col=1)
    if has_true:
        fig.update_xaxes(title_text=x_label, row=1, col=2)
        fig.update_yaxes(title_text=y_label, row=1, col=2)

    title = 'Epi4Ab Regression Dashboard'
    if true_kind == 'proteinmpnn_nll' and (nll_min is not None) and (nll_max is not None):
        if pred_kind == 'raw_nll':
            title += f' (ProteinMPNN NLL targets: raw range {nll_min:.3f}..{nll_max:.3f})'
        else:
            title += f' (ProteinMPNN NLL targets: raw range {nll_min:.3f}..{nll_max:.3f}; plotted as normalized 0-1)'

    fig.update_layout(
        title=title,
        height=920,
        showlegend=True,
    )
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html), include_plotlyjs='cdn')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test_record_dir', required=True)
    ap.add_argument('--pdb_id', default='')
    ap.add_argument('--run_id', default=None, help='Hub run id (default: $RUN_ID or infer from path)')
    ap.add_argument('--build-index', action='store_true')
    args = ap.parse_args()

    test_record_dir = Path(args.test_record_dir)
    if not test_record_dir.exists():
        raise FileNotFoundError(str(test_record_dir))

    run_id = resolve_run_id(args.run_id)
    if not run_id:
        run_id = test_record_dir.parent.name
    out_parent = run_html_dir(run_id) if run_id else test_record_dir.parent
    csv_parent = run_csv_dir(run_id) if run_id else test_record_dir

    if args.pdb_id:
        df = load_final_results(test_record_dir, args.pdb_id)
        if 'pred_score' not in df.columns:
            raise KeyError('Expected pred_score column in *_final_result.txt (regression inference output)')
        viz_dir = out_parent / 'visualizations'
        out_html = viz_dir / f'{args.pdb_id}.html'
        make_pdb_plot(df, args.pdb_id, out_html)
        print(f'Wrote {out_html}')
        maybe_build_index(args)
        return

    df_all, _pdb_ids = load_all_pdb_results(test_record_dir)
    df_all = add_derived_columns(df_all)
    if 'pred_score' not in df_all.columns:
        raise KeyError('Expected pred_score column in *_final_result.txt (regression inference output)')

    metrics = per_pdb_metrics(df_all)
    ensure_dir(csv_parent)
    summary_path = csv_parent / 'regression_summary.csv'
    metrics.to_csv(summary_path, index=False)
    metrics.to_csv(test_record_dir / 'regression_summary.csv', index=False)
    out_html = default_run_html(run_id, 'dashboard') if run_id else test_record_dir.parent / 'dashboard.html'
    make_dashboard(df_all, metrics, out_html)
    if run_id:
        register_run(
            run_id,
            out_dir=test_record_dir.parent,
            test_record_dir=test_record_dir,
        )
    print(f'Wrote {out_html}')
    print(f'Wrote {summary_path}')
    maybe_build_index(args)


if __name__ == '__main__':
    main()
