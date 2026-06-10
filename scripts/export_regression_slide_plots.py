#!/usr/bin/env python3
"""Export static PNGs from Epi4Ab regression test_record outputs.

Creates slide-friendly plots from `*_final_result.txt` files and
`regression_summary.csv`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from epi4ab_plot_paths import resolve_run_id, run_png_dir

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_test_record(test_record_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_path = test_record_dir / 'regression_summary.csv'
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    summary = pd.read_csv(summary_path)

    dfs = []
    for f in sorted(test_record_dir.glob('*_final_result.txt')):
        pdb_id = f.name.replace('_final_result.txt', '')
        df = pd.read_csv(f, sep='\t')
        df['pdb_id'] = pdb_id
        dfs.append(df)
    if not dfs:
        raise FileNotFoundError(f'No *_final_result.txt in {test_record_dir}')
    all_df = pd.concat(dfs, ignore_index=True)
    return all_df, summary


def compute_micro_metrics(df: pd.DataFrame) -> dict[str, float]:
    y = df['true_score'].to_numpy(float)
    yhat = df['pred_score'].to_numpy(float)
    pear = float(np.corrcoef(yhat, y)[0, 1])
    ra = pd.Series(yhat).rank(method='average').to_numpy(float)
    rb = pd.Series(y).rank(method='average').to_numpy(float)
    spear = float(np.corrcoef(ra, rb)[0, 1])
    mse = float(np.mean((yhat - y) ** 2))
    mae = float(np.mean(np.abs(yhat - y)))
    return {
        'micro_pearson': pear,
        'micro_spearman': spear,
        'micro_mse': mse,
        'micro_mae': mae,
        'n_res': int(len(df)),
    }


def plot_scatter(df: pd.DataFrame, title: str, out_path: Path):
    y = df['true_score'].to_numpy(float)
    yhat = df['pred_score'].to_numpy(float)
    lim_lo = float(min(y.min(), yhat.min()))
    lim_hi = float(max(y.max(), yhat.max()))

    plt.figure(figsize=(7.5, 6.2))
    plt.scatter(y, yhat, s=10, alpha=0.35, edgecolors='none')
    plt.plot([lim_lo, lim_hi], [lim_lo, lim_hi], '--', linewidth=1)
    plt.xlabel('True ProteinMPNN NLL')
    plt.ylabel('Predicted NLL')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches='tight')
    plt.close()


def plot_metrics(summary: pd.DataFrame, title: str, out_path: Path):
    s = summary.sort_values('spearman', ascending=False).copy()
    labels = s['pdb_id'].tolist()
    x = np.arange(len(labels))

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].bar(x, s['spearman'], color='#2c7d32')
    axes[0].set_ylabel('Spearman')
    axes[0].set_title(f'{title} - Held-out per-PDB rank correlation')
    axes[0].axhline(s['spearman'].mean(), color='black', linestyle='--', linewidth=1, label='mean')
    axes[0].legend(loc='lower right')

    axes[1].bar(x, s['mse'], color='#c62828')
    axes[1].set_ylabel('MSE')
    axes[1].set_title(f'{title} - Held-out per-PDB error')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=75, ha='right', fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def plot_summary_card(summary: pd.DataFrame, micro: dict[str, float], title: str, out_path: Path):
    macro_s = float(summary['spearman'].mean())
    macro_p = float(summary['pearson'].mean())
    macro_mse = float(summary['mse'].mean())
    macro_mae = float(summary['mae'].mean())
    lines = [
        title,
        '',
        f"Held-out test PDBs: {len(summary)}",
        f"Held-out residues: {micro['n_res']}",
        '',
        f"Macro Spearman: {macro_s:.3f}",
        f"Micro Spearman: {micro['micro_spearman']:.3f}",
        f"Macro Pearson:  {macro_p:.3f}",
        f"Micro Pearson:  {micro['micro_pearson']:.3f}",
        '',
        f"Macro MSE: {macro_mse:.3f}",
        f"Micro MSE: {micro['micro_mse']:.3f}",
        f"Macro MAE: {macro_mae:.3f}",
        f"Micro MAE: {micro['micro_mae']:.3f}",
    ]

    fig = plt.figure(figsize=(7.5, 5.2))
    ax = fig.add_subplot(111)
    ax.axis('off')
    ax.text(0.03, 0.97, '\n'.join(lines), va='top', ha='left', fontsize=14, family='monospace')
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test_record_dir', required=True)
    ap.add_argument('--out_dir', default=None, help='Default: plots/runs/<run_id>/png/')
    ap.add_argument('--run_id', default=None)
    ap.add_argument('--title', required=True)
    args = ap.parse_args()

    test_record_dir = Path(args.test_record_dir)
    run_id = resolve_run_id(args.run_id) or test_record_dir.parent.name
    out_dir = Path(args.out_dir) if args.out_dir else run_png_dir(run_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    df, summary = load_test_record(test_record_dir)
    micro = compute_micro_metrics(df)

    plot_scatter(df, args.title, out_dir / 'scatter_true_vs_pred.png')
    plot_metrics(summary, args.title, out_dir / 'per_pdb_metrics.png')
    plot_summary_card(summary, micro, args.title, out_dir / 'summary_card.png')

    print(out_dir / 'scatter_true_vs_pred.png')
    print(out_dir / 'per_pdb_metrics.png')
    print(out_dir / 'summary_card.png')


if __name__ == '__main__':
    main()
