#!/usr/bin/env python3
"""
Extract per-class metrics from inference results.

Usage:
    python scripts/extract_per_class_metrics.py <output_dir> [--baseline_dir <baseline_dir>]
"""

import argparse
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import json

def parse_evaluation_file(eval_file: Path) -> pd.DataFrame:
    """Parse evaluation_each_pdb.txt file."""
    if not eval_file.exists():
        print(f"Warning: Evaluation file not found: {eval_file}")
        return None
    
    try:
        # Read the file
        with open(eval_file, 'r') as f:
            lines = f.readlines()
        
        # Find header and data lines (skip separator lines)
        header_line = None
        data_lines = []
        
        for line in lines:
            line_stripped = line.strip()
            # Skip empty lines and separator lines (mostly dashes)
            if not line_stripped or line_stripped.replace('-', '').replace(' ', '').replace('|', '') == '':
                continue
            
            if header_line is None:
                header_line = line_stripped
            else:
                # Check if this is a separator line
                if not (len(line_stripped.replace('-', '').replace(' ', '').replace('|', '')) < 5):
                    data_lines.append(line_stripped)
        
        if not header_line or not data_lines:
            print(f"Warning: Could not parse {eval_file}")
            return None
        
        # Parse header
        headers = [h.strip() for h in header_line.split() if h.strip()]
        
        # Parse data lines
        rows = []
        for line in data_lines:
            values = [v.strip() for v in line.split() if v.strip()]
            if len(values) >= len(headers):
                rows.append(values[:len(headers)])
        
        if not rows:
            return None
        
        df = pd.DataFrame(rows, columns=headers)
        
        # Convert numeric columns
        numeric_cols = ['Recall', 'Precision', 'f1', 'Accuracy', 'ROC AUC', 'Average Precision', 'Label']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    except Exception as e:
        print(f"Error parsing {eval_file}: {e}")
        return None


def extract_per_class_metrics(df: pd.DataFrame) -> Dict:
    """Extract per-class metrics from evaluation dataframe."""
    if df is None or df.empty:
        return None
    
    metrics = {
        'overall': {},
        'per_class': {},
        'per_pdb': {}
    }
    
    # Extract per-class metrics
    for label in [0, 1, 2]:
        label_df = df[df['Label'] == label]
        if len(label_df) > 0:
            metrics['per_class'][f'Label_{label}'] = {
                'count': len(label_df),
                'recall_mean': label_df['Recall'].mean(),
                'recall_std': label_df['Recall'].std(),
                'precision_mean': label_df['Precision'].mean(),
                'precision_std': label_df['Precision'].std(),
                'f1_mean': label_df['f1'].mean(),
                'f1_std': label_df['f1'].std(),
                'accuracy_mean': label_df['Accuracy'].mean(),
                'accuracy_std': label_df['Accuracy'].std(),
                'roc_auc_mean': label_df['ROC AUC'].mean() if 'ROC AUC' in label_df.columns else None,
                'roc_auc_std': label_df['ROC AUC'].std() if 'ROC AUC' in label_df.columns else None,
            }
    
    # Extract per-PDB metrics
    for pdb_id in df['pdbId'].unique():
        pdb_df = df[df['pdbId'] == pdb_id]
        metrics['per_pdb'][pdb_id] = {}
        
        for label in [0, 1, 2]:
            label_pdb_df = pdb_df[pdb_df['Label'] == label]
            if len(label_pdb_df) > 0:
                row = label_pdb_df.iloc[0]
                metrics['per_pdb'][pdb_id][f'Label_{label}'] = {
                    'recall': row['Recall'],
                    'precision': row['Precision'],
                    'f1': row['f1'],
                    'accuracy': row['Accuracy'],
                    'roc_auc': row.get('ROC AUC', None),
                }
    
    # Overall statistics
    metrics['overall'] = {
        'total_pdbs': len(df['pdbId'].unique()),
        'total_evaluations': len(df),
        'labels': sorted(df['Label'].unique().tolist()),
    }
    
    return metrics


def print_metrics_summary(metrics: Dict, title: str = "Metrics Summary"):
    """Print a formatted summary of metrics."""
    print(f"\n{'='*80}")
    print(f"{title:^80}")
    print(f"{'='*80}\n")
    
    if not metrics:
        print("No metrics available.")
        return
    
    # Overall statistics
    print(f"Overall Statistics:")
    print(f"  Total PDBs: {metrics['overall']['total_pdbs']}")
    print(f"  Labels evaluated: {metrics['overall']['labels']}")
    print()
    
    # Per-class metrics
    print(f"Per-Class Metrics:")
    print(f"{'-'*80}")
    
    label_names = {0: 'Non-epitope', 1: 'CIPS', 2: 'BepiPred/Ellipro'}
    
    for label_key in ['Label_0', 'Label_1', 'Label_2']:
        if label_key in metrics['per_class']:
            label_num = int(label_key.split('_')[1])
            label_name = label_names.get(label_num, f'Label {label_num}')
            m = metrics['per_class'][label_key]
            
            print(f"\n{label_name} (Label {label_num}):")
            print(f"  Recall:    {m['recall_mean']:.4f} ± {m['recall_std']:.4f}")
            print(f"  Precision: {m['precision_mean']:.4f} ± {m['precision_std']:.4f}")
            print(f"  F1 Score:  {m['f1_mean']:.4f} ± {m['f1_std']:.4f}")
            print(f"  Accuracy:  {m['accuracy_mean']:.4f} ± {m['accuracy_std']:.4f}")
            if m['roc_auc_mean'] is not None:
                print(f"  ROC AUC:   {m['roc_auc_mean']:.4f} ± {m['roc_auc_std']:.4f}")
    
    print(f"\n{'='*80}\n")


def compare_metrics(focal_metrics: Dict, baseline_metrics: Dict):
    """Compare focal loss metrics vs baseline metrics."""
    print(f"\n{'='*80}")
    print(f"{'FOCAL LOSS vs BASELINE COMPARISON':^80}")
    print(f"{'='*80}\n")
    
    if not focal_metrics or not baseline_metrics:
        print("Cannot compare: missing metrics")
        return
    
    label_names = {0: 'Non-epitope', 1: 'CIPS', 2: 'BepiPred/Ellipro'}
    
    print(f"{'Metric':<30} {'Baseline':<15} {'Focal':<15} {'Change':<15}")
    print(f"{'-'*80}")
    
    for label_key in ['Label_0', 'Label_1', 'Label_2']:
        if label_key in focal_metrics['per_class'] and label_key in baseline_metrics['per_class']:
            label_num = int(label_key.split('_')[1])
            label_name = label_names.get(label_num, f'Label {label_num}')
            
            focal = focal_metrics['per_class'][label_key]
            baseline = baseline_metrics['per_class'][label_key]
            
            print(f"\n{label_name} (Label {label_num}):")
            
            # Recall
            recall_change = focal['recall_mean'] - baseline['recall_mean']
            recall_pct = (recall_change / baseline['recall_mean'] * 100) if baseline['recall_mean'] > 0 else 0
            print(f"  {'Recall':<28} {baseline['recall_mean']:>6.4f}         {focal['recall_mean']:>6.4f}         {recall_change:>+6.4f} ({recall_pct:>+6.1f}%)")
            
            # Precision
            prec_change = focal['precision_mean'] - baseline['precision_mean']
            prec_pct = (prec_change / baseline['precision_mean'] * 100) if baseline['precision_mean'] > 0 else 0
            print(f"  {'Precision':<28} {baseline['precision_mean']:>6.4f}         {focal['precision_mean']:>6.4f}         {prec_change:>+6.4f} ({prec_pct:>+6.1f}%)")
            
            # F1
            f1_change = focal['f1_mean'] - baseline['f1_mean']
            f1_pct = (f1_change / baseline['f1_mean'] * 100) if baseline['f1_mean'] > 0 else 0
            print(f"  {'F1 Score':<28} {baseline['f1_mean']:>6.4f}         {focal['f1_mean']:>6.4f}         {f1_change:>+6.4f} ({f1_pct:>+6.1f}%)")
    
    print(f"\n{'='*80}\n")


def save_metrics(metrics: Dict, output_file: Path):
    """Save metrics to JSON file."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Extract per-class metrics from inference results')
    parser.add_argument('output_dir', help='Directory containing inference output')
    parser.add_argument('--baseline_dir', help='Directory containing baseline inference output (for comparison)', default=None)
    parser.add_argument('--save_json', help='Save metrics to JSON file', action='store_true')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    # Find evaluation file
    eval_files = list(output_dir.glob('**/evaluation_each_pdb.txt'))
    if not eval_files:
        print(f"Error: No evaluation_each_pdb.txt found in {output_dir}")
        sys.exit(1)
    
    eval_file = eval_files[0]
    print(f"Reading evaluation file: {eval_file}")
    
    # Parse evaluation file
    df = parse_evaluation_file(eval_file)
    if df is None:
        print("Error: Could not parse evaluation file")
        sys.exit(1)
    
    # Extract metrics
    focal_metrics = extract_per_class_metrics(df)
    print_metrics_summary(focal_metrics, "Focal Loss Model Metrics")
    
    # Save metrics if requested
    if args.save_json:
        save_metrics(focal_metrics, output_dir / 'per_class_metrics.json')
    
    # Compare with baseline if provided
    if args.baseline_dir:
        baseline_dir = Path(args.baseline_dir)
        baseline_eval_files = list(baseline_dir.glob('**/evaluation_each_pdb.txt'))
        
        if baseline_eval_files:
            baseline_eval_file = baseline_eval_files[0]
            print(f"\nReading baseline evaluation file: {baseline_eval_file}")
            
            baseline_df = parse_evaluation_file(baseline_eval_file)
            if baseline_df is not None:
                baseline_metrics = extract_per_class_metrics(baseline_df)
                print_metrics_summary(baseline_metrics, "Baseline Model Metrics")
                compare_metrics(focal_metrics, baseline_metrics)
                
                if args.save_json:
                    save_metrics(baseline_metrics, baseline_dir / 'per_class_metrics.json')
        else:
            print(f"Warning: No evaluation file found in baseline directory: {baseline_dir}")


if __name__ == '__main__':
    main()

