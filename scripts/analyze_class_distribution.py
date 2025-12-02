#!/usr/bin/env python3
"""
Class Distribution Analysis Script

Calculates class distribution from training data to inform weight selection.
Computes inverse frequency weights and recommends focal loss alpha values.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import argparse
from typing import List, Dict

def analyze_pdb_labels(processed_dir: Path, pdb_id: str) -> Dict:
    """Analyze label distribution for a single PDB."""
    pdb_dir = processed_dir / pdb_id
    label_file = pdb_dir / 'node_label_pi.parquet'
    
    if not label_file.exists():
        return None
    
    try:
        labels_df = pd.read_parquet(label_file)
        if 'isInterface' not in labels_df.columns:
            return None
        
        label_counts = labels_df['isInterface'].value_counts().to_dict()
        total = len(labels_df)
        
        return {
            'pdb_id': pdb_id,
            'total': total,
            'label_0': int(label_counts.get(0, 0)),
            'label_1': int(label_counts.get(1, 0)),
            'label_2': int(label_counts.get(2, 0)),
            'label_0_pct': label_counts.get(0, 0) / total * 100,
            'label_1_pct': label_counts.get(1, 0) / total * 100,
            'label_2_pct': label_counts.get(2, 0) / total * 100,
        }
    except Exception as e:
        print(f"Warning: Error analyzing {pdb_id}: {e}")
        return None


def analyze_class_distribution(processed_dir: Path, pdb_list: List[str] = None) -> Dict:
    """Analyze class distribution across all PDBs."""
    if pdb_list is None:
        # Auto-detect PDBs from directory structure
        pdb_list = [d.name for d in processed_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    results = []
    for pdb_id in sorted(pdb_list):
        result = analyze_pdb_labels(processed_dir, pdb_id)
        if result:
            results.append(result)
    
    if len(results) == 0:
        return None
    
    # Aggregate statistics
    total_residues = sum(r['total'] for r in results)
    total_label_0 = sum(r['label_0'] for r in results)
    total_label_1 = sum(r['label_1'] for r in results)
    total_label_2 = sum(r['label_2'] for r in results)
    
    # Calculate inverse frequency weights
    # Formula: weight_i = total_samples / (num_classes * count_i)
    num_classes = 3
    if total_label_0 > 0 and total_label_1 > 0 and total_label_2 > 0:
        weight_0 = total_residues / (num_classes * total_label_0)
        weight_1 = total_residues / (num_classes * total_label_1)
        weight_2 = total_residues / (num_classes * total_label_2)
        
        # Normalize weights (optional - can also use raw inverse frequency)
        # Normalize so smallest weight is 1.0
        min_weight = min(weight_0, weight_1, weight_2)
        normalized_weight_0 = weight_0 / min_weight
        normalized_weight_1 = weight_1 / min_weight
        normalized_weight_2 = weight_2 / min_weight
    else:
        weight_0 = weight_1 = weight_2 = 1.0
        normalized_weight_0 = normalized_weight_1 = normalized_weight_2 = 1.0
    
    # Calculate focal loss alpha values (inverse frequency, normalized to sum to 1)
    if total_residues > 0:
        alpha_0 = total_label_0 / total_residues
        alpha_1 = total_label_1 / total_residues
        alpha_2 = total_label_2 / total_residues
    else:
        alpha_0 = alpha_1 = alpha_2 = 1.0 / 3.0
    
    return {
        'total_pdbs': len(results),
        'total_residues': total_residues,
        'label_counts': {
            'label_0': total_label_0,
            'label_1': total_label_1,
            'label_2': total_label_2
        },
        'label_percentages': {
            'label_0': total_label_0 / total_residues * 100 if total_residues > 0 else 0,
            'label_1': total_label_1 / total_residues * 100 if total_residues > 0 else 0,
            'label_2': total_label_2 / total_residues * 100 if total_residues > 0 else 0
        },
        'inverse_frequency_weights': {
            'label_0': weight_0,
            'label_1': weight_1,
            'label_2': weight_2
        },
        'normalized_weights': {
            'label_0': normalized_weight_0,
            'label_1': normalized_weight_1,
            'label_2': normalized_weight_2
        },
        'focal_loss_alphas': {
            'label_0': alpha_0,
            'label_1': alpha_1,
            'label_2': alpha_2
        },
        'class_imbalance_ratio': {
            'label_0_to_1': total_label_0 / max(total_label_1, 1),
            'label_0_to_2': total_label_0 / max(total_label_2, 1),
            'label_2_to_1': total_label_2 / max(total_label_1, 1)
        },
        'per_pdb_results': results
    }


def print_analysis_report(analysis: Dict):
    """Print class distribution analysis report."""
    print("=" * 80)
    print("CLASS DISTRIBUTION ANALYSIS")
    print("=" * 80)
    
    print(f"\nDataset Statistics:")
    print(f"  Total PDBs: {analysis['total_pdbs']}")
    print(f"  Total residues: {analysis['total_residues']}")
    
    print(f"\nLabel Distribution:")
    counts = analysis['label_counts']
    pcts = analysis['label_percentages']
    print(f"  Label 0 (Non-epitope):     {counts['label_0']:8d} ({pcts['label_0']:5.2f}%)")
    print(f"  Label 1 (CIPS):            {counts['label_1']:8d} ({pcts['label_1']:5.2f}%)")
    print(f"  Label 2 (BepiPred/Ellipro): {counts['label_2']:8d} ({pcts['label_2']:5.2f}%)")
    
    print(f"\nClass Imbalance Ratios:")
    ratios = analysis['class_imbalance_ratio']
    print(f"  Non-epitope : CIPS = {ratios['label_0_to_1']:.1f} : 1")
    print(f"  Non-epitope : BepiPred/Ellipro = {ratios['label_0_to_2']:.1f} : 1")
    print(f"  BepiPred/Ellipro : CIPS = {ratios['label_2_to_1']:.1f} : 1")
    
    print(f"\nRecommended Class Weights (Inverse Frequency):")
    weights = analysis['inverse_frequency_weights']
    print(f"  Label 0: {weights['label_0']:.4f}")
    print(f"  Label 1: {weights['label_1']:.4f}")
    print(f"  Label 2: {weights['label_2']:.4f}")
    print(f"\n  For parameters_example.txt:")
    print(f"    cross_entropy_weight={weights['label_0']:.2f} {weights['label_1']:.2f} {weights['label_2']:.2f}")
    
    print(f"\nNormalized Weights (min=1.0):")
    norm_weights = analysis['normalized_weights']
    print(f"  Label 0: {norm_weights['label_0']:.4f}")
    print(f"  Label 1: {norm_weights['label_1']:.4f}")
    print(f"  Label 2: {norm_weights['label_2']:.4f}")
    print(f"\n  For parameters_example.txt (normalized):")
    print(f"    cross_entropy_weight={norm_weights['label_0']:.2f} {norm_weights['label_1']:.2f} {norm_weights['label_2']:.2f}")
    
    print(f"\nRecommended Focal Loss Alpha Values:")
    alphas = analysis['focal_loss_alphas']
    print(f"  Label 0: {alphas['label_0']:.4f}")
    print(f"  Label 1: {alphas['label_1']:.4f}")
    print(f"  Label 2: {alphas['label_2']:.4f}")
    print(f"\n  For focal loss (should sum to ~1.0):")
    print(f"    focal_alpha={alphas['label_0']:.4f} {alphas['label_1']:.4f} {alphas['label_2']:.4f}")
    
    # Recommendations
    print(f"\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    if ratios['label_0_to_1'] > 100:
        print("⚠️  EXTREME class imbalance detected (Non-epitope:CIPS > 100:1)")
        print("   - Consider using focal loss with high gamma (e.g., gamma=2.0)")
        print("   - Use normalized weights to prevent numerical instability")
    elif ratios['label_0_to_1'] > 50:
        print("⚠️  Severe class imbalance detected (Non-epitope:CIPS > 50:1)")
        print("   - Focal loss recommended (gamma=1.0-2.0)")
        print("   - Use inverse frequency weights")
    else:
        print("✓  Moderate class imbalance - standard class weighting should work")
    
    if pcts['label_1'] < 1.0:
        print("⚠️  CIPS labels are very rare (<1% of residues)")
        print("   - This explains poor CIPS detection performance")
        print("   - Consider adjusting CIPS distance threshold or labeling strategy")
    
    print(f"\nCurrent weights in parameters_example.txt: 10 35 20")
    print(f"Recommended weights: {weights['label_0']:.2f} {weights['label_1']:.2f} {weights['label_2']:.2f}")
    
    if abs(weights['label_1'] - 35.0) / 35.0 > 0.2:
        print("⚠️  Current CIPS weight (35) differs significantly from recommended")
        print(f"   Recommended: {weights['label_1']:.2f}")


def main():
    parser = argparse.ArgumentParser(description='Analyze class distribution from training data')
    parser.add_argument('processed_dir', type=str, help='Path to processed data directory')
    parser.add_argument('--pdb-list', type=str, help='Comma-separated list of PDB IDs (optional, auto-detects if not provided)')
    parser.add_argument('--output', type=str, help='Output file for results (JSON format)')
    
    args = parser.parse_args()
    
    processed_dir = Path(args.processed_dir)
    if not processed_dir.exists():
        print(f"Error: Processed directory not found: {processed_dir}")
        sys.exit(1)
    
    pdb_list = None
    if args.pdb_list:
        pdb_list = [pdb.strip() for pdb in args.pdb_list.split(',')]
    
    analysis = analyze_class_distribution(processed_dir, pdb_list)
    
    if analysis is None:
        print("Error: No valid label data found")
        sys.exit(1)
    
    print_analysis_report(analysis)
    
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()

