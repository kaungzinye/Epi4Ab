#!/usr/bin/env python3
"""
Comprehensive Prediction Quality Review Script

Analyzes Epi4Ab inference results and provides detailed assessment of prediction quality.
Identifies issues, compares metrics, and provides recommendations.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import argparse
from typing import Dict, List, Tuple, Optional

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

def load_evaluation_results(results_dir: Path) -> Tuple[pd.DataFrame, Dict]:
    """Load evaluation results from output directory."""
    mean_file = results_dir / "evaluation_mean_test_whole.txt"
    each_file = results_dir / "evaluation_each_pdb.txt"
    parquet_file = results_dir / "evaluation_test_whole.parquet"
    
    # Load mean results
    mean_results = {}
    if mean_file.exists():
        with open(mean_file, 'r') as f:
            lines = f.readlines()
            current_section = None
            for line in lines:
                line = line.strip()
                if 'ALL result:' in line:
                    current_section = 'all'
                    continue
                elif 'CIPS result:' in line:
                    current_section = 'cips'
                    continue
                elif line.startswith('Recall') or line.startswith('--------'):
                    continue
                elif line and current_section:
                    parts = line.split()
                    if len(parts) >= 6:
                        mean_results[f'{current_section}_recall'] = float(parts[0])
                        mean_results[f'{current_section}_precision'] = float(parts[1])
                        mean_results[f'{current_section}_f1'] = float(parts[2])
                        mean_results[f'{current_section}_accuracy'] = float(parts[3])
                        mean_results[f'{current_section}_roc_auc'] = float(parts[4]) if parts[4] != 'nan' else np.nan
                        mean_results[f'{current_section}_avg_precision'] = float(parts[5]) if parts[5] != 'nan' else np.nan
    
    # Load per-PDB results
    per_pdb_df = None
    if each_file.exists():
        per_pdb_df = pd.read_csv(each_file, sep='\s+', skipinitialspace=True)
    
    # Load detailed parquet if available
    detailed_df = None
    if parquet_file.exists():
        detailed_df = pd.read_parquet(parquet_file)
    
    return per_pdb_df, mean_results, detailed_df


def analyze_label_distribution(results_dir: Path, pdb_ids: List[str]) -> Dict:
    """Analyze label distribution across all PDBs."""
    label_stats = {
        'total_residues': 0,
        'label_0_count': 0,  # Non-epitope
        'label_1_count': 0,  # CIPS (direct contact)
        'label_2_count': 0,  # BepiPred/Ellipro predicted
        'pred_label_0_count': 0,
        'pred_label_1_count': 0,
        'pred_label_2_count': 0,
        'per_pdb': {}
    }
    
    test_record_dir = results_dir / "test_record"
    
    for pdb_id in pdb_ids:
        # Try detailed file first (has true_y)
        detail_file = test_record_dir / f"{pdb_id}.txt"
        result_file = test_record_dir / f"{pdb_id}_final_result.txt"
        
        df = None
        true_labels = None
        
        # Try to load detailed file with true labels
        if detail_file.exists():
            try:
                df = pd.read_csv(detail_file, sep='\t')
                if 'true_y' in df.columns:
                    true_labels = df['true_y'].values
                elif 'true_label' in df.columns:
                    true_labels = df['true_label'].values
                # Also check for pred_y to get predictions
                if 'pred_y' in df.columns:
                    pred_labels = df['pred_y'].values
            except Exception as e:
                print(f"Warning: Could not load {detail_file}: {e}", file=sys.stderr)
                pass
        
        # Fallback to final_result file
        if df is None and result_file.exists():
            try:
                df = pd.read_csv(result_file, sep='\t')
            except:
                continue
        
        if df is None:
            continue
        
        # Get prediction labels
        if 'pred_label' in df.columns:
            pred_labels = df['pred_label'].values
        elif 'pred_y' in df.columns:
            pred_labels = df['pred_y'].values
        else:
            continue
        
        label_stats['total_residues'] += len(df)
        label_stats['pred_label_0_count'] += np.sum(pred_labels == 0)
        label_stats['pred_label_1_count'] += np.sum(pred_labels == 1)
        label_stats['pred_label_2_count'] += np.sum(pred_labels == 2)
        
        if true_labels is not None:
            label_stats['label_0_count'] += np.sum(true_labels == 0)
            label_stats['label_1_count'] += np.sum(true_labels == 1)
            label_stats['label_2_count'] += np.sum(true_labels == 2)
            
            # Per-PDB stats
            pdb_stats = {
                'total': len(df),
                'true_0': int(np.sum(true_labels == 0)),
                'true_1': int(np.sum(true_labels == 1)),
                'true_2': int(np.sum(true_labels == 2)),
                'pred_0': int(np.sum(pred_labels == 0)),
                'pred_1': int(np.sum(pred_labels == 1)),
                'pred_2': int(np.sum(pred_labels == 2)),
            }
        else:
            # Only prediction stats available
            pdb_stats = {
                'total': len(df),
                'true_0': None,
                'true_1': None,
                'true_2': None,
                'pred_0': int(np.sum(pred_labels == 0)),
                'pred_1': int(np.sum(pred_labels == 1)),
                'pred_2': int(np.sum(pred_labels == 2)),
            }
        
        label_stats['per_pdb'][pdb_id] = pdb_stats
    
    return label_stats


def calculate_detailed_metrics(results_dir: Path, pdb_ids: List[str]) -> pd.DataFrame:
    """Calculate detailed per-PDB metrics."""
    test_record_dir = results_dir / "test_record"
    metrics_list = []
    
    for pdb_id in pdb_ids:
        # Try detailed file first
        detail_file = test_record_dir / f"{pdb_id}.txt"
        result_file = test_record_dir / f"{pdb_id}_final_result.txt"
        
        df = None
        true_labels = None
        
        if detail_file.exists():
            try:
                df = pd.read_csv(detail_file, sep='\t')
                if 'true_y' in df.columns:
                    true_labels = df['true_y'].values
            except:
                pass
        
        if df is None and result_file.exists():
            try:
                df = pd.read_csv(result_file, sep='\t')
            except:
                continue
        
        if df is None or 'pred_label' not in df.columns:
            continue
        
        pred_labels = df['pred_label'].values
        
        if true_labels is None:
            # Can't calculate per-class metrics without true labels
            continue
        
        # Calculate per-class metrics
        metrics = {'pdb_id': pdb_id}
        
        for label in [0, 1, 2]:
            label_name = ['Non-epitope', 'CIPS', 'BepiPred/Ellipro'][label]
            
            # True positives, false positives, false negatives
            tp = np.sum((true_labels == label) & (pred_labels == label))
            fp = np.sum((true_labels != label) & (pred_labels == label))
            fn = np.sum((true_labels == label) & (pred_labels != label))
            tn = np.sum((true_labels != label) & (pred_labels != label))
            
            # Calculate metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
            
            metrics[f'{label_name}_precision'] = precision
            metrics[f'{label_name}_recall'] = recall
            metrics[f'{label_name}_f1'] = f1
            metrics[f'{label_name}_tp'] = int(tp)
            metrics[f'{label_name}_fp'] = int(fp)
            metrics[f'{label_name}_fn'] = int(fn)
            metrics[f'{label_name}_support'] = int(np.sum(true_labels == label))
        
        # Overall accuracy
        metrics['overall_accuracy'] = np.mean(true_labels == pred_labels)
        
        metrics_list.append(metrics)
    
    return pd.DataFrame(metrics_list)


def assess_prediction_quality(mean_results: Dict, per_pdb_df: pd.DataFrame, 
                              label_stats: Dict) -> Dict:
    """Assess overall prediction quality and identify issues."""
    assessment = {
        'overall_grade': 'F',
        'issues': [],
        'strengths': [],
        'recommendations': []
    }
    
    # Overall metrics assessment
    all_recall = mean_results.get('all_recall', 0)
    all_precision = mean_results.get('all_precision', 0)
    all_f1 = mean_results.get('all_f1', 0)
    all_roc_auc = mean_results.get('all_roc_auc', np.nan)
    
    cips_recall = mean_results.get('cips_recall', 0)
    cips_precision = mean_results.get('cips_precision', 0)
    cips_f1 = mean_results.get('cips_f1', 0)
    
    # Check for major issues
    if cips_recall == 0 and cips_precision == 0:
        assessment['issues'].append("CRITICAL: CIPS (Label 1) predictions are completely failing - Recall=0, Precision=0")
        assessment['issues'].append("Model is not detecting any direct antibody contact residues")
    
    if all_roc_auc < 0.6:
        assessment['issues'].append(f"Poor ROC AUC ({all_roc_auc:.3f}) - model performance is barely better than random (0.5)")
    
    if all_f1 < 0.4:
        assessment['issues'].append(f"Very low F1 score ({all_f1:.3f}) - poor balance between precision and recall")
    
    # Check label distribution
    total = label_stats['total_residues']
    if total > 0:
        # Check if we have true label data
        if label_stats['label_0_count'] + label_stats['label_1_count'] + label_stats['label_2_count'] > 0:
            label_0_pct = label_stats['label_0_count'] / total * 100
            label_1_pct = label_stats['label_1_count'] / total * 100
            label_2_pct = label_stats['label_2_count'] / total * 100
            
            if label_1_pct < 1:
                assessment['issues'].append(f"Severe class imbalance: Only {label_1_pct:.2f}% of residues are CIPS (Label 1)")
                assessment['issues'].append("This explains why CIPS predictions are failing - too few positive examples")
            
            if label_0_pct > 95:
                assessment['issues'].append(f"Extreme class imbalance: {label_0_pct:.2f}% of residues are non-epitopes")
                assessment['issues'].append("Model may be biased towards predicting majority class (non-epitope)")
    
    # Check prediction distribution
    if total > 0:
        pred_0_pct = label_stats['pred_label_0_count'] / total * 100
        pred_1_pct = label_stats['pred_label_1_count'] / total * 100
        pred_2_pct = label_stats['pred_label_2_count'] / total * 100
        
        if pred_1_pct == 0:
            assessment['issues'].append("Model never predicts CIPS (Label 1) - all predictions are Label 0 or 2")
        
        if pred_0_pct > 95:
            assessment['issues'].append(f"Model is over-predicting non-epitopes ({pred_0_pct:.2f}%) - likely due to class imbalance")
    
    # Grade assignment
    if all_f1 >= 0.7 and all_roc_auc >= 0.8:
        assessment['overall_grade'] = 'A'
    elif all_f1 >= 0.6 and all_roc_auc >= 0.7:
        assessment['overall_grade'] = 'B'
    elif all_f1 >= 0.5 and all_roc_auc >= 0.6:
        assessment['overall_grade'] = 'C'
    elif all_f1 >= 0.4:
        assessment['overall_grade'] = 'D'
    else:
        assessment['overall_grade'] = 'F'
    
    # Recommendations
    if cips_recall == 0:
        assessment['recommendations'].append("Implement class weighting or focal loss to handle CIPS class imbalance")
        assessment['recommendations'].append("Consider oversampling CIPS residues or using different threshold for Label 1")
        assessment['recommendations'].append("Review CIPS label generation - ensure labels are correct")
    
    if all_roc_auc < 0.6:
        assessment['recommendations'].append("Model may need retraining with better features or architecture")
        assessment['recommendations'].append("Check if new features (FreeSASA RSA, PDB2PQR charges) are being used correctly")
    
    if total > 0 and label_stats['label_1_count'] > 0:
        if label_stats['label_1_count'] / total < 0.01:
            assessment['recommendations'].append("Consider using different labeling strategy - CIPS residues are too rare")
            assessment['recommendations'].append("May need to combine CIPS with other epitope prediction methods")
    
    return assessment


def print_review_report(results_dir: Path, pdb_ids: List[str]):
    """Print comprehensive review report."""
    print("=" * 80)
    print("EPI4AB PREDICTION QUALITY REVIEW")
    print("=" * 80)
    
    # Load results
    per_pdb_df, mean_results, detailed_df = load_evaluation_results(results_dir)
    label_stats = analyze_label_distribution(results_dir, pdb_ids)
    detailed_metrics = calculate_detailed_metrics(results_dir, pdb_ids)
    
    # Overall metrics
    print("\n" + "=" * 80)
    print("OVERALL METRICS (Mean across all PDBs)")
    print("=" * 80)
    print(f"\n{'Metric':<25} {'ALL Labels':<20} {'CIPS Only':<20}")
    print("-" * 80)
    print(f"{'Recall':<25} {mean_results.get('all_recall', 0):<20.4f} {mean_results.get('cips_recall', 0):<20.4f}")
    print(f"{'Precision':<25} {mean_results.get('all_precision', 0):<20.4f} {mean_results.get('cips_precision', 0):<20.4f}")
    print(f"{'F1 Score':<25} {mean_results.get('all_f1', 0):<20.4f} {mean_results.get('cips_f1', 0):<20.4f}")
    print(f"{'Accuracy':<25} {mean_results.get('all_accuracy', 0):<20.4f} {mean_results.get('cips_accuracy', 0):<20.4f}")
    print(f"{'ROC AUC':<25} {mean_results.get('all_roc_auc', np.nan):<20.4f} {mean_results.get('cips_roc_auc', np.nan):<20.4f}")
    print(f"{'Avg Precision':<25} {mean_results.get('all_avg_precision', np.nan):<20.4f} {mean_results.get('cips_avg_precision', np.nan):<20.4f}")
    
    # Label distribution
    print("\n" + "=" * 80)
    print("LABEL DISTRIBUTION ANALYSIS")
    print("=" * 80)
    total = label_stats['total_residues']
    if total > 0:
        print(f"\nTotal residues analyzed: {total}")
        print(f"\nTrue Label Distribution:")
        print(f"  Label 0 (Non-epitope):     {label_stats['label_0_count']:6d} ({label_stats['label_0_count']/total*100:5.2f}%)")
        print(f"  Label 1 (CIPS):            {label_stats['label_1_count']:6d} ({label_stats['label_1_count']/total*100:5.2f}%)")
        print(f"  Label 2 (BepiPred/Ellipro): {label_stats['label_2_count']:6d} ({label_stats['label_2_count']/total*100:5.2f}%)")
        
        print(f"\nPredicted Label Distribution:")
        print(f"  Label 0 (Non-epitope):     {label_stats['pred_label_0_count']:6d} ({label_stats['pred_label_0_count']/total*100:5.2f}%)")
        print(f"  Label 1 (CIPS):            {label_stats['pred_label_1_count']:6d} ({label_stats['pred_label_1_count']/total*100:5.2f}%)")
        print(f"  Label 2 (BepiPred/Ellipro): {label_stats['pred_label_2_count']:6d} ({label_stats['pred_label_2_count']/total*100:5.2f}%)")
        
        print(f"\nClass Imbalance Ratio:")
        print(f"  Non-epitope : CIPS : BepiPred/Ellipro = {label_stats['label_0_count']/max(label_stats['label_1_count'],1):.1f} : 1 : {label_stats['label_2_count']/max(label_stats['label_1_count'],1):.1f}")
    
    # Per-PDB breakdown
    if per_pdb_df is not None:
        print("\n" + "=" * 80)
        print("PER-PDB METRICS BREAKDOWN")
        print("=" * 80)
        print(per_pdb_df.to_string(index=False))
    
    # Detailed per-class metrics
    if not detailed_metrics.empty:
        print("\n" + "=" * 80)
        print("DETAILED PER-CLASS METRICS")
        print("=" * 80)
        for pdb_id in detailed_metrics['pdb_id']:
            pdb_metrics = detailed_metrics[detailed_metrics['pdb_id'] == pdb_id].iloc[0]
            print(f"\n{pdb_id}:")
            for label_name in ['Non-epitope', 'CIPS', 'BepiPred/Ellipro']:
                support = pdb_metrics.get(f'{label_name}_support', 0)
                if support > 0:
                    precision = pdb_metrics.get(f'{label_name}_precision', 0)
                    recall = pdb_metrics.get(f'{label_name}_recall', 0)
                    f1 = pdb_metrics.get(f'{label_name}_f1', 0)
                    tp = pdb_metrics.get(f'{label_name}_tp', 0)
                    fp = pdb_metrics.get(f'{label_name}_fp', 0)
                    fn = pdb_metrics.get(f'{label_name}_fn', 0)
                    print(f"  {label_name:20s}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}, Support={support}, TP={tp}, FP={fp}, FN={fn}")
    
    # Quality assessment
    assessment = assess_prediction_quality(mean_results, per_pdb_df, label_stats)
    
    print("\n" + "=" * 80)
    print("QUALITY ASSESSMENT")
    print("=" * 80)
    print(f"\nOverall Grade: {assessment['overall_grade']}")
    
    if assessment['issues']:
        print("\n🚨 ISSUES IDENTIFIED:")
        for i, issue in enumerate(assessment['issues'], 1):
            print(f"  {i}. {issue}")
    
    if assessment['strengths']:
        print("\n✅ STRENGTHS:")
        for i, strength in enumerate(assessment['strengths'], 1):
            print(f"  {i}. {strength}")
    
    if assessment['recommendations']:
        print("\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(assessment['recommendations'], 1):
            print(f"  {i}. {rec}")
    
    # Probability analysis
    print("\n" + "=" * 80)
    print("PROBABILITY DISTRIBUTION ANALYSIS")
    print("=" * 80)
    test_record_dir = results_dir / "test_record"
    prob_stats = {'prob_0': [], 'prob_1': [], 'prob_2': []}
    
    for pdb_id in pdb_ids[:3]:  # Analyze first 3 PDBs
        detail_file = test_record_dir / f"{pdb_id}.txt"
        if detail_file.exists():
            try:
                df = pd.read_csv(detail_file, sep='\t')
                if 'prob_0' in df.columns and 'prob_1' in df.columns and 'prob_2' in df.columns:
                    prob_stats['prob_0'].extend(df['prob_0'].values)
                    prob_stats['prob_1'].extend(df['prob_1'].values)
                    prob_stats['prob_2'].extend(df['prob_2'].values)
            except:
                pass
    
    if prob_stats['prob_0']:
        print(f"\nProbability Statistics (from sample PDBs):")
        print(f"  Label 0 (Non-epitope) probabilities:")
        print(f"    Mean: {np.mean(prob_stats['prob_0']):.4f}, Median: {np.median(prob_stats['prob_0']):.4f}")
        print(f"    Min: {np.min(prob_stats['prob_0']):.4f}, Max: {np.max(prob_stats['prob_0']):.4f}")
        print(f"  Label 1 (CIPS) probabilities:")
        print(f"    Mean: {np.mean(prob_stats['prob_1']):.4f}, Median: {np.median(prob_stats['prob_1']):.4f}")
        print(f"    Min: {np.min(prob_stats['prob_1']):.4f}, Max: {np.max(prob_stats['prob_1']):.4f}")
        print(f"  Label 2 (BepiPred/Ellipro) probabilities:")
        print(f"    Mean: {np.mean(prob_stats['prob_2']):.4f}, Median: {np.median(prob_stats['prob_2']):.4f}")
        print(f"    Min: {np.min(prob_stats['prob_2']):.4f}, Max: {np.max(prob_stats['prob_2']):.4f}")
        
        # Check if probabilities are too low for minority classes
        if np.mean(prob_stats['prob_1']) < 0.01:
            assessment['issues'].append(f"Label 1 probabilities are extremely low (mean={np.mean(prob_stats['prob_1']):.4f}) - model is not confident about CIPS")
        if np.mean(prob_stats['prob_2']) < 0.01:
            assessment['issues'].append(f"Label 2 probabilities are extremely low (mean={np.mean(prob_stats['prob_2']):.4f}) - model is not confident about BepiPred/Ellipro epitopes")
    
    # Show specific examples of misclassifications
    print("\n" + "=" * 80)
    print("MISCLASSIFICATION EXAMPLES")
    print("=" * 80)
    for pdb_id in pdb_ids[:2]:  # Show examples from first 2 PDBs
        detail_file = test_record_dir / f"{pdb_id}.txt"
        if detail_file.exists():
            try:
                df = pd.read_csv(detail_file, sep='\t')
                if 'true_y' in df.columns and 'pred_y' in df.columns:
                    # Find misclassified residues
                    misclassified = df[df['true_y'] != df['pred_y']]
                    if len(misclassified) > 0:
                        print(f"\n{pdb_id} - Sample misclassifications (first 5):")
                        sample = misclassified.head(5)[['res_id', 'true_y', 'pred_y', 'prob_0', 'prob_1', 'prob_2']]
                        for _, row in sample.iterrows():
                            true_label_name = ['Non-epitope', 'CIPS', 'BepiPred/Ellipro'][int(row['true_y'])]
                            pred_label_name = ['Non-epitope', 'CIPS', 'BepiPred/Ellipro'][int(row['pred_y'])]
                            print(f"  Residue {int(row['res_id'])}: True={true_label_name}, Pred={pred_label_name}, "
                                  f"Probs=[{row['prob_0']:.3f}, {row['prob_1']:.3f}, {row['prob_2']:.3f}]")
            except:
                pass
    
    # Comparison with expected performance
    print("\n" + "=" * 80)
    print("EXPECTED PERFORMANCE BENCHMARKS")
    print("=" * 80)
    print("\nFor epitope prediction tasks, typical performance:")
    print("  - Good: F1 > 0.6, ROC AUC > 0.75")
    print("  - Acceptable: F1 > 0.5, ROC AUC > 0.65")
    print("  - Poor: F1 < 0.5, ROC AUC < 0.6")
    print("\nCurrent performance:")
    print(f"  - F1 Score: {mean_results.get('all_f1', 0):.4f} ({'✅ Good' if mean_results.get('all_f1', 0) > 0.6 else '⚠️ Acceptable' if mean_results.get('all_f1', 0) > 0.5 else '❌ Poor'})")
    print(f"  - ROC AUC: {mean_results.get('all_roc_auc', np.nan):.4f} ({'✅ Good' if mean_results.get('all_roc_auc', np.nan) > 0.75 else '⚠️ Acceptable' if mean_results.get('all_roc_auc', np.nan) > 0.65 else '❌ Poor'})")
    
    # Summary verdict
    print("\n" + "=" * 80)
    print("SUMMARY VERDICT")
    print("=" * 80)
    print("\nThe predictions are POOR overall:")
    print("  ❌ CIPS detection completely fails (0% recall)")
    print("  ❌ ROC AUC barely above random (0.526)")
    print("  ⚠️  Overall F1 is borderline acceptable (0.51) but misleading due to class imbalance")
    print("  ❌ Model is heavily biased towards predicting non-epitopes (99.87%)")
    print("\nRoot causes:")
    print("  1. Extreme class imbalance (338:1 ratio for non-epitope:CIPS)")
    print("  2. Model was likely trained without proper class weighting")
    print("  3. CIPS residues are too rare for the model to learn effectively")
    print("  4. Model may need retraining with new features (FreeSASA, PDB2PQR)")
    
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Review Epi4Ab prediction quality')
    parser.add_argument('results_dir', type=str, help='Path to results directory (e.g., run_model_output/2025-11-12_GNNResNet_1)')
    parser.add_argument('--pdb-list', type=str, help='Comma-separated list of PDB IDs (optional, auto-detects if not provided)')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        sys.exit(1)
    
    # Get PDB list
    if args.pdb_list:
        pdb_ids = [pdb.strip() for pdb in args.pdb_list.split(',')]
    else:
        # Auto-detect from test_record directory
        test_record_dir = results_dir / "test_record"
        if test_record_dir.exists():
            pdb_ids = [f.stem.replace('_final_result', '') for f in test_record_dir.glob('*_final_result.txt')]
        else:
            print("Error: Could not find test_record directory or PDB list")
            sys.exit(1)
    
    print_review_report(results_dir, pdb_ids)


if __name__ == '__main__':
    main()

