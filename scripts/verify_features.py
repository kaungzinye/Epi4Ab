#!/usr/bin/env python3
"""
Feature Verification Script

Checks that processed data includes FreeSASA/PDB2PQR features.
Verifies RSA (normalized sasa) and charge columns exist in node_feature.parquet files.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import argparse
from typing import List, Dict, Tuple

def verify_pdb_features(processed_dir: Path, pdb_id: str) -> Dict:
    """Verify features for a single PDB."""
    pdb_dir = processed_dir / pdb_id
    result = {
        'pdb_id': pdb_id,
        'has_node_feature': False,
        'has_sasa': False,
        'has_charge': False,
        'sasa_stats': None,
        'charge_stats': None,
        'has_rsa_cache': False,
        'has_pdb2pqr_cache': False,
        'label_stats': None,
        'issues': []
    }
    
    # Check node_feature.parquet
    node_feature_file = pdb_dir / 'node_feature.parquet'
    if not node_feature_file.exists():
        result['issues'].append(f"Missing node_feature.parquet")
        return result
    
    result['has_node_feature'] = True
    
    try:
        df = pd.read_parquet(node_feature_file)
        
        # Check for sasa column
        if 'sasa' in df.columns:
            result['has_sasa'] = True
            sasa_values = df['sasa'].values
            non_zero = np.sum(sasa_values > 0.0)
            result['sasa_stats'] = {
                'total': len(sasa_values),
                'non_zero': int(non_zero),
                'min': float(np.min(sasa_values)),
                'max': float(np.max(sasa_values)),
                'mean': float(np.mean(sasa_values)),
                'median': float(np.median(sasa_values))
            }
            if non_zero == 0:
                result['issues'].append("All RSA values are zero (FreeSASA may have failed)")
        else:
            result['issues'].append("Missing 'sasa' column in node_feature.parquet")
        
        # Check for charge column
        if 'charge' in df.columns:
            result['has_charge'] = True
            charge_values = df['charge'].values
            non_zero = np.sum(np.abs(charge_values) > 1e-6)
            result['charge_stats'] = {
                'total': len(charge_values),
                'non_zero': int(non_zero),
                'min': float(np.min(charge_values)),
                'max': float(np.max(charge_values)),
                'mean': float(np.mean(charge_values)),
                'median': float(np.median(charge_values))
            }
            if non_zero == 0:
                result['issues'].append("All charge values are zero (PDB2PQR may have failed)")
        else:
            result['issues'].append("Missing 'charge' column in node_feature.parquet")
        
    except Exception as e:
        result['issues'].append(f"Error reading node_feature.parquet: {e}")
        return result
    
    # Check cache directories (new FreeSASA cache and legacy Naccess cache)
    rsa_cache = pdb_dir / 'rsa_cache'
    legacy_cache = pdb_dir / 'naccess_cache'
    cache_files = []
    if rsa_cache.exists() and rsa_cache.is_dir():
        cache_files.extend(rsa_cache.glob(f"{pdb_id}_rsa_freesasa.json"))
    if legacy_cache.exists() and legacy_cache.is_dir():
        cache_files.extend(legacy_cache.glob(f"{pdb_id}_naccess_rsa.json"))
    result['has_rsa_cache'] = len(cache_files) > 0
    
    pdb2pqr_cache = pdb_dir / 'pdb2pqr_cache'
    if pdb2pqr_cache.exists() and pdb2pqr_cache.is_dir():
        cache_files = list(pdb2pqr_cache.glob(f"{pdb_id}_pdb2pqr_charges.json"))
        result['has_pdb2pqr_cache'] = len(cache_files) > 0
    
    # Check labels
    label_file = pdb_dir / 'node_label_pi.parquet'
    if label_file.exists():
        try:
            labels_df = pd.read_parquet(label_file)
            if 'isInterface' in labels_df.columns:
                label_counts = labels_df['isInterface'].value_counts().to_dict()
                result['label_stats'] = {
                    'label_0': int(label_counts.get(0, 0)),
                    'label_1': int(label_counts.get(1, 0)),
                    'label_2': int(label_counts.get(2, 0)),
                    'total': len(labels_df)
                }
                total = result['label_stats']['total']
                if total > 0:
                    label_1_pct = result['label_stats']['label_1'] / total * 100
                    if label_1_pct == 0:
                        result['issues'].append(f"No CIPS labels (Label 1) found - may indicate antibody detection issue")
        except Exception as e:
            result['issues'].append(f"Error reading labels: {e}")
    
    return result


def verify_all_features(processed_dir: Path, pdb_list: List[str] = None) -> Tuple[List[Dict], Dict]:
    """Verify features for all PDBs in processed directory."""
    if pdb_list is None:
        # Auto-detect PDBs from directory structure
        pdb_list = [d.name for d in processed_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    results = []
    summary = {
        'total_pdbs': len(pdb_list),
        'has_sasa': 0,
        'has_charge': 0,
        'missing_sasa': 0,
        'missing_charge': 0,
        'all_zero_rsa': 0,
        'all_zero_charge': 0,
        'no_cips_labels': 0,
        'pdbs_with_issues': 0
    }
    
    for pdb_id in sorted(pdb_list):
        result = verify_pdb_features(processed_dir, pdb_id)
        results.append(result)
        
        if result['has_sasa']:
            summary['has_sasa'] += 1
        else:
            summary['missing_sasa'] += 1
        
        if result['has_charge']:
            summary['has_charge'] += 1
        else:
            summary['missing_charge'] += 1
        
        if result['sasa_stats'] and result['sasa_stats']['non_zero'] == 0:
            summary['all_zero_rsa'] += 1
        
        if result['charge_stats'] and result['charge_stats']['non_zero'] == 0:
            summary['all_zero_charge'] += 1
        
        if result['label_stats'] and result['label_stats']['label_1'] == 0:
            summary['no_cips_labels'] += 1
        
        if len(result['issues']) > 0:
            summary['pdbs_with_issues'] += 1
    
    return results, summary


def print_verification_report(results: List[Dict], summary: Dict):
    """Print verification report."""
    print("=" * 80)
    print("FEATURE VERIFICATION REPORT")
    print("=" * 80)
    
    print(f"\nSummary:")
    print(f"  Total PDBs checked: {summary['total_pdbs']}")
    print(f"  PDBs with 'sasa' column: {summary['has_sasa']}/{summary['total_pdbs']}")
    print(f"  PDBs with 'charge' column: {summary['has_charge']}/{summary['total_pdbs']}")
    print(f"  PDBs missing 'sasa': {summary['missing_sasa']}")
    print(f"  PDBs missing 'charge': {summary['missing_charge']}")
    print(f"  PDBs with all-zero RSA: {summary['all_zero_rsa']}")
    print(f"  PDBs with all-zero charge: {summary['all_zero_charge']}")
    print(f"  PDBs with no CIPS labels: {summary['no_cips_labels']}")
    print(f"  PDBs with issues: {summary['pdbs_with_issues']}")
    
    # Detailed results
    print("\n" + "=" * 80)
    print("DETAILED RESULTS")
    print("=" * 80)
    
    for result in results:
        status = "✓" if len(result['issues']) == 0 else "✗"
        print(f"\n{status} {result['pdb_id']}:")
        
        if result['has_sasa']:
            stats = result['sasa_stats']
            print(f"  RSA (sasa): {stats['non_zero']}/{stats['total']} non-zero, "
                  f"range=[{stats['min']:.4f}, {stats['max']:.4f}], mean={stats['mean']:.4f}")
        else:
            print(f"  RSA (sasa): MISSING")
        
        if result['has_charge']:
            stats = result['charge_stats']
            print(f"  Charge: {stats['non_zero']}/{stats['total']} non-zero, "
                  f"range=[{stats['min']:.4f}, {stats['max']:.4f}], mean={stats['mean']:.4f}")
        else:
            print(f"  Charge: MISSING")
        
        if result['label_stats']:
            ls = result['label_stats']
            total = ls['total']
            print(f"  Labels: 0={ls['label_0']} ({ls['label_0']/total*100:.1f}%), "
                  f"1={ls['label_1']} ({ls['label_1']/total*100:.1f}%), "
                  f"2={ls['label_2']} ({ls['label_2']/total*100:.1f}%)")
        
        if len(result['issues']) > 0:
            print(f"  Issues:")
            for issue in result['issues']:
                print(f"    - {issue}")
    
    # PDBs that need reprocessing
    need_reprocessing = [r for r in results if not r['has_sasa'] or not r['has_charge']]
    if need_reprocessing:
        print("\n" + "=" * 80)
        print("PDBs NEEDING REPROCESSING")
        print("=" * 80)
        for r in need_reprocessing:
            print(f"  {r['pdb_id']}: {', '.join(r['issues'])}")


def main():
    parser = argparse.ArgumentParser(description='Verify FreeSASA/PDB2PQR features in processed data')
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
    
    results, summary = verify_all_features(processed_dir, pdb_list)
    print_verification_report(results, summary)
    
    if args.output:
        import json
        output_data = {
            'summary': summary,
            'results': results
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    
    # Exit with error code if issues found
    if summary['pdbs_with_issues'] > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()

