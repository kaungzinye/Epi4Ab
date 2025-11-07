#!/usr/bin/env python3
"""
Comprehensive Validation Script for Epi4Ab Pipeline

This script performs ALL validation checks in one unified tool:
- Column presence (all 50 required columns including VH/VK/VLa families)
- Data integrity (distances, angles, sequences)
- Model compatibility (matches trained model requirements)
- File completeness (all required parquet and JSON files)
- Edge attribute correctness (no zeros in distances)
- Sequence validation (correct format and content)

Usage:
    # Validate all processed data
    python validate_all.py --processed_dir /path/to/processed/
    
    # Validate single PDB
    python validate_all.py --processed_dir /path/to/processed/ --pdb_id 1N8Z
    
    # Verbose output
    python validate_all.py --processed_dir /path/to/processed/ --verbose
    
    # Quiet mode (only show failures)
    python validate_all.py --processed_dir /path/to/processed/ --quiet
"""

import argparse
import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# ============================================================================
# EXPECTED SCHEMA DEFINITION
# ============================================================================

# Based on final_trained_Epi4Ab/log.json
EXPECTED_CONTINUOUS = [
    'resDepth', 'caDepth', 'psi', 'phi', 'omega', 'chi', 
    'aac', 'cc', 'H1_len', 'H2_len', 'H3_len', 'L1_len', 
    'L2_len', 'L3_len', 'H3_score', 'L1_score'
]

EXPECTED_ONEHOT = [
    'angleNan', 'chiNan',
    # VH (heavy chain) families
    'VH1', 'VH2', 'VH3', 'VH4', 'VH5', 'VH6', 'VH7', 'VH8', 
    'VH9', 'VH14', 'VH_unk',
    # VK (kappa light chain) families
    'VK1', 'VK2', 'VK3', 'VK4', 'VK5', 'VK6', 'VK8', 'VK10', 
    'VK12', 'VK13', 'VK14', 'VK_others',
    # VLa (lambda light chain) families - CRITICAL!
    'VLa1', 'VLa2', 'VLa3', 'VLa6', 'VLa_others', 'VL_unk'
]

REQUIRED_ADDITIONAL = [
    'resId',      # Residue ID
    'resShort',   # One-letter amino acid code
]

ALL_EXPECTED_COLUMNS = REQUIRED_ADDITIONAL + EXPECTED_CONTINUOUS + EXPECTED_ONEHOT

# Antibody family column groups for detailed checking
VH_FAMILIES = ['VH1', 'VH2', 'VH3', 'VH4', 'VH5', 'VH6', 'VH7', 'VH8', 'VH9', 'VH14', 'VH_unk']
VK_FAMILIES = ['VK1', 'VK2', 'VK3', 'VK4', 'VK5', 'VK6', 'VK8', 'VK10', 'VK12', 'VK13', 'VK14', 'VK_others']
VLA_FAMILIES = ['VLa1', 'VLa2', 'VLa3', 'VLa6', 'VLa_others', 'VL_unk']

# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_file_existence(pdb_dir: Path, pdb_id: str) -> Tuple[bool, List[str]]:
    """Check that all required files exist."""
    required_files = [
        'node_feature.parquet',
        'node_label_pi.parquet',
        'edge_index.parquet',
        'edge_attribute_dist.parquet',
        'edge_attribute_charge.parquet',
        'sequence/antigen_sequence.json',
        'sequence/cdr_sequence.json'  # Required for AntiBERTy
    ]
    
    missing = []
    for file in required_files:
        if not (pdb_dir / file).exists():
            missing.append(file)
    
    return len(missing) == 0, missing


def validate_node_features_columns(df: pd.DataFrame, pdb_id: str) -> Tuple[bool, Dict]:
    """
    Validate node features have all required columns.
    Returns detailed breakdown by column type.
    """
    actual_columns = set(df.columns)
    expected_columns = set(ALL_EXPECTED_COLUMNS)
    
    missing = sorted(list(expected_columns - actual_columns))
    extra = sorted(list(actual_columns - expected_columns))
    
    # Check each column group
    missing_additional = [c for c in REQUIRED_ADDITIONAL if c not in actual_columns]
    missing_continuous = [c for c in EXPECTED_CONTINUOUS if c not in actual_columns]
    missing_onehot = [c for c in EXPECTED_ONEHOT if c not in actual_columns]
    
    # Check antibody family columns specifically
    missing_vh = [c for c in VH_FAMILIES if c not in actual_columns]
    missing_vk = [c for c in VK_FAMILIES if c not in actual_columns]
    missing_vla = [c for c in VLA_FAMILIES if c not in actual_columns]
    
    is_valid = len(missing) == 0
    
    details = {
        'total_columns': len(actual_columns),
        'expected_columns': len(expected_columns),
        'missing': missing,
        'extra': extra,
        'missing_additional': missing_additional,
        'missing_continuous': missing_continuous,
        'missing_onehot': missing_onehot,
        'missing_vh': missing_vh,
        'missing_vk': missing_vk,
        'missing_vla': missing_vla,
    }
    
    return is_valid, details


def validate_node_features_values(df: pd.DataFrame, pdb_id: str) -> Tuple[bool, Dict]:
    """Validate node feature values are reasonable."""
    issues = []
    
    # Check for NaN values in critical columns
    critical_cols = ['resId', 'resShort', 'resDepth']
    for col in critical_cols:
        if col in df.columns:
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                issues.append(f"{col} has {nan_count} NaN values")
    
    # Check resShort contains valid amino acids
    if 'resShort' in df.columns:
        valid_aa = set('ACDEFGHIKLMNPQRSTVWYX')
        invalid = df[df['resShort'].apply(lambda x: x not in valid_aa)]
        if len(invalid) > 0:
            issues.append(f"resShort has {len(invalid)} invalid amino acid codes")
    
    # Check depth values are reasonable (0-20 Å)
    if 'resDepth' in df.columns:
        bad_depth = df[(df['resDepth'] < 0) | (df['resDepth'] > 30)]
        if len(bad_depth) > 0:
            issues.append(f"resDepth has {len(bad_depth)} values outside [0, 30] range")
    
    # Check angles are in valid range or NaN
    angle_cols = ['psi', 'phi', 'omega', 'chi']
    for col in angle_cols:
        if col in df.columns:
            # Angles should be [-180, 180] or NaN (replaced with 0)
            bad_angles = df[(df[col] < -180) | (df[col] > 180)]
            if len(bad_angles) > 0:
                issues.append(f"{col} has {len(bad_angles)} values outside [-180, 180]")
    
    # Check VH/VK/VLa families are binary (0 or 1)
    for family_group, family_list in [('VH', VH_FAMILIES), ('VK', VK_FAMILIES), ('VLa', VLA_FAMILIES)]:
        for col in family_list:
            if col in df.columns:
                non_binary = df[~df[col].isin([0, 1])]
                if len(non_binary) > 0:
                    issues.append(f"{col} has {len(non_binary)} non-binary values")
    
    is_valid = len(issues) == 0
    details = {'issues': issues}
    
    return is_valid, details


def validate_edge_attributes(pdb_dir: Path, pdb_id: str) -> Tuple[bool, Dict]:
    """
    Validate edge attribute files.
    This is where Bug #2 (critical edge attribute bug) would be caught.
    """
    issues = []
    details = {}
    
    # Check edge_attribute_dist.parquet
    dist_file = pdb_dir / 'edge_attribute_dist.parquet'
    try:
        df_dist = pd.read_parquet(dist_file)
        details['edge_count'] = len(df_dist)
        
        # Check has 'dist' column
        if 'dist' not in df_dist.columns:
            issues.append("edge_attribute_dist.parquet missing 'dist' column")
        else:
            # Check for zero or negative distances (Bug #2 symptom)
            zero_dist = df_dist[df_dist['dist'] <= 0]
            if len(zero_dist) > 0:
                issues.append(f"Found {len(zero_dist)} zero/negative distances (BUG #2 indicator!)")
            
            # Check distance range (should be 2-10 Å for CA-CA contacts)
            min_dist = df_dist['dist'].min()
            max_dist = df_dist['dist'].max()
            details['min_distance'] = float(min_dist)
            details['max_distance'] = float(max_dist)
            
            if min_dist < 2.0:
                issues.append(f"Minimum distance {min_dist:.2f} Å is suspiciously small")
            if max_dist > 12.0:
                issues.append(f"Maximum distance {max_dist:.2f} Å exceeds cutoff")
    
    except Exception as e:
        issues.append(f"Error reading edge_attribute_dist.parquet: {e}")
    
    # Check edge_attribute_charge.parquet
    charge_file = pdb_dir / 'edge_attribute_charge.parquet'
    try:
        df_charge = pd.read_parquet(charge_file)
        
        if 'charge' not in df_charge.columns:
            issues.append("edge_attribute_charge.parquet missing 'charge' column")
        else:
            # Check charge values are in expected range
            charge_values = df_charge['charge'].unique()
            details['charge_values'] = sorted([int(c) for c in charge_values])
            
            # Charges should be -1, 0, or 1
            invalid_charges = df_charge[~df_charge['charge'].isin([-1, 0, 1])]
            if len(invalid_charges) > 0:
                issues.append(f"Found {len(invalid_charges)} charges outside [-1, 0, 1]")
    
    except Exception as e:
        issues.append(f"Error reading edge_attribute_charge.parquet: {e}")
    
    is_valid = len(issues) == 0
    details['issues'] = issues
    
    return is_valid, details


def validate_sequence_file(pdb_dir: Path, pdb_id: str) -> Tuple[bool, Dict]:
    """
    Validate sequence file exists and has correct format.
    This is where Bug #3 (missing sequence files) would be caught.
    """
    issues = []
    details = {}
    
    seq_file = pdb_dir / 'sequence' / 'antigen_sequence.json'
    
    if not seq_file.exists():
        issues.append("Missing sequence/antigen_sequence.json (BUG #3!)")
        return False, {'issues': issues}
    
    try:
        with open(seq_file, 'r') as f:
            seq_data = json.load(f)
        
        if 'pdb_sequence' not in seq_data:
            issues.append("Missing 'pdb_sequence' key in JSON")
        else:
            sequence = seq_data['pdb_sequence']
            
            if not isinstance(sequence, str):
                issues.append("'pdb_sequence' is not a string")
            elif len(sequence) == 0:
                issues.append("Empty sequence")
            else:
                details['sequence_length'] = len(sequence)
                details['sequence_preview'] = sequence[:20] + '...' if len(sequence) > 20 else sequence
                
                # Check for valid amino acids
                valid_aa = set('ACDEFGHIKLMNPQRSTVWYX')
                invalid_aa = [aa for aa in sequence if aa not in valid_aa]
                if invalid_aa:
                    issues.append(f"Invalid amino acids in sequence: {set(invalid_aa)}")
    
    except Exception as e:
        issues.append(f"Error reading sequence file: {e}")
    
    is_valid = len(issues) == 0
    details['issues'] = issues
    
    return is_valid, details


def validate_sequence_consistency(pdb_dir: Path, pdb_id: str) -> Tuple[bool, Dict]:
    """
    Validate sequence length matches node features count.
    """
    issues = []
    details = {}
    
    try:
        # Load node features
        df_nodes = pd.read_parquet(pdb_dir / 'node_feature.parquet')
        node_count = len(df_nodes)
        details['node_count'] = node_count
        
        # Load sequence
        seq_file = pdb_dir / 'sequence' / 'antigen_sequence.json'
        if seq_file.exists():
            with open(seq_file, 'r') as f:
                seq_data = json.load(f)
            
            sequence = seq_data.get('pdb_sequence', '')
            seq_length = len(sequence)
            details['sequence_length'] = seq_length
            
            if seq_length != node_count:
                issues.append(f"Sequence length ({seq_length}) doesn't match node count ({node_count})")
        else:
            issues.append("Cannot check consistency - sequence file missing")
    
    except Exception as e:
        issues.append(f"Error checking consistency: {e}")
    
    is_valid = len(issues) == 0
    details['issues'] = issues
    
    return is_valid, details


def validate_cdr_sequences(pdb_dir: Path, pdb_id: str) -> Tuple[bool, Dict]:
    """
    Validate CDR sequence file exists and has correct format.
    Required for AntiBERTy feature extraction during inference.
    """
    issues = []
    details = {}
    
    cdr_file = pdb_dir / 'sequence' / 'cdr_sequence.json'
    
    if not cdr_file.exists():
        issues.append("Missing sequence/cdr_sequence.json (required for AntiBERTy!)")
        return False, {'issues': issues}
    
    try:
        with open(cdr_file, 'r') as f:
            cdr_data = json.load(f)
        
        # Expected CDR regions - check both key formats (H1 vs H1_seq)
        expected_cdrs_base = ['H1', 'H2', 'H3', 'L1', 'L2', 'L3']
        missing_cdrs = []
        present_cdrs = []
        
        for cdr_base in expected_cdrs_base:
            # Try both key formats
            cdr_key = cdr_base  # Try H1, H2, etc.
            cdr_key_alt = f"{cdr_base}_seq"  # Try H1_seq, H2_seq, etc.
            
            # Check which format exists
            if cdr_key in cdr_data:
                seq = cdr_data[cdr_key]
            elif cdr_key_alt in cdr_data:
                seq = cdr_data[cdr_key_alt]
            else:
                missing_cdrs.append(cdr_base)
                continue
            
            if isinstance(seq, str) and len(seq) > 0:
                # Validate amino acid codes
                valid_aa = set('ACDEFGHIKLMNPQRSTVWYX')
                invalid_aa = [aa for aa in seq if aa not in valid_aa]
                if invalid_aa:
                    issues.append(f"CDR {cdr_base} contains invalid amino acids: {set(invalid_aa)}")
                else:
                    present_cdrs.append(cdr_base)
                    details[f'{cdr_base}_length'] = len(seq)
            elif not isinstance(seq, str):
                issues.append(f"CDR {cdr_base} is not a string (got {type(seq).__name__})")
            # Empty strings are OK - some structures may not have antibody chains
        
        if missing_cdrs:
            # Missing keys is an issue (file format problem)
            issues.append(f"Missing CDR sequence keys: {', '.join(missing_cdrs)}")
        
        details['present_cdrs'] = present_cdrs
        details['missing_cdrs'] = missing_cdrs
        details['total_cdrs'] = len(present_cdrs)
        details['empty_cdrs'] = [cdr for cdr in expected_cdrs_base if cdr not in present_cdrs and cdr not in missing_cdrs]
        
        # Empty CDR sequences are OK (some structures may not have antibody chains)
        # Only fail if keys are completely missing (file format issue)
    
    except json.JSONDecodeError as e:
        issues.append(f"Invalid JSON in cdr_sequence.json: {e}")
    except Exception as e:
        issues.append(f"Error reading CDR sequence file: {e}")
    
    is_valid = len(issues) == 0
    details['issues'] = issues
    
    return is_valid, details


def validate_label_distribution(pdb_dir: Path, pdb_id: str) -> Tuple[bool, Dict]:
    """
    Validate label distribution makes sense.
    Label 2 (predicted epitopes) should not be >50% - indicates threshold issue.
    """
    issues = []
    details = {}
    
    label_file = pdb_dir / 'node_label_pi.parquet'
    
    if not label_file.exists():
        issues.append("Missing node_label_pi.parquet")
        return False, {'issues': issues}
    
    try:
        df_labels = pd.read_parquet(label_file)
        
        if 'isInterface' not in df_labels.columns:
            issues.append("Missing 'isInterface' column in node_label_pi.parquet")
            return False, {'issues': issues}
        
        total = len(df_labels)
        label_counts = df_labels['isInterface'].value_counts().sort_index()
        
        details['total_residues'] = total
        details['label_counts'] = {int(k): int(v) for k, v in label_counts.items()}
        details['label_percentages'] = {int(k): float(v/total*100) for k, v in label_counts.items()}
        
        # Check for expected labels
        has_label_0 = 0 in label_counts.index
        has_label_1 = 1 in label_counts.index
        has_label_2 = 2 in label_counts.index
        
        details['has_label_0'] = has_label_0
        details['has_label_1'] = has_label_1
        details['has_label_2'] = has_label_2
        
        # Warning if Label 2 is too high (>50% is suspicious)
        if has_label_2:
            label2_pct = label_counts[2] / total * 100
            details['label2_percentage'] = label2_pct
            
            if label2_pct > 50.0:
                issues.append(f"Label 2 (predicted epitopes) is {label2_pct:.1f}% - suspiciously high! "
                            f"Expected <30% typically. Threshold may be too low or bug in label extraction.")
            elif label2_pct > 30.0:
                issues.append(f"Label 2 (predicted epitopes) is {label2_pct:.1f}% - higher than expected "
                            f"(typically <30%). Verify threshold is correct.")
        
        # Warning if no Label 0 (all residues are epitopes - unlikely)
        if not has_label_0:
            issues.append("No Label 0 (non-epitopes) found - all residues are predicted epitopes (unlikely!)")
        
        # Warning if no Label 1 (no antibody contacts - check if structure has antibody)
        if not has_label_1:
            issues.append("No Label 1 (CIPS, antibody contacts) found - verify structure has antibody chains")
        
        # Check if Label 2 is present but no predictions available (should not happen)
        if has_label_2:
            # Try to check if BepiPred/Ellipro cache exists
            cache_dir = pdb_dir / 'epitope_predictions'
            bepipred_cache = cache_dir / f"{pdb_id}_bepipred.json"
            ellipro_cache = cache_dir / f"{pdb_id}_ellipro.json"
            
            if not bepipred_cache.exists() and not ellipro_cache.exists():
                # Check parent cache directory
                parent_cache = pdb_dir.parent.parent / 'epitope_predictions_cache'
                if parent_cache.exists():
                    bepipred_cache = parent_cache / f"{pdb_id}_bepipred.json"
                    ellipro_cache = parent_cache / f"{pdb_id}_ellipro.json"
                
                if not bepipred_cache.exists() and not ellipro_cache.exists():
                    issues.append("Label 2 present but no BepiPred/Ellipro cache found - verify predictions were used")
    
    except Exception as e:
        issues.append(f"Error reading label file: {e}")
    
    is_valid = len(issues) == 0
    details['issues'] = issues
    
    return is_valid, details


def validate_pdb(processed_dir: Path, pdb_id: str, verbose: bool = True) -> Tuple[bool, Dict]:
    """
    Perform comprehensive validation for a single PDB.
    Returns (is_valid, detailed_results)
    """
    pdb_dir = processed_dir / pdb_id
    results = {
        'pdb_id': pdb_id,
        'checks': {}
    }
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Validating: {pdb_id}")
        print(f"{'='*70}")
    
    all_valid = True
    
    # 1. File Existence
    files_valid, missing_files = validate_file_existence(pdb_dir, pdb_id)
    results['checks']['file_existence'] = {
        'valid': files_valid,
        'missing_files': missing_files
    }
    
    if not files_valid:
        if verbose or not files_valid:
            print(f"❌ File Existence: Missing {len(missing_files)} required files")
            for f in missing_files:
                print(f"   - {f}")
        all_valid = False
        return all_valid, results
    
    if verbose:
        print(f"✅ File Existence: All required files present")
    
    # 2. Node Features - Column Validation
    try:
        df_nodes = pd.read_parquet(pdb_dir / 'node_feature.parquet')
        cols_valid, col_details = validate_node_features_columns(df_nodes, pdb_id)
        results['checks']['node_features_columns'] = {
            'valid': cols_valid,
            **col_details
        }
        
        if cols_valid:
            if verbose:
                print(f"✅ Node Feature Columns: All {len(ALL_EXPECTED_COLUMNS)} columns present")
                print(f"   - Additional: resId, resShort")
                print(f"   - Continuous: {len(EXPECTED_CONTINUOUS)} features")
                print(f"   - One-hot: {len(EXPECTED_ONEHOT)} features")
                print(f"     • VH families: {len(VH_FAMILIES)} columns")
                print(f"     • VK families: {len(VK_FAMILIES)} columns")
                print(f"     • VLa families: {len(VLA_FAMILIES)} columns ✓")
        else:
            print(f"❌ Node Feature Columns: MISSING {len(col_details['missing'])} columns")
            
            if col_details['missing_additional']:
                print(f"   ❌ Missing additional: {', '.join(col_details['missing_additional'])}")
            if col_details['missing_continuous']:
                print(f"   ❌ Missing continuous: {', '.join(col_details['missing_continuous'])}")
            if col_details['missing_onehot']:
                print(f"   ❌ Missing one-hot: {', '.join(col_details['missing_onehot'])}")
            
            # Highlight antibody family issues (Bug #5)
            if col_details['missing_vh']:
                print(f"   ❌ Missing VH families: {', '.join(col_details['missing_vh'])}")
            if col_details['missing_vk']:
                print(f"   ❌ Missing VK families: {', '.join(col_details['missing_vk'])}")
            if col_details['missing_vla']:
                print(f"   ❌ Missing VLa families: {', '.join(col_details['missing_vla'])} (BUG #5!)")
            
            if col_details['extra']:
                print(f"   ℹ️  Extra columns: {', '.join(col_details['extra'])}")
            
            all_valid = False
        
        # 3. Node Features - Value Validation
        vals_valid, val_details = validate_node_features_values(df_nodes, pdb_id)
        results['checks']['node_features_values'] = {
            'valid': vals_valid,
            **val_details
        }
        
        if vals_valid:
            if verbose:
                print(f"✅ Node Feature Values: All values within expected ranges")
        else:
            print(f"❌ Node Feature Values: Found {len(val_details['issues'])} issues")
            for issue in val_details['issues']:
                print(f"   - {issue}")
            all_valid = False
    
    except Exception as e:
        print(f"❌ Error reading node features: {e}")
        results['checks']['node_features'] = {'valid': False, 'error': str(e)}
        all_valid = False
    
    # 4. Edge Attributes
    edges_valid, edge_details = validate_edge_attributes(pdb_dir, pdb_id)
    results['checks']['edge_attributes'] = {
        'valid': edges_valid,
        **edge_details
    }
    
    if edges_valid:
        if verbose:
            print(f"✅ Edge Attributes: Valid")
            print(f"   - Edges: {edge_details.get('edge_count', 'N/A')}")
            print(f"   - Distance range: {edge_details.get('min_distance', 0):.2f} - {edge_details.get('max_distance', 0):.2f} Å")
            print(f"   - Charge values: {edge_details.get('charge_values', [])}")
    else:
        print(f"❌ Edge Attributes: Found {len(edge_details['issues'])} issues")
        for issue in edge_details['issues']:
            print(f"   - {issue}")
        all_valid = False
    
    # 5. Sequence File
    seq_valid, seq_details = validate_sequence_file(pdb_dir, pdb_id)
    results['checks']['sequence_file'] = {
        'valid': seq_valid,
        **seq_details
    }
    
    if seq_valid:
        if verbose:
            print(f"✅ Sequence File: Valid")
            print(f"   - Length: {seq_details.get('sequence_length', 'N/A')} residues")
            print(f"   - Preview: {seq_details.get('sequence_preview', 'N/A')}")
    else:
        print(f"❌ Sequence File: Found {len(seq_details['issues'])} issues")
        for issue in seq_details['issues']:
            print(f"   - {issue}")
        all_valid = False
    
    # 6. Sequence Consistency
    consistency_valid, consistency_details = validate_sequence_consistency(pdb_dir, pdb_id)
    results['checks']['sequence_consistency'] = {
        'valid': consistency_valid,
        **consistency_details
    }
    
    if consistency_valid:
        if verbose:
            print(f"✅ Sequence Consistency: Sequence length matches node count")
    else:
        print(f"❌ Sequence Consistency: Mismatch detected")
        for issue in consistency_details['issues']:
            print(f"   - {issue}")
        all_valid = False
    
    # 7. CDR Sequences (for AntiBERTy)
    cdr_valid, cdr_details = validate_cdr_sequences(pdb_dir, pdb_id)
    results['checks']['cdr_sequences'] = {
        'valid': cdr_valid,
        **cdr_details
    }
    
    if cdr_valid:
        if verbose:
            print(f"✅ CDR Sequences: Valid")
            print(f"   - Present: {', '.join(cdr_details.get('present_cdrs', []))}")
            if cdr_details.get('present_cdrs'):
                for cdr in cdr_details['present_cdrs']:
                    length = cdr_details.get(f'{cdr}_length', 0)
                    print(f"     • {cdr}: {length} residues")
    else:
        print(f"❌ CDR Sequences: Found {len(cdr_details['issues'])} issues")
        for issue in cdr_details['issues']:
            print(f"   - {issue}")
        all_valid = False
    
    # 8. Label Distribution
    label_dist_valid, label_dist_details = validate_label_distribution(pdb_dir, pdb_id)
    results['checks']['label_distribution'] = {
        'valid': label_dist_valid,
        **label_dist_details
    }
    
    if label_dist_valid:
        if verbose:
            print(f"✅ Label Distribution: Valid")
            print(f"   - Total residues: {label_dist_details.get('total_residues', 'N/A')}")
            for label, count in label_dist_details.get('label_counts', {}).items():
                pct = label_dist_details.get('label_percentages', {}).get(label, 0)
                label_name = {0: 'Non-epitope', 1: 'CIPS (5Å)', 2: 'BepiPred/Ellipro'}.get(label, f'Unknown({label})')
                print(f"     • Label {label} ({label_name}): {count:4d} ({pct:5.2f}%)")
    else:
        print(f"⚠️  Label Distribution: Found {len(label_dist_details['issues'])} issues")
        for issue in label_dist_details['issues']:
            print(f"   - {issue}")
        # Don't fail validation, but warn
        if verbose:
            print(f"   Label counts: {label_dist_details.get('label_counts', {})}")
            print(f"   Label percentages: {label_dist_details.get('label_percentages', {})}")
    
    # Final Summary
    if verbose:
        if all_valid:
            print(f"\n✅✅✅ {pdb_id}: ALL VALIDATIONS PASSED ✅✅✅")
        else:
            print(f"\n❌❌❌ {pdb_id}: VALIDATION FAILED ❌❌❌")
    
    results['overall_valid'] = all_valid
    return all_valid, results


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive validation for Epi4Ab processed data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Validate all processed structures
    python validate_all.py --processed_dir /leonardo_scratch/fast/AIFAC_F01_302/epi4ab/processed/
    
    # Validate single structure
    python validate_all.py --processed_dir /path/to/processed --pdb_id 1N8Z
    
    # Verbose output
    python validate_all.py --processed_dir /path/to/processed --verbose
    
    # Quiet mode (only failures)
    python validate_all.py --processed_dir /path/to/processed --quiet
        """
    )
    parser.add_argument(
        '--processed_dir',
        type=str,
        required=True,
        help='Directory containing processed PDB files'
    )
    parser.add_argument(
        '--pdb_id',
        type=str,
        help='Specific PDB ID to validate (if not provided, validates all)'
    )
    parser.add_argument(
        '--model_log',
        type=str,
        default='final_trained_Epi4Ab/log.json',
        help='Path to model log.json to verify column requirements'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed validation output'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Only show failures and summary'
    )
    parser.add_argument(
        '--json_output',
        type=str,
        help='Save detailed results to JSON file'
    )
    
    args = parser.parse_args()
    
    processed_dir = Path(args.processed_dir)
    if not processed_dir.exists():
        print(f"❌ Error: Directory not found: {processed_dir}")
        sys.exit(1)
    
    verbose = args.verbose and not args.quiet
    
    # Verify expected columns against model log if available
    if args.model_log:
        model_log_path = Path(args.model_log)
        if model_log_path.exists():
            try:
                with open(model_log_path, 'r') as f:
                    log_data = json.load(f)
                
                model_continuous = set(log_data.get('continuous_columns', []))
                model_onehot = set(log_data.get('onehot_columns', []))
                
                print(f"\n{'='*70}")
                print(f"Model Requirements from {args.model_log}")
                print(f"{'='*70}")
                print(f"Continuous columns: {len(model_continuous)}")
                print(f"One-hot columns: {len(model_onehot)}")
                print(f"Total feature columns: {len(model_continuous) + len(model_onehot)}")
                print(f"Additional required: resId, resShort")
                print(f"Total expected: {len(ALL_EXPECTED_COLUMNS)} columns")
                
                # Verify our expectations match the model
                if model_continuous != set(EXPECTED_CONTINUOUS):
                    print("⚠️  WARNING: EXPECTED_CONTINUOUS doesn't match model!")
                    missing = model_continuous - set(EXPECTED_CONTINUOUS)
                    if missing:
                        print(f"   Missing in validator: {missing}")
                
                if model_onehot != set(EXPECTED_ONEHOT):
                    print("⚠️  WARNING: EXPECTED_ONEHOT doesn't match model!")
                    missing = model_onehot - set(EXPECTED_ONEHOT)
                    if missing:
                        print(f"   Missing in validator: {missing}")
                else:
                    print("✅ Validator schema matches model requirements")
                    
            except Exception as e:
                print(f"⚠️  Could not read model log: {e}")
    
    # Validate PDB(s)
    all_results = {}
    
    if args.pdb_id:
        # Single PDB validation
        success, results = validate_pdb(processed_dir, args.pdb_id, verbose=verbose)
        all_results[args.pdb_id] = results
    else:
        # Validate all PDBs in directory
        pdb_dirs = [d for d in processed_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
        
        if not pdb_dirs:
            print(f"❌ No PDB directories found in {processed_dir}")
            sys.exit(1)
        
        print(f"\n{'='*70}")
        print(f"Validating {len(pdb_dirs)} PDB structures")
        print(f"{'='*70}")
        
        for pdb_dir in sorted(pdb_dirs):
            pdb_id = pdb_dir.name
            success, results = validate_pdb(processed_dir, pdb_id, verbose=verbose)
            all_results[pdb_id] = results
    
    # Save JSON output if requested
    if args.json_output:
        with open(args.json_output, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nDetailed results saved to: {args.json_output}")
    
    # Summary
    print(f"\n{'='*70}")
    print(f"VALIDATION SUMMARY")
    print(f"{'='*70}")
    
    passed = sum(1 for r in all_results.values() if r.get('overall_valid', False))
    failed = len(all_results) - passed
    
    print(f"Total structures: {len(all_results)}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    
    if failed > 0:
        print(f"\nFailed structures:")
        for pdb_id, results in all_results.items():
            if not results.get('overall_valid', False):
                print(f"  - {pdb_id}")
                # Show which checks failed
                failed_checks = [k for k, v in results['checks'].items() if not v.get('valid', True)]
                if failed_checks:
                    print(f"    Failed checks: {', '.join(failed_checks)}")
        sys.exit(1)
    else:
        print(f"\n🎉🎉🎉 ALL VALIDATIONS PASSED! 🎉🎉🎉")
        print(f"\n✅ Ready for inference!")
        sys.exit(0)


if __name__ == '__main__':
    main()

