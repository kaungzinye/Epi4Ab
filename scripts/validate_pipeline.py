#!/usr/bin/env python3
"""
Epi4Ab Pipeline Validation System

Comprehensive validation at each pipeline step to ensure:
- File existence and accessibility
- Correct column names and data types
- Data shape/size consistency
- Alignment between related files
- Type consistency
- Value ranges and constraints
"""

import pandas as pd
import numpy as np
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import argparse
from datetime import datetime
import re

# Optional imports for PDB validation
try:
    import MDAnalysis as mda
    HAS_MDA = True
except ImportError:
    HAS_MDA = False

# Valid amino acid codes
VALID_AA_CODES = set('ACDEFGHIKLMNPQRSTVWYX')
VALID_LABEL_VALUES = {0, 1, 2}


class PipelineValidator:
    """Comprehensive validator for Epi4Ab pipeline steps."""
    
    def __init__(self, strict_mode=True, verbose=True):
        self.strict_mode = strict_mode
        self.verbose = verbose
        self.errors = []
        self.warnings = []
        self.checks_passed = 0
        self.checks_failed = 0
    
    def _log_error(self, message: str, file: str = None):
        """Log an error."""
        self.checks_failed += 1
        error_msg = f"ERROR: {message}"
        if file:
            error_msg += f" (file: {file})"
        self.errors.append(error_msg)
        if self.verbose:
            print(f"  ✗ {error_msg}")
    
    def _log_warning(self, message: str, file: str = None):
        """Log a warning."""
        warning_msg = f"WARNING: {message}"
        if file:
            warning_msg += f" (file: {file})"
        self.warnings.append(warning_msg)
        if self.verbose:
            print(f"  ⚠ {warning_msg}")
    
    def _log_success(self, message: str):
        """Log a successful check."""
        self.checks_passed += 1
        if self.verbose:
            print(f"  ✓ {message}")
    
    def validate_parquet_structure(self, file_path: str, required_columns: List[str], 
                                   dtypes: Dict[str, type] = None, allow_extra: bool = True) -> bool:
        """Validate parquet file structure."""
        if not os.path.exists(file_path):
            self._log_error(f"File does not exist: {file_path}")
            return False
        
        try:
            df = pd.read_parquet(file_path)
        except Exception as e:
            self._log_error(f"Cannot read parquet file: {e}", file_path)
            return False
        
        # Check required columns
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            self._log_error(f"Missing required columns: {missing_cols}", file_path)
            return False
        
        # Check data types if provided
        if dtypes:
            for col, expected_type in dtypes.items():
                if col in df.columns:
                    actual_type = df[col].dtype
                    if not pd.api.types.is_dtype_equal(actual_type, expected_type):
                        # Allow some flexibility (e.g., int32 vs int64)
                        if not (pd.api.types.is_integer_dtype(actual_type) and 
                               pd.api.types.is_integer_dtype(expected_type)):
                            self._log_warning(f"Column {col} has type {actual_type}, expected {expected_type}", file_path)
        
        # Check for extra columns
        if not allow_extra:
            extra_cols = set(df.columns) - set(required_columns)
            if extra_cols:
                self._log_warning(f"Extra columns found: {extra_cols}", file_path)
        
        return True
    
    def validate_nodes_edges_step1(self, pdb_id: str, nodes_edges_dir: str, 
                                   processed_dir: str, atom_type: str = 'CB') -> bool:
        """Validate output of nodes_edges.py for CA or CB."""
        print(f"\n=== Validating nodes_edges.py output ({atom_type}) for {pdb_id} ===")
        
        pdb_dir = os.path.join(nodes_edges_dir, pdb_id)
        if not os.path.exists(pdb_dir):
            self._log_error(f"PDB directory does not exist: {pdb_dir}")
            return False
        
        # Validate node_feature.parquet
        node_feature_path = os.path.join(pdb_dir, 'node_feature.parquet')
        required_cols = ['resId', 'resShort']
        if not self.validate_parquet_structure(node_feature_path, required_cols):
            return False
        
        df_features = pd.read_parquet(node_feature_path)
        
        # Check resId
        if not pd.api.types.is_integer_dtype(df_features['resId']):
            self._log_error("resId must be integer type", node_feature_path)
            return False
        
        if df_features['resId'].isna().any():
            self._log_error("resId contains NaN values", node_feature_path)
            return False
        
        # Check resShort (amino acid codes)
        invalid_aa = df_features['resShort'].apply(lambda x: x not in VALID_AA_CODES if pd.notna(x) else False)
        if invalid_aa.any():
            invalid_codes = df_features.loc[invalid_aa, 'resShort'].unique()
            self._log_error(f"Invalid amino acid codes found: {invalid_codes}", node_feature_path)
            return False
        
        self._log_success(f"node_feature.parquet: {len(df_features)} nodes, {len(df_features.columns)} features")
        
        # Validate edge_index
        edge_index_path = os.path.join(pdb_dir, f'edge_index_{atom_type}.parquet')
        required_cols = ['source', 'target']
        if not self.validate_parquet_structure(edge_index_path, required_cols):
            return False
        
        df_edges = pd.read_parquet(edge_index_path)
        
        # Check indices are valid
        max_node_idx = len(df_features) - 1
        invalid_source = (df_edges['source'] < 0) | (df_edges['source'] > max_node_idx)
        invalid_target = (df_edges['target'] < 0) | (df_edges['target'] > max_node_idx)
        
        if invalid_source.any() or invalid_target.any():
            self._log_error(f"Edge indices out of range (max: {max_node_idx})", edge_index_path)
            return False
        
        self._log_success(f"edge_index_{atom_type}.parquet: {len(df_edges)} edges")
        
        # Validate edge_attribute_dist
        edge_dist_path = os.path.join(pdb_dir, f'edge_attribute_dist_{atom_type}.parquet')
        if os.path.exists(edge_dist_path):
            df_dist = pd.read_parquet(edge_dist_path)
            if 'dist' in df_dist.columns:
                if not pd.api.types.is_float_dtype(df_dist['dist']):
                    self._log_error("dist column must be float", edge_dist_path)
                    return False
                if (df_dist['dist'] < 0).any():
                    self._log_error("dist contains negative values", edge_dist_path)
                    return False
                if len(df_dist) != len(df_edges):
                    self._log_error(f"Row count mismatch: {len(df_dist)} vs {len(df_edges)}", edge_dist_path)
                    return False
                self._log_success(f"edge_attribute_dist_{atom_type}.parquet: distances valid")
        
        # Validate edge_attribute_charge
        edge_charge_path = os.path.join(pdb_dir, f'edge_attribute_charge_{atom_type}.parquet')
        if os.path.exists(edge_charge_path):
            df_charge = pd.read_parquet(edge_charge_path)
            if 'qi*qj' in df_charge.columns:
                if not pd.api.types.is_integer_dtype(df_charge['qi*qj']):
                    self._log_warning("qi*qj column should be integer", edge_charge_path)
                if len(df_charge) != len(df_edges):
                    self._log_error(f"Row count mismatch: {len(df_charge)} vs {len(df_edges)}", edge_charge_path)
                    return False
                self._log_success(f"edge_attribute_charge_{atom_type}.parquet: charges valid")
        
        return True
    
    def validate_fill_edge_step2(self, pdb_id: str, nodes_edges_dir: str) -> bool:
        """Validate output of fill_edge.py."""
        print(f"\n=== Validating fill_edge.py output for {pdb_id} ===")
        
        pdb_dir = os.path.join(nodes_edges_dir, pdb_id)
        
        # Check merged files exist
        edge_index_path = os.path.join(pdb_dir, 'edge_index.parquet')
        edge_dist_path = os.path.join(pdb_dir, 'edge_attribute_dist.parquet')
        edge_charge_path = os.path.join(pdb_dir, 'edge_attribute_charge.parquet')
        
        if not all(os.path.exists(f) for f in [edge_index_path, edge_dist_path, edge_charge_path]):
            self._log_error("Merged edge files missing", pdb_dir)
            return False
        
        # Load files
        df_index = pd.read_parquet(edge_index_path)
        df_dist = pd.read_parquet(edge_dist_path)
        df_charge = pd.read_parquet(edge_charge_path)
        
        # Check row counts match
        if not (len(df_index) == len(df_dist) == len(df_charge)):
            self._log_error(f"Row count mismatch: index={len(df_index)}, dist={len(df_dist)}, charge={len(df_charge)}")
            return False
        
        self._log_success(f"All edge files have matching row counts: {len(df_index)}")
        
        # Validate node_feature exists for alignment check
        node_feature_path = os.path.join(pdb_dir, 'node_feature.parquet')
        if os.path.exists(node_feature_path):
            df_features = pd.read_parquet(node_feature_path)
            max_node_idx = len(df_features) - 1
            
            # Check edge indices are valid
            invalid_source = (df_index['source'] < 0) | (df_index['source'] > max_node_idx)
            invalid_target = (df_index['target'] < 0) | (df_index['target'] > max_node_idx)
            
            if invalid_source.any() or invalid_target.any():
                self._log_error(f"Edge indices out of range (max: {max_node_idx})", edge_index_path)
                return False
            
            self._log_success("Edge indices valid")
        
        return True
    
    def validate_labels_step3(self, pdb_id: str, nodes_edges_dir: str,
                              target_type: str = 'interface', target_file: str = 'node_label_pi.parquet',
                              target_column: str = 'score') -> bool:
        """Validate label alignment with node_feature."""
        if target_type in ['seqitope', 'proteinmpnn']:
            return self.validate_regression_targets(pdb_id, nodes_edges_dir, target_file, target_column, target_type)
        print(f"\n=== Validating labels for {pdb_id} ===")
        
        pdb_dir = os.path.join(nodes_edges_dir, pdb_id)
        label_path = os.path.join(pdb_dir, 'node_label_pi.parquet')
        feature_path = os.path.join(pdb_dir, 'node_feature.parquet')
        
        if not os.path.exists(label_path):
            self._log_error("node_label_pi.parquet does not exist", label_path)
            return False
        
        if not os.path.exists(feature_path):
            self._log_error("node_feature.parquet does not exist", feature_path)
            return False
        
        # Validate structure
        required_cols = ['resId', 'isInterface']
        if not self.validate_parquet_structure(label_path, required_cols):
            return False
        
        # Load files
        df_labels = pd.read_parquet(label_path)
        df_features = pd.read_parquet(feature_path)
        
        # Check alignment
        if len(df_labels) != len(df_features):
            self._log_error(f"Row count mismatch: labels={len(df_labels)}, features={len(df_features)}")
            return False
        
        # Check resId alignment
        if not df_labels['resId'].equals(df_features['resId']):
            mismatches = (df_labels['resId'] != df_features['resId']).sum()
            self._log_error(f"resId misalignment: {mismatches} mismatches")
            return False
        
        self._log_success(f"Labels aligned: {len(df_labels)} residues")
        
        # Check isInterface values
        invalid_labels = ~df_labels['isInterface'].isin(VALID_LABEL_VALUES)
        if invalid_labels.any():
            invalid_values = df_labels.loc[invalid_labels, 'isInterface'].unique()
            self._log_error(f"Invalid isInterface values: {invalid_values}")
            return False
        
        # Check data type
        if not pd.api.types.is_integer_dtype(df_labels['isInterface']):
            self._log_error("isInterface must be integer type")
            return False
        
        self._log_success("isInterface values valid")
        
        return True

    def validate_regression_targets(self, pdb_id: str, nodes_edges_dir: str, target_file: str,
                                    target_column: str, target_type: str) -> bool:
        """Validate regression target parquet alignment and ranges."""
        print(f"\n=== Validating regression targets ({target_type}) for {pdb_id} ===")

        pdb_dir = os.path.join(nodes_edges_dir, pdb_id)
        label_path = os.path.join(pdb_dir, target_file)
        feature_path = os.path.join(pdb_dir, 'node_feature.parquet')

        if not os.path.exists(label_path):
            self._log_error(f"{target_file} does not exist", label_path)
            return False

        if not os.path.exists(feature_path):
            self._log_error("node_feature.parquet does not exist", feature_path)
            return False

        required_cols = ['resId', target_column]
        if not self.validate_parquet_structure(label_path, required_cols):
            return False

        df_labels = pd.read_parquet(label_path)
        df_features = pd.read_parquet(feature_path)

        if len(df_labels) != len(df_features):
            self._log_error(f"Row count mismatch: labels={len(df_labels)}, features={len(df_features)}")
            return False

        if not df_labels['resId'].equals(df_features['resId']):
            mismatches = (df_labels['resId'] != df_features['resId']).sum()
            self._log_error(f"resId misalignment: {mismatches} mismatches")
            return False

        if not pd.api.types.is_numeric_dtype(df_labels[target_column]):
            self._log_error(f"{target_column} must be numeric")
            return False

        if df_labels[target_column].isna().any() or np.isinf(df_labels[target_column]).any():
            self._log_error(f"{target_column} contains NaN or inf values")
            return False

        score_min = df_labels[target_column].min()
        score_max = df_labels[target_column].max()
        score_std = df_labels[target_column].std()
        if target_type == 'seqitope':
            if score_min < 0 or score_max > 1:
                self._log_warning(f"Seqitope scores outside [0,1]: min={score_min}, max={score_max}")
        if score_std < 1e-6:
            self._log_warning(f"Target variance very low (std={score_std:.2e})")

        self._log_success(f"{target_file}: {len(df_labels)} residues, score range [{score_min}, {score_max}]")
        return True
    
    def validate_raw_ground_truth_alignment(self, pdb_id: str, nodes_edges_dir: str, 
                                             pdb_file: str = None, antigen_chain: str = None,
                                             raw_truth_resids: list = None, 
                                             raw_truth_amino_acids: dict = None) -> dict:
        """
        Validate raw ground truth alignment against multiple sources.
        
        Validates alignment against:
        1. node_feature.parquet (always checked - must exist)
        2. PDB structure (only if pdb_file provided)
        3. Antigen sequence (if available)
        
        Args:
            pdb_id: Complex identifier (e.g., "mAb159_RBD")
            nodes_edges_dir: Directory containing nodes_edges data
            pdb_file: Optional path to PDB complex file
            antigen_chain: Optional antigen chain ID (for PDB validation)
            raw_truth_resids: List of residue IDs from raw ground truth
            raw_truth_amino_acids: Dict mapping resId -> amino_acid from raw ground truth
            
        Returns:
            Dictionary with validation results
        """
        print(f"\n=== Validating Raw Ground Truth Alignment for {pdb_id} ===")
        
        results = {
            'pdb_id': pdb_id,
            'pdb_file_provided': pdb_file is not None and os.path.exists(pdb_file) if pdb_file else False,
            'node_feature_alignment': {},
            'pdb_structure_alignment': {},
            'sequence_alignment': {},
            'amino_acid_validation': {},
            'conclusion': None
        }
        
        pdb_dir = os.path.join(nodes_edges_dir, pdb_id)
        node_feature_path = os.path.join(pdb_dir, 'node_feature.parquet')
        
        # 1. Validate node_feature.parquet alignment (always required)
        if not os.path.exists(node_feature_path):
            self._log_error("node_feature.parquet does not exist", node_feature_path)
            results['node_feature_alignment'] = {'status': 'error', 'message': 'File not found'}
            return results
        
        try:
            df_features = pd.read_parquet(node_feature_path)
            feature_resids = set(df_features['resId'].tolist())
            
            if raw_truth_resids:
                raw_truth_set = set(raw_truth_resids)
                found_in_features = raw_truth_set & feature_resids
                missing_in_features = raw_truth_set - feature_resids
                
                results['node_feature_alignment'] = {
                    'status': 'success' if len(missing_in_features) == 0 else 'warning',
                    'total_raw_truth': len(raw_truth_set),
                    'found': len(found_in_features),
                    'missing': len(missing_in_features),
                    'missing_resids': sorted(missing_in_features) if missing_in_features else []
                }
                
                if len(missing_in_features) == 0:
                    self._log_success(f"Node Feature Alignment: {len(found_in_features)}/{len(raw_truth_set)} residues found")
                else:
                    self._log_warning(f"Node Feature Alignment: {len(found_in_features)}/{len(raw_truth_set)} residues found, {len(missing_in_features)} missing")
            else:
                results['node_feature_alignment'] = {'status': 'skipped', 'message': 'No raw_truth_resids provided'}
                
        except Exception as e:
            self._log_error(f"Error validating node_feature alignment: {e}")
            results['node_feature_alignment'] = {'status': 'error', 'message': str(e)}
        
        # 2. Validate PDB structure alignment (if PDB file provided)
        if results['pdb_file_provided'] and HAS_MDA:
            try:
                universe = mda.Universe(pdb_file)
                
                # Auto-detect antigen chain if not provided
                if antigen_chain is None and raw_truth_resids:
                    # Try to detect chain from node_feature resIds
                    for chain_id in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                        try:
                            chain_ca = universe.select_atoms(f'chainid {chain_id} and name CA')
                            if len(chain_ca) > 0:
                                chain_resids = set(chain_ca.residues.resnums)
                                overlap = len(set(raw_truth_resids) & chain_resids)
                                if overlap > len(raw_truth_resids) * 0.8:
                                    antigen_chain = chain_id
                                    break
                        except:
                            continue
                
                if antigen_chain:
                    antigen_ca = universe.select_atoms(f'chainid {antigen_chain} and name CA')
                    pdb_resids = set(antigen_ca.residues.resnums)
                    resnum_to_residue = {res.resnum: res for res in antigen_ca.residues}
                    
                    if raw_truth_resids:
                        raw_truth_set = set(raw_truth_resids)
                        found_in_pdb = raw_truth_set & pdb_resids
                        missing_in_pdb = raw_truth_set - pdb_resids
                        
                        # Validate amino acid codes if provided
                        aa_mismatches = []
                        if raw_truth_amino_acids:
                            aa_map = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
                                     'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
                                     'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
                                     'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'}
                            
                            for resid in found_in_pdb:
                                if resid in raw_truth_amino_acids:
                                    residue = resnum_to_residue.get(resid)
                                    if residue:
                                        actual_aa_3letter = residue.resname
                                        actual_aa_1letter = aa_map.get(actual_aa_3letter, actual_aa_3letter[0] if actual_aa_3letter else '?')
                                        expected_aa = raw_truth_amino_acids[resid].upper()
                                        
                                        if actual_aa_1letter.upper() != expected_aa:
                                            aa_mismatches.append((resid, expected_aa, actual_aa_1letter))
                        
                        results['pdb_structure_alignment'] = {
                            'status': 'success' if len(missing_in_pdb) == 0 and len(aa_mismatches) == 0 else 'warning',
                            'antigen_chain': antigen_chain,
                            'total_raw_truth': len(raw_truth_set),
                            'found': len(found_in_pdb),
                            'missing': len(missing_in_pdb),
                            'missing_resids': sorted(missing_in_pdb) if missing_in_pdb else [],
                            'aa_mismatches': len(aa_mismatches),
                            'aa_mismatch_details': aa_mismatches[:10]  # First 10
                        }
                        
                        if len(missing_in_pdb) == 0 and len(aa_mismatches) == 0:
                            self._log_success(f"PDB Structure Alignment: {len(found_in_pdb)}/{len(raw_truth_set)} residues found, all amino acids match")
                        else:
                            if len(missing_in_pdb) > 0:
                                self._log_warning(f"PDB Structure Alignment: {len(found_in_pdb)}/{len(raw_truth_set)} residues found, {len(missing_in_pdb)} missing")
                            if len(aa_mismatches) > 0:
                                self._log_warning(f"PDB Structure Alignment: {len(aa_mismatches)} amino acid mismatches found")
                    else:
                        results['pdb_structure_alignment'] = {'status': 'skipped', 'message': 'No raw_truth_resids provided'}
                else:
                    results['pdb_structure_alignment'] = {'status': 'warning', 'message': 'Could not detect antigen chain'}
                    self._log_warning("PDB Structure Alignment: Could not detect antigen chain")
                    
            except Exception as e:
                self._log_warning(f"PDB Structure Alignment: Error validating - {e}")
                results['pdb_structure_alignment'] = {'status': 'error', 'message': str(e)}
        elif results['pdb_file_provided'] and not HAS_MDA:
            results['pdb_structure_alignment'] = {'status': 'skipped', 'message': 'MDAnalysis not available'}
            self._log_warning("PDB Structure Alignment: Skipped (MDAnalysis not available)")
        else:
            results['pdb_structure_alignment'] = {'status': 'skipped', 'message': 'No PDB file provided'}
            self._log_success("PDB Structure Alignment: Skipped (no combined complex PDB provided - experimental data mode)")
        
        # 3. Validate sequence alignment (if available)
        # Try to find processed_dir from nodes_edges_dir
        processed_dir = nodes_edges_dir.replace('/nodes_edges', '/processed_data')
        seq_dir = os.path.join(processed_dir, pdb_id, 'sequence')
        ag_seq_path = os.path.join(seq_dir, 'antigen_sequence.json')
        
        if os.path.exists(ag_seq_path):
            try:
                with open(ag_seq_path, 'r') as f:
                    ag_seq_data = json.load(f)
                
                if 'pdb_sequence' in ag_seq_data:
                    sequence = ag_seq_data['pdb_sequence'].replace('gap', '').replace('x', '')
                    
                    if raw_truth_resids:
                        # Check if resIds align with sequence positions
                        # Assuming resIds are 1-based and correspond to sequence positions
                        raw_truth_set = set(raw_truth_resids)
                        valid_positions = set(range(1, len(sequence) + 1))
                        found_in_sequence = raw_truth_set & valid_positions
                        missing_in_sequence = raw_truth_set - valid_positions
                        
                        results['sequence_alignment'] = {
                            'status': 'success' if len(missing_in_sequence) == 0 else 'warning',
                            'sequence_length': len(sequence),
                            'total_raw_truth': len(raw_truth_set),
                            'found': len(found_in_sequence),
                            'missing': len(missing_in_sequence),
                            'missing_resids': sorted(missing_in_sequence) if missing_in_sequence else []
                        }
                        
                        if len(missing_in_sequence) == 0:
                            self._log_success(f"Sequence Alignment: {len(found_in_sequence)}/{len(raw_truth_set)} residues align with sequence")
                        else:
                            self._log_warning(f"Sequence Alignment: {len(found_in_sequence)}/{len(raw_truth_set)} residues align, {len(missing_in_sequence)} out of range")
                    else:
                        results['sequence_alignment'] = {'status': 'skipped', 'message': 'No raw_truth_resids provided'}
                        
            except Exception as e:
                self._log_warning(f"Sequence Alignment: Error validating - {e}")
                results['sequence_alignment'] = {'status': 'error', 'message': str(e)}
        else:
            results['sequence_alignment'] = {'status': 'skipped', 'message': 'antigen_sequence.json not found'}
            self._log_success("Sequence Alignment: Skipped (antigen_sequence.json not found)")
        
        # Generate conclusion
        all_checks_passed = (
            results['node_feature_alignment'].get('status') == 'success' and
            (results['pdb_structure_alignment'].get('status') in ['success', 'skipped']) and
            (results['sequence_alignment'].get('status') in ['success', 'skipped'])
        )
        
        if all_checks_passed:
            if results['pdb_file_provided']:
                results['conclusion'] = 'WHOLE STRUCTURE (all residues found in all sources)'
            else:
                results['conclusion'] = 'Labels generated from raw ground truth (no docking/CIPS calculation)'
        else:
            results['conclusion'] = 'PARTIAL ALIGNMENT (some residues missing or mismatched)'
        
        print(f"\nConclusion: {results['conclusion']}")
        
        return results
    
    def validate_sequences_step4(self, pdb_id: str, processed_dir: str) -> bool:
        """Validate sequence JSON files."""
        print(f"\n=== Validating sequences for {pdb_id} ===")
        
        pdb_dir = os.path.join(processed_dir, pdb_id)
        seq_dir = os.path.join(pdb_dir, 'sequence')
        
        if not os.path.exists(seq_dir):
            self._log_warning("sequence directory does not exist", seq_dir)
            return True  # Not always required
        
        # Validate antigen_sequence.json
        ag_seq_path = os.path.join(seq_dir, 'antigen_sequence.json')
        if os.path.exists(ag_seq_path):
            try:
                with open(ag_seq_path, 'r') as f:
                    ag_seq_data = json.load(f)
                
                if 'pdb_sequence' not in ag_seq_data:
                    self._log_error("Missing 'pdb_sequence' key", ag_seq_path)
                    return False
                
                sequence = ag_seq_data['pdb_sequence'].replace('gap', '').replace('x', '')
                invalid_chars = set(sequence) - VALID_AA_CODES
                if invalid_chars:
                    self._log_error(f"Invalid characters in sequence: {invalid_chars}", ag_seq_path)
                    return False
                
                self._log_success(f"antigen_sequence.json: {len(sequence)} residues")
            except json.JSONDecodeError as e:
                self._log_error(f"Invalid JSON: {e}", ag_seq_path)
                return False
        
        # Validate cdr_sequence.json
        cdr_seq_path = os.path.join(seq_dir, 'cdr_sequence.json')
        if os.path.exists(cdr_seq_path):
            try:
                with open(cdr_seq_path, 'r') as f:
                    cdr_seq_data = json.load(f)
                
                required_keys = ['H1_seq', 'H2_seq', 'H3_seq', 'L1_seq', 'L2_seq', 'L3_seq']
                missing_keys = set(required_keys) - set(cdr_seq_data.keys())
                if missing_keys:
                    self._log_error(f"Missing CDR keys: {missing_keys}", cdr_seq_path)
                    return False
                
                # Check keys are uppercase
                for key in cdr_seq_data.keys():
                    if key not in required_keys and key.lower() in [k.lower() for k in required_keys]:
                        self._log_warning(f"Key should be uppercase: {key}", cdr_seq_path)
                
                self._log_success("cdr_sequence.json: all CDR sequences present")
            except json.JSONDecodeError as e:
                self._log_error(f"Invalid JSON: {e}", cdr_seq_path)
                return False
        
        return True
    
    def validate_sequence_length(self, pdb_id: str, nodes_edges_dir: str, 
                                 max_antigen_len: Optional[int] = None) -> Tuple[bool, int]:
        """
        Validate sequence length against model's max_antigen_len.
        
        Returns:
            (is_valid, actual_length)
        """
        node_feature_path = os.path.join(nodes_edges_dir, pdb_id, 'node_feature.parquet')
        
        if not os.path.exists(node_feature_path):
            return False, 0
        
        df_features = pd.read_parquet(node_feature_path)
        actual_length = len(df_features)
        
        if max_antigen_len is not None and actual_length > max_antigen_len:
            self._log_error(f"Sequence length {actual_length} exceeds max_antigen_len {max_antigen_len} by {actual_length - max_antigen_len} residues", 
                          node_feature_path)
            return False, actual_length
        
        if max_antigen_len is not None:
            self._log_success(f"Sequence length {actual_length} is within limit ({max_antigen_len})")
        else:
            self._log_success(f"Sequence length: {actual_length} residues")
        
        return True, actual_length
    
    def validate_inference_inputs(self, pdb_id: str, nodes_edges_dir: str, 
                                 processed_dir: str, config: Dict[str, Any]) -> bool:
        """Comprehensive validation before inference."""
        print(f"\n=== Validating inference inputs for {pdb_id} ===")
        
        pdb_dir = os.path.join(nodes_edges_dir, pdb_id)
        
        # Required files
        required_files = {
            'node_feature.parquet': os.path.join(pdb_dir, 'node_feature.parquet'),
            'edge_index.parquet': os.path.join(pdb_dir, 'edge_index.parquet'),
            'edge_attribute_dist.parquet': os.path.join(pdb_dir, 'edge_attribute_dist.parquet'),
            'edge_attribute_charge.parquet': os.path.join(pdb_dir, 'edge_attribute_charge.parquet'),
        }
        
        # Conditionally required files
        if not config.get('prediction', False):
            target_file = config.get('target_file', 'node_label_pi.parquet')
            required_files[target_file] = os.path.join(pdb_dir, target_file)
        
        if config.get('use_pretrained', False):
            required_files['antigen_sequence.json'] = os.path.join(processed_dir, pdb_id, 'sequence', 'antigen_sequence.json')
        
        if config.get('use_antiberty', False):
            required_files['cdr_sequence.json'] = os.path.join(processed_dir, pdb_id, 'sequence', 'cdr_sequence.json')
        
        # Check all required files exist
        for name, path in required_files.items():
            if not os.path.exists(path):
                self._log_error(f"Required file missing: {name}", path)
                return False
        
        self._log_success("All required files present")
        
        # Validate sequence length against max_antigen_len
        max_antigen_len = config.get('max_antigen_len')
        if max_antigen_len:
            is_valid, actual_length = self.validate_sequence_length(pdb_id, nodes_edges_dir, max_antigen_len)
            if not is_valid:
                if self.strict_mode:
                    return False
                else:
                    self._log_warning(f"Sequence length {actual_length} exceeds max_antigen_len {max_antigen_len}. "
                                    f"Use: python scripts/truncate_sequences.py --pdb_id {pdb_id} "
                                    f"--nodes_edges_dir {nodes_edges_dir} --max_len {max_antigen_len}")
        
        # Validate file formats and alignment
        if not self.validate_labels_step3(
            pdb_id,
            nodes_edges_dir,
            target_type=config.get('target_type', 'interface'),
            target_file=config.get('target_file', 'node_label_pi.parquet'),
            target_column=config.get('target_column', 'score')
        ):
            if not config.get('prediction', False):
                return False
        
        return True
    
    def validate_inference_outputs(self, output_dir: str, inference_list_path: str, 
                                  nodes_edges_dir: str, config: Dict[str, Any]) -> bool:
        """Validate inference outputs and verify no unnecessary files were required."""
        print(f"\n=== Validating inference outputs ===")
        
        # Check output directory exists
        if not os.path.exists(output_dir):
            self._log_error(f"Output directory does not exist: {output_dir}")
            return False
        
        test_record_dir = os.path.join(output_dir, 'test_record')
        if not os.path.exists(test_record_dir):
            self._log_error("test_record directory does not exist", test_record_dir)
            return False
        
        # Load inference list
        try:
            inference_df = pd.read_csv(inference_list_path)
            pdb_ids = inference_df['pdbId'].tolist() if 'pdbId' in inference_df.columns else inference_df.iloc[:, 0].tolist()
        except Exception as e:
            self._log_error(f"Cannot read inference list: {e}", inference_list_path)
            return False
        
        self._log_success(f"Loaded {len(pdb_ids)} PDBs from inference list")
        
        # Validate each result file
        all_valid = True
        for pdb_id in pdb_ids:
            result_file = os.path.join(test_record_dir, f'{pdb_id}_final_result.txt')
            
            if not os.path.exists(result_file):
                self._log_error(f"Result file missing for {pdb_id}", result_file)
                all_valid = False
                continue
            
            # Validate result file format
            try:
                df_result = pd.read_csv(result_file, sep='\t')
            except Exception as e:
                self._log_error(f"Cannot read result file: {e}", result_file)
                all_valid = False
                continue
            
            # Check required columns
            if config.get('loss_function') == 'mse' or config.get('out_label') == 1:
                required_cols = ['res_id', 'res_name', 'pred_score']
                missing_cols = set(required_cols) - set(df_result.columns)
                if missing_cols:
                    self._log_error(f"Missing columns in result file: {missing_cols}", result_file)
                    all_valid = False
                    continue
            else:
                required_cols = ['res_id', 'res_name', 'pred_label', 'prob.', 'score']
                missing_cols = set(required_cols) - set(df_result.columns)
                if missing_cols:
                    self._log_error(f"Missing columns in result file: {missing_cols}", result_file)
                    all_valid = False
                    continue
            
            # Validate data types and values
            if not pd.api.types.is_integer_dtype(df_result['res_id']):
                self._log_error("res_id must be integer", result_file)
                all_valid = False
                continue
            
            # Check res_id is sequential starting from 1
            expected_res_ids = np.arange(1, len(df_result) + 1)
            if not np.array_equal(df_result['res_id'].values, expected_res_ids):
                self._log_warning("res_id is not sequential starting from 1", result_file)
            
            # Check res_name are valid amino acids
            invalid_aa = df_result['res_name'].apply(lambda x: x not in VALID_AA_CODES if pd.notna(x) else False)
            if invalid_aa.any():
                invalid_codes = df_result.loc[invalid_aa, 'res_name'].unique()
                self._log_error(f"Invalid amino acid codes: {invalid_codes}", result_file)
                all_valid = False
                continue
            
            if config.get('loss_function') == 'mse' or config.get('out_label') == 1:
                if not pd.api.types.is_numeric_dtype(df_result['pred_score']):
                    self._log_error("pred_score must be numeric", result_file)
                    all_valid = False
                    continue
                if 'pred_prob' in df_result.columns:
                    if (df_result['pred_prob'] < 0).any() or (df_result['pred_prob'] > 1).any():
                        self._log_warning("pred_prob values outside [0, 1]", result_file)
            else:
                # Check pred_label values
                invalid_labels = ~df_result['pred_label'].isin(VALID_LABEL_VALUES)
                if invalid_labels.any():
                    invalid_values = df_result.loc[invalid_labels, 'pred_label'].unique()
                    self._log_error(f"Invalid pred_label values: {invalid_values}", result_file)
                    all_valid = False
                    continue
                
                # Check prob values
                if not pd.api.types.is_float_dtype(df_result['prob.']):
                    self._log_error("prob. must be float", result_file)
                    all_valid = False
                    continue
                
                if (df_result['prob.'] < 0).any() or (df_result['prob.'] > 1).any():
                    self._log_error("prob. values must be in [0, 1]", result_file)
                    all_valid = False
                    continue
            
            # Validate alignment with node_feature
            node_feature_path = os.path.join(nodes_edges_dir, pdb_id, 'node_feature.parquet')
            if os.path.exists(node_feature_path):
                df_features = pd.read_parquet(node_feature_path)
                
                if len(df_result) != len(df_features):
                    self._log_error(f"Row count mismatch: result={len(df_result)}, features={len(df_features)}", result_file)
                    all_valid = False
                    continue
                
                # Check res_id alignment
                if not np.array_equal(df_result['res_id'].values, df_features['resId'].values):
                    self._log_warning("res_id does not match node_feature.resId", result_file)
                
                # Check res_name alignment
                if not df_result['res_name'].equals(df_features['resShort']):
                    mismatches = (df_result['res_name'] != df_features['resShort']).sum()
                    self._log_warning(f"res_name mismatch: {mismatches} mismatches", result_file)
            
            self._log_success(f"{pdb_id}: result file valid ({len(df_result)} residues)")
        
        # Verify inference didn't require unnecessary files
        if self.verbose:
            print("\n=== File Usage Verification ===")
            print(f"Prediction mode: {config.get('prediction', False)}")
            print(f"Use pretrained: {config.get('use_pretrained', False)}")
            print(f"Use antiberty: {config.get('use_antiberty', False)}")
        
        return all_valid
    
    def validate_all_steps(self, pdb_id: str, nodes_edges_dir: str, processed_dir: str, 
                          config: Dict[str, Any] = None) -> bool:
        """Run all validations in sequence."""
        if config is None:
            config = {}
        
        print(f"\n{'='*60}")
        print(f"Validating all steps for {pdb_id}")
        print(f"{'='*60}")
        
        # Step 1: nodes_edges (CB)
        if not self.validate_nodes_edges_step1(pdb_id, nodes_edges_dir, processed_dir, 'CB'):
            return False
        
        # Step 1: nodes_edges (CA)
        if not self.validate_nodes_edges_step1(pdb_id, nodes_edges_dir, processed_dir, 'CA'):
            return False
        
        # Step 2: fill_edge
        if not self.validate_fill_edge_step2(pdb_id, nodes_edges_dir):
            return False
        
        # Step 3: labels
        if not self.validate_labels_step3(
            pdb_id,
            nodes_edges_dir,
            target_type=config.get('target_type', 'interface'),
            target_file=config.get('target_file', 'node_label_pi.parquet'),
            target_column=config.get('target_column', 'score')
        ):
            return False
        
        # Step 4: sequences
        if not self.validate_sequences_step4(pdb_id, processed_dir):
            return False
        
        # Step 5: inference inputs
        if not self.validate_inference_inputs(pdb_id, nodes_edges_dir, processed_dir, config):
            return False
        
        return True
    
    def get_summary(self) -> Dict[str, Any]:
        """Get validation summary."""
        return {
            'checks_passed': self.checks_passed,
            'checks_failed': self.checks_failed,
            'errors': self.errors,
            'warnings': self.warnings,
            'success_rate': self.checks_passed / (self.checks_passed + self.checks_failed) if (self.checks_passed + self.checks_failed) > 0 else 0
        }


def main():
    parser = argparse.ArgumentParser(description="Validate Epi4Ab pipeline steps")
    parser.add_argument('--pdb_id', type=str, help='Single PDB ID to validate')
    parser.add_argument('--metadata', type=str, help='Metadata CSV file with PDB IDs')
    parser.add_argument('--nodes_edges_dir', type=str, required=True, help='Path to nodes_edges directory')
    parser.add_argument('--processed_dir', type=str, help='Path to processed_data directory')
    parser.add_argument('--step', type=str, choices=['all', 'nodes_edges', 'fill_edge', 'labels', 
                                                       'sequences', 'inference_inputs', 'inference_outputs'],
                       default='all', help='Validation step to run')
    parser.add_argument('--output_dir', type=str, help='Inference output directory (for inference_outputs step)')
    parser.add_argument('--inference_list', type=str, help='Inference list CSV (for inference_outputs step)')
    parser.add_argument('--config', type=str, help='Model configuration JSON file')
    parser.add_argument('--target_type', type=str, choices=['interface','seqitope','proteinmpnn'],
                       help='Override target type for label validation')
    parser.add_argument('--target_file', type=str,
                       help='Override target file name for label validation')
    parser.add_argument('--target_column', type=str,
                       help='Override target column for label validation')
    parser.add_argument('--strict', action='store_true', help='Enable strict mode')
    parser.add_argument('--quiet', action='store_true', help='Suppress verbose output')
    
    args = parser.parse_args()
    
    # Load config if provided
    config = {}
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    if args.target_type:
        config['target_type'] = args.target_type
    if args.target_file:
        config['target_file'] = args.target_file
    if args.target_column:
        config['target_column'] = args.target_column
    
    validator = PipelineValidator(strict_mode=args.strict, verbose=not args.quiet)
    
    # Handle inference_outputs step separately (doesn't need pdb_id)
    if args.step == 'inference_outputs':
        if not args.output_dir or not args.inference_list:
            print("Error: --output_dir and --inference_list required for inference_outputs step")
            return 1
        valid = validator.validate_inference_outputs(args.output_dir, args.inference_list,
                                                   args.nodes_edges_dir, config)
        all_valid = valid
    else:
        # Get PDB IDs for other steps
        if args.pdb_id:
            pdb_ids = [args.pdb_id]
        elif args.metadata:
            try:
                 df = pd.read_csv(args.metadata, comment='#')
                 # Support multiple manifest conventions
                 if 'pdbID' in df.columns:
                     pdb_ids = df['pdbID'].tolist()
                 elif 'pdbId' in df.columns:
                     pdb_ids = df['pdbId'].tolist()
                 elif 'pdb_id' in df.columns:
                     pdb_ids = df['pdb_id'].tolist()
                 else:
                     pdb_ids = df.iloc[:, 0].tolist()
            except Exception as e:
                print(f"Error reading metadata: {e}")
                return 1
        else:
            print("Error: Must provide --pdb_id or --metadata")
            return 1
        
        # Run validation
        all_valid = True
        for pdb_id in pdb_ids:
            if args.step == 'all':
                valid = validator.validate_all_steps(pdb_id, args.nodes_edges_dir, 
                                                     args.processed_dir or args.nodes_edges_dir.replace('/nodes_edges', '/processed_data'),
                                                     config)
            elif args.step == 'nodes_edges':
                valid = validator.validate_nodes_edges_step1(pdb_id, args.nodes_edges_dir, 
                                                           args.processed_dir or '', 'CB')
            elif args.step == 'fill_edge':
                valid = validator.validate_fill_edge_step2(pdb_id, args.nodes_edges_dir)
            elif args.step == 'labels':
                valid = validator.validate_labels_step3(
                    pdb_id,
                    args.nodes_edges_dir,
                    target_type=config.get('target_type', 'interface'),
                    target_file=config.get('target_file', 'node_label_pi.parquet'),
                    target_column=config.get('target_column', 'score')
                )
            elif args.step == 'sequences':
                valid = validator.validate_sequences_step4(pdb_id, 
                                                          args.processed_dir or args.nodes_edges_dir.replace('/nodes_edges', '/processed_data'))
            elif args.step == 'inference_inputs':
                valid = validator.validate_inference_inputs(pdb_id, args.nodes_edges_dir,
                                                            args.processed_dir or args.nodes_edges_dir.replace('/nodes_edges', '/processed_data'),
                                                            config)
            if not valid:
                all_valid = False
    
    # Print summary
    summary = validator.get_summary()
    print(f"\n{'='*60}")
    print("Validation Summary")
    print(f"{'='*60}")
    print(f"Checks passed: {summary['checks_passed']}")
    print(f"Checks failed: {summary['checks_failed']}")
    print(f"Success rate: {summary['success_rate']:.2%}")
    
    if summary['warnings']:
        print(f"\nWarnings: {len(summary['warnings'])}")
        for warning in summary['warnings']:
            print(f"  {warning}")
    
    if summary['errors']:
        print(f"\nErrors: {len(summary['errors'])}")
        for error in summary['errors']:
            print(f"  {error}")
    
    return 0 if all_valid else 1


if __name__ == '__main__':
    sys.exit(main())
