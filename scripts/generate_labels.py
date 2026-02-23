#!/usr/bin/env python3
"""
Generate node_label_pi.parquet files for PDBs.

This script generates epitope labels using two modes:

1. Standard mode (CIPS/BepiPred):
   - Label 1 (CIPS): Antigen residues within 5Å of antibody atoms
   - Label 2 (BepiPred): IEDB BepiPred 2.0 predicted epitopes (from cache)
   - Requires combined PDB complex file

2. Raw ground truth mode:
   - Label 1: Residues specified in raw ground truth CSV file
   - Label 0: All other residues
   - Works with or without combined PDB complex (for experimental data)

IMPORTANT: Labels are aligned to existing node_feature.parquet files.
The script reads resIds from node_feature.parquet and generates labels
for exactly those residues, ensuring alignment for inference.

Usage (Standard mode):
    python scripts/generate_labels.py --pdb_file <pdb_file> --output_dir <output_dir> --cache_dir <cache_dir> --antigen_chain <chain_id> --pdb_id <pdb_id>

Usage (Raw ground truth mode):
    python scripts/generate_labels.py --raw_ground_truth <csv_file> --output_dir <output_dir> --pdb_id <complex_id> [--pdb_file <pdb_file>] [--antigen_chain <chain_id>]
"""

import os
import sys
import json
import argparse
import string
from pathlib import Path

import numpy as np
import pandas as pd
import MDAnalysis as mda
from MDAnalysis.analysis.distances import distance_array

# Ensure the project root (containing the scripts package) is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import residue mapping functions
try:
    from scripts.extract_residue_mapping import extract_residue_mapping, load_mapping, save_mapping
except ImportError as e:
    # Fallback if import fails – this should normally not happen when running
    # inside the Epi4Ab project, but we keep a graceful degradation path.
    print(f"  WARNING: could not import residue mapping utilities: {e}")

    def extract_residue_mapping(*args, **kwargs):
        raise ImportError("extract_residue_mapping module not available")

    def load_mapping(*args, **kwargs):
        return None

    def save_mapping(*args, **kwargs):
        return None

# Configuration
PDB_DIR = "/leonardo_scratch/fast/EUHPC_D29_035/epi4ab/pdb_complexes"
OUTPUT_DIR = "/leonardo_scratch/fast/EUHPC_D29_035/epi4ab/upstream_preprocess/nodes_edges"
CACHE_DIR = "/leonardo_scratch/fast/EUHPC_D29_035/epi4ab/bepipred_cache"

# 19 PDBs that need labels
PDB_LIST = [
    "1s78_DCA", "3be1_HLA", "3n85_HLA", "3wlw_CDA", "3wsq_HLA",
    "4lst_HLG", "4mwf_ABD", "4ywg_HLG", "5o4g_BAC",
    "6att_HLA", "6j6y_EFD", "6mug_HLG", "6nms_HLS", "6nmu_BAC",
    "6urm_DEC", "6wo5_HLE", "7l7r_DCG", "7lf7_ABM", "7mn8_DCB"
]


def load_node_feature_resids(output_dir: str) -> list:
    """
    Load resIds from existing node_feature.parquet.
    
    This ensures labels are generated for exactly the same residues
    as the node features, maintaining alignment for inference.
    
    Args:
        output_dir: Directory containing node_feature.parquet
        
    Returns:
        List of resIds, or None if file not found
    """
    node_file = os.path.join(output_dir, "node_feature.parquet")
    if not os.path.exists(node_file):
        print(f"  Warning: node_feature.parquet not found at {node_file}")
        return None
    
    try:
        df = pd.read_parquet(node_file)
        resids = df['resId'].tolist()
        print(f"  Loaded {len(resids)} resIds from node_feature.parquet")
        return resids
    except Exception as e:
        print(f"  Error loading node_feature.parquet: {e}")
        return None


def detect_chain_from_resids(universe: mda.Universe, target_resids: list) -> str:
    """
    Find which chain in the PDB matches the target resIds.
    
    Args:
        universe: MDAnalysis Universe
        target_resids: List of resIds to match
        
    Returns:
        Chain ID that matches, or None if no match found
    """
    target_set = set(target_resids)
    target_sorted = sorted(target_resids)
    
    best_match = None
    best_overlap = 0
    
    for chain_id in string.ascii_uppercase:
        try:
            chain_ca = universe.select_atoms(f'chainid {chain_id} and name CA')
            if len(chain_ca) == 0:
                continue
            
            chain_resids = sorted(set(chain_ca.residues.resnums))
            
            # Check for exact match
            if chain_resids == target_sorted:
                print(f"  ✓ Exact chain match: {chain_id} ({len(chain_resids)} residues)")
                return chain_id
            
            # Track best overlap for fallback
            overlap = len(target_set & set(chain_resids))
            if overlap > best_overlap:
                best_overlap = overlap
                best_match = chain_id
                
        except Exception:
            continue
    
    # If no exact match, use best overlap if it's significant
    if best_match and best_overlap > len(target_resids) * 0.8:
        print(f"  ⚠ Using best overlap chain: {best_match} ({best_overlap}/{len(target_resids)} resIds match)")
        return best_match
    
    return None


def detect_antibody_chains(universe: mda.Universe, antigen_chain: str) -> mda.core.groups.AtomGroup:
    """
    Detect antibody chains using multiple strategies.
    
    Args:
        universe: MDAnalysis Universe
        antigen_chain: Antigen chain ID
        
    Returns:
        AtomGroup containing antibody CA atoms
    """
    # Strategy 1: Try standard antibody chain IDs
    standard_ab_chains = ['B', 'C', 'H', 'L', 'D', 'E', 'F', 'G']
    for chain_id in standard_ab_chains:
        if chain_id == antigen_chain:
            continue
        try:
            ab = universe.select_atoms(f'chainid {chain_id} and name CA')
            if len(ab) > 0:
                print(f"  Found antibody chain(s) using standard ID: {chain_id} ({len(ab)} CA atoms)")
                return ab
        except Exception:
            continue
    
    # Strategy 2: Identify by size (antibody chains typically 90-150 residues)
    try:
        antigen_size = len(universe.select_atoms(f'chainid {antigen_chain} and name CA'))
        ab_chains = []
        for chain_id in string.ascii_uppercase:
            if chain_id == antigen_chain:
                continue
            chain_atoms = universe.select_atoms(f'chainid {chain_id} and name CA')
            size = len(chain_atoms)
            if 90 <= size <= 150:  # Typical antibody chain size
                ab_chains.append(chain_id)
        
        if ab_chains:
            selection_expr = ' or '.join([f'chainid {cid}' for cid in ab_chains])
            ab = universe.select_atoms(f'({selection_expr}) and name CA')
            if len(ab) > 0:
                print(f"  Found antibody chain(s) by size: {ab_chains} ({len(ab)} CA atoms)")
                return ab
    except Exception as e:
        print(f"  Warning: Size-based detection failed: {e}")
    
    # Strategy 3: All non-antigen chains
    try:
        ab = universe.select_atoms(f'name CA and (not chainid {antigen_chain})')
        if len(ab) > 0:
            print(f"  Warning: Using fallback (all non-{antigen_chain} chains, {len(ab)} CA atoms)")
            return ab
    except Exception:
        pass
    
    # Final fallback: empty
    print(f"  WARNING: Could not detect antibody chains")
    return universe.select_atoms('name CA and resnum -1')  # Empty


def load_raw_ground_truth(csv_path: str, complex_id: str) -> list:
    """
    Load raw ground truth epitope values from CSV file.
    
    CSV format can be either:
    1. complex_id,resId,amino_acid (e.g., "mAb159_RBD,384,P")
    2. mAb_id,antigen_id,resId,amino_acid (e.g., "mAb159,RBD,384,P")
    
    Args:
        csv_path: Path to CSV file
        complex_id: Complex identifier (e.g., "mAb159_RBD")
        
    Returns:
        List of (resId, amino_acid) tuples, or None if not found
    """
    if not os.path.exists(csv_path):
        print(f"  ERROR: Raw ground truth CSV not found: {csv_path}")
        return None
    
    try:
        df = pd.read_csv(csv_path)
        
        # Check which format we have
        if 'complex_id' in df.columns:
            # Format 1: complex_id,resId,amino_acid
            df_filtered = df[df['complex_id'] == complex_id].copy()
        elif 'mAb_id' in df.columns and 'antigen_id' in df.columns:
            # Format 2: mAb_id,antigen_id,resId,amino_acid
            # Extract mAb_id and antigen_id from complex_id
            parts = complex_id.split('_', 1)
            if len(parts) == 2:
                mab_id, antigen_id = parts
                df_filtered = df[(df['mAb_id'] == mab_id) & (df['antigen_id'] == antigen_id)].copy()
            else:
                print(f"  ERROR: Invalid complex_id format: {complex_id} (expected format: mAb159_RBD)")
                return None
        else:
            print(f"  ERROR: CSV must have either 'complex_id' column or 'mAb_id' and 'antigen_id' columns")
            return None
        
        if len(df_filtered) == 0:
            print(f"  WARNING: No entries found for complex_id: {complex_id}")
            return None
        
        # Extract resId and amino_acid
        if 'resId' not in df_filtered.columns or 'amino_acid' not in df_filtered.columns:
            print(f"  ERROR: CSV must have 'resId' and 'amino_acid' columns")
            return None
        
        raw_truth = [(int(row['resId']), str(row['amino_acid']).upper()) 
                     for _, row in df_filtered.iterrows()]
        
        print(f"  Loaded {len(raw_truth)} raw ground truth residues for {complex_id}")
        return raw_truth
        
    except Exception as e:
        print(f"  ERROR: Could not load raw ground truth CSV: {e}")
        return None


def load_bepipred_cache(pdb_id: str, cache_dir: str) -> dict:
    """
    Load cached BepiPred predictions.
    
    Args:
        pdb_id: Full PDB ID (e.g., "1s78_DCA")
        cache_dir: Cache directory path
        
    Returns:
        Dictionary mapping residue index (1-based) to BepiPred score, or None if not found
    """
    cache_file = os.path.join(cache_dir, f"{pdb_id}_bepipred.json")
    
    if not os.path.exists(cache_file):
        return None
    
    try:
        with open(cache_file, 'r') as f:
            data = json.load(f)
        
        # Convert scores dict to residue-indexed (1-based)
        scores = {}
        for pos_str, score in data.get('scores', {}).items():
            pos = int(pos_str)
            scores[pos + 1] = float(score)  # Convert 0-based to 1-based
        
        return scores
    except Exception as e:
        print(f"  Warning: Could not load BepiPred cache: {e}")
        return None


def generate_labels_from_raw_truth(output_dir: str, raw_truth_list: list, pdb_id: str, 
                                    pdb_file: str = None, antigen_chain: str = None) -> bool:
    """
    Generate node_label_pi.parquet from raw ground truth values.
    
    This function works with or without a combined PDB complex. If PDB file is provided,
    it validates amino acid codes match. If not, it generates labels directly from raw
    ground truth without PDB validation.
    
    IMPORTANT: Labels are aligned to existing node_feature.parquet.
    The script reads resIds from node_feature.parquet and generates labels
    for exactly those residues, ensuring alignment for inference.
    
    Args:
        output_dir: Output directory (nodes_edges/<pdb_id>/)
        raw_truth_list: List of (resId, amino_acid) tuples from raw ground truth
        pdb_id: Complex identifier (e.g., "mAb159_RBD")
        pdb_file: Optional path to PDB complex file (for validation)
        antigen_chain: Optional antigen chain ID (for PDB validation)
        
    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"Processing: {pdb_id} (Raw Ground Truth Mode)")
    print(f"{'='*60}")
    
    # Load resIds from node_feature.parquet (required)
    target_resids = load_node_feature_resids(output_dir)
    if target_resids is None:
        print(f"  ERROR: node_feature.parquet is required but not found")
        return False
    
    # Create mapping from raw ground truth
    raw_truth_dict = {resid: aa for resid, aa in raw_truth_list}
    raw_truth_resids = set(raw_truth_dict.keys())
    
    # Check if residue number mapping is needed
    # Mapping needed if: node features use sequential (1-N) but raw_truth has PDB numbers (>N)
    target_resids_set = set(target_resids)
    max_target_resid = max(target_resids) if target_resids else 0
    max_raw_truth_resid = max(raw_truth_resids) if raw_truth_resids else 0
    
    # Detect if mapping is needed: sequential numbering in node features but PDB numbers in raw_truth
    needs_mapping = (max_target_resid <= len(target_resids) and 
                     max_raw_truth_resid > max_target_resid and
                     len(raw_truth_resids & target_resids_set) == 0)
    
    mapping = None
    if needs_mapping:
        print(f"  Detected numbering mismatch:")
        print(f"    Node features: sequential (1-{max_target_resid})")
        print(f"    Raw ground truth: PDB numbers (max: {max_raw_truth_resid})")
        print(f"    Attempting to load/create residue mapping...")
        
        # Try to find structure file for mapping
        structure_file = None
        if pdb_file and os.path.exists(pdb_file):
            structure_file = pdb_file
        else:
            # Try to find CIF file in processed_data directory
            processed_dir = os.path.dirname(os.path.dirname(output_dir))
            if 'seqitope_ground_truth' in processed_dir:
                cif_file = os.path.join(processed_dir.replace('nodes_edges', 'processed_data'), 
                                       pdb_id, f"{pdb_id}.cif")
                if os.path.exists(cif_file):
                    structure_file = cif_file
        
        if structure_file:
            # Try to load existing mapping
            mapping_file = os.path.join(output_dir, "residue_mapping.json")
            mapping = load_mapping(mapping_file)
            
            if mapping is None:
                # Generate mapping from structure file
                try:
                    print(f"  Extracting mapping from: {structure_file}")
                    mapping = extract_residue_mapping(structure_file, antigen_chain)
                    save_mapping(mapping, mapping_file)
                    print(f"  ✓ Mapping created and saved")
                except Exception as e:
                    print(f"  WARNING: Could not extract mapping: {e}")
                    print(f"  Continuing without mapping (labels may be incorrect)")
                    mapping = None
            else:
                print(f"  ✓ Loaded existing mapping from: {mapping_file}")
        else:
            print(f"  WARNING: Structure file not found, cannot create mapping")
            print(f"  Labels will be generated without mapping (may be incorrect)")
    
    # Convert PDB resIds to sequential if mapping is available
    if mapping and needs_mapping:
        pdb_to_seq = mapping.get('pdb_to_sequential', {})
        converted_resids = set()
        unconverted_resids = []
        
        for pdb_resid in raw_truth_resids:
            if pdb_resid in pdb_to_seq:
                seq_resid = pdb_to_seq[pdb_resid]
                converted_resids.add(seq_resid)
                print(f"    Mapped: PDB {pdb_resid} -> Sequential {seq_resid}")
            else:
                unconverted_resids.append(pdb_resid)
        
        if unconverted_resids:
            print(f"  WARNING: {len(unconverted_resids)} raw ground truth resIds could not be mapped:")
            print(f"    Unmapped: {sorted(unconverted_resids)[:10]}{'...' if len(unconverted_resids) > 10 else ''}")
        
        # Use converted sequential resIds for label generation
        raw_truth_resids = converted_resids
        print(f"  Using {len(converted_resids)} mapped sequential resIds for label generation")
    
    # If PDB file is provided, validate amino acid codes
    if pdb_file and os.path.exists(pdb_file):
        try:
            universe = mda.Universe(pdb_file)
            
            # Auto-detect antigen chain if not provided
            if antigen_chain is None:
                detected_chain = detect_chain_from_resids(universe, target_resids)
                if detected_chain:
                    antigen_chain = detected_chain
                    print(f"  Auto-detected antigen chain: {antigen_chain}")
                else:
                    print(f"  WARNING: Could not detect antigen chain, skipping PDB validation")
                    pdb_file = None
            
            if pdb_file and antigen_chain:
                # Validate amino acid codes
                antigen_ca = universe.select_atoms(f'chainid {antigen_chain} and name CA')
                resnum_to_residue = {res.resnum: res for res in antigen_ca.residues}
                
                mismatches = []
                for resid, expected_aa in raw_truth_dict.items():
                    residue = resnum_to_residue.get(resid)
                    if residue:
                        actual_aa = residue.resname
                        # Convert 3-letter to 1-letter code
                        aa_map = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
                                 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
                                 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
                                 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'}
                        actual_aa_1letter = aa_map.get(actual_aa, actual_aa[0] if actual_aa else '?')
                        
                        if actual_aa_1letter.upper() != expected_aa.upper():
                            mismatches.append((resid, expected_aa, actual_aa_1letter))
                
                if mismatches:
                    print(f"  WARNING: {len(mismatches)} amino acid mismatches found:")
                    for resid, expected, actual in mismatches[:5]:  # Show first 5
                        print(f"    Residue {resid}: expected {expected}, found {actual}")
                    if len(mismatches) > 5:
                        print(f"    ... and {len(mismatches) - 5} more")
                else:
                    print(f"  ✓ All amino acid codes match PDB structure")
                    
        except Exception as e:
            print(f"  WARNING: Could not validate against PDB file: {e}")
            print(f"  Continuing without PDB validation...")
    else:
        if pdb_file:
            print(f"  WARNING: PDB file not found: {pdb_file}")
        print(f"  Generating labels without PDB validation (experimental data mode)")
    
    # Generate labels - use target_resids for alignment
    print(f"  Generating labels from raw ground truth...")
    labels = []
    
    for res_id in target_resids:
        # Check if this residue is in raw ground truth
        if res_id in raw_truth_resids:
            label = 1  # Epitope (raw ground truth)
        else:
            label = 0  # Non-epitope
        
        labels.append({
            'resId': res_id,
            'isInterface': label
        })
    
    # Create DataFrame
    df_labels = pd.DataFrame(labels)
    
    # Log distribution
    label_counts = df_labels['isInterface'].value_counts().to_dict()
    total = len(df_labels)
    print(f"  Label distribution:")
    print(f"    Label 0 (Non-epitope):     {label_counts.get(0, 0):4d} ({label_counts.get(0, 0)/total*100:5.2f}%)")
    print(f"    Label 1 (Raw Ground Truth): {label_counts.get(1, 0):4d} ({label_counts.get(1, 0)/total*100:5.2f}%)")
    
    # Check how many raw truth residues were found
    found_count = len([r for r in raw_truth_resids if r in target_resids])
    missing_count = len(raw_truth_resids) - found_count
    if missing_count > 0:
        missing_resids = sorted(raw_truth_resids - set(target_resids))
        print(f"  WARNING: {missing_count} raw ground truth residues not found in node_feature:")
        print(f"    Missing: {missing_resids[:10]}{'...' if len(missing_resids) > 10 else ''}")
    
    # Verify alignment
    if len(df_labels) == len(target_resids):
        print(f"  ✓ Alignment verified: {len(df_labels)} labels match {len(target_resids)} node features")
    else:
        print(f"  ⚠ Alignment warning: {len(df_labels)} labels vs {len(target_resids)} node features")
    
    # Save to parquet
    output_file = os.path.join(output_dir, "node_label_pi.parquet")
    df_labels.to_parquet(output_file, index=False)
    print(f"  ✓ Saved: {output_file}")
    
    return True


def generate_labels(pdb_file: str, output_dir: str, cache_dir: str, antigen_chain: str, pdb_id: str) -> bool:
    """
    Generate node_label_pi.parquet for a single PDB.
    
    IMPORTANT: Labels are aligned to existing node_feature.parquet.
    The script reads resIds from node_feature.parquet and generates labels
    for exactly those residues, ensuring alignment for inference.
    
    Args:
        pdb_file: Path to PDB complex file
        output_dir: Output directory (nodes_edges/<pdb_id>/)
        cache_dir: BepiPred cache directory
        antigen_chain: Antigen chain ID (used as fallback if no node_feature.parquet)
        pdb_id: Full PDB ID (e.g., "1s78_DCA")
        
    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"Processing: {pdb_id}")
    print(f"{'='*60}")
    
    # Load PDB
    try:
        universe = mda.Universe(pdb_file)
    except Exception as e:
        print(f"  ERROR: Could not load PDB: {e}")
        return False
    
    # ALIGNMENT: Load resIds from existing node_feature.parquet
    target_resids = load_node_feature_resids(output_dir)
    
    if target_resids is not None:
        # Auto-detect which chain matches the node_feature resIds
        detected_chain = detect_chain_from_resids(universe, target_resids)
        if detected_chain:
            if detected_chain != antigen_chain:
                print(f"  Note: Using detected chain {detected_chain} (not {antigen_chain} from PDB ID)")
            antigen_chain = detected_chain
        else:
            print(f"  Warning: Could not detect chain for resIds, using specified chain {antigen_chain}")
    
    # Select antigen chain
    try:
        antigen_ca = universe.select_atoms(f'chainid {antigen_chain} and name CA')
        if len(antigen_ca) == 0:
            print(f"  ERROR: No CA atoms found in antigen chain {antigen_chain}")
            return False
        print(f"  Antigen chain ({antigen_chain}): {len(antigen_ca)} CA atoms")
    except Exception as e:
        print(f"  ERROR: Could not select antigen chain: {e}")
        return False
    
    # Detect antibody chains (exclude detected antigen chain)
    antibody_ca = detect_antibody_chains(universe, antigen_chain)
    if len(antibody_ca) == 0:
        print(f"  WARNING: No antibody chains detected. Label 1 (CIPS) will be all zeros.")
    
    # Load BepiPred cache
    bepipred_scores = load_bepipred_cache(pdb_id, cache_dir)
    if bepipred_scores:
        print(f"  Loaded BepiPred cache: {len(bepipred_scores)} predictions")
    else:
        print(f"  Warning: No BepiPred cache found. Label 2 will be all zeros.")
    
    # Generate labels - use target_resids if available for alignment
    print(f"  Generating labels...")
    labels = []
    
    # Build a mapping from resnum to residue for fast lookup
    resnum_to_residue = {res.resnum: res for res in antigen_ca.residues}
    
    # Determine which resIds to process
    if target_resids is not None:
        # Use resIds from node_feature.parquet for alignment
        resids_to_process = target_resids
        print(f"  Using {len(resids_to_process)} resIds from node_feature.parquet for alignment")
    else:
        # Fallback: use all residues from antigen chain
        resids_to_process = [res.resnum for res in antigen_ca.residues]
        print(f"  Using {len(resids_to_process)} resIds from antigen chain {antigen_chain}")
    
    for res_id in resids_to_process:
        residue = resnum_to_residue.get(res_id)
        
        if residue is None:
            # Residue not found in this chain - assign label 0
            print(f"    Warning: resId {res_id} not found in chain {antigen_chain}, assigning label 0")
            labels.append({
                'resId': res_id,
                'isInterface': 0
            })
            continue
        
        ca_atoms = residue.atoms.select_atoms('name CA')
        if len(ca_atoms) == 0:
            labels.append({
                'resId': res_id,
                'isInterface': 0
            })
            continue
            
        ca_pos = ca_atoms.positions[0]
        
        # Calculate minimum distance to antibody
        min_distance = float('inf')
        if len(antibody_ca) > 0:
            dists = distance_array(
                ca_pos.reshape(1, -1),
                antibody_ca.positions
            )
            min_distance = np.min(dists)
        
        # Label priority: CIPS (1) > BepiPred (2) > Non-epitope (0)
        if min_distance <= 5.0:
            label = 1  # CIPS contact
        elif bepipred_scores:
            # BepiPred 2.0 threshold: score >= 0.5 (Assignment='E')
            bepipred_score = bepipred_scores.get(res_id, 0.0)
            if bepipred_score >= 0.5:
                label = 2  # BepiPred predicted epitope
            else:
                label = 0  # Non-epitope
        else:
            label = 0  # No predictions available
        
        labels.append({
            'resId': res_id,
            'isInterface': label
        })
    
    # Create DataFrame
    df_labels = pd.DataFrame(labels)
    
    # Log distribution
    label_counts = df_labels['isInterface'].value_counts().to_dict()
    total = len(df_labels)
    print(f"  Label distribution:")
    print(f"    Label 0 (Non-epitope):     {label_counts.get(0, 0):4d} ({label_counts.get(0, 0)/total*100:5.2f}%)")
    print(f"    Label 1 (CIPS):            {label_counts.get(1, 0):4d} ({label_counts.get(1, 0)/total*100:5.2f}%)")
    print(f"    Label 2 (BepiPred):         {label_counts.get(2, 0):4d} ({label_counts.get(2, 0)/total*100:5.2f}%)")
    
    # Verify alignment
    if target_resids is not None:
        if len(df_labels) == len(target_resids):
            print(f"  ✓ Alignment verified: {len(df_labels)} labels match {len(target_resids)} node features")
        else:
            print(f"  ⚠ Alignment warning: {len(df_labels)} labels vs {len(target_resids)} node features")
    
    # Save to parquet
    output_file = os.path.join(output_dir, "node_label_pi.parquet")
    df_labels.to_parquet(output_file, index=False)
    print(f"  ✓ Saved: {output_file}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Generate node_label_pi.parquet files")
    parser.add_argument('--pdb_file', type=str, help='Path to PDB complex file (optional when using --raw_ground_truth)')
    parser.add_argument('--output_dir', type=str, help='Output directory (nodes_edges/<pdb_id>/)')
    parser.add_argument('--cache_dir', type=str, default=CACHE_DIR, help='BepiPred cache directory')
    parser.add_argument('--antigen_chain', type=str, help='Antigen chain ID (optional when using --raw_ground_truth)')
    parser.add_argument('--pdb_id', type=str, help='Full PDB ID or complex identifier (e.g., 1s78_DCA or mAb159_RBD)')
    parser.add_argument('--raw_ground_truth', type=str, help='Path to CSV file with raw ground truth values')
    parser.add_argument('--all', action='store_true', help='Process all 19 PDBs')
    
    args = parser.parse_args()
    
    # Check if raw ground truth mode
    if args.raw_ground_truth:
        if not args.output_dir or not args.pdb_id:
            print("ERROR: --output_dir and --pdb_id are required when using --raw_ground_truth")
            parser.print_help()
            sys.exit(1)
        
        # Load raw ground truth
        raw_truth_list = load_raw_ground_truth(args.raw_ground_truth, args.pdb_id)
        if raw_truth_list is None:
            print(f"ERROR: Could not load raw ground truth for {args.pdb_id}")
            sys.exit(1)
        
        # Generate labels from raw ground truth
        os.makedirs(args.output_dir, exist_ok=True)
        if not generate_labels_from_raw_truth(
            args.output_dir, 
            raw_truth_list, 
            args.pdb_id,
            pdb_file=args.pdb_file,
            antigen_chain=args.antigen_chain
        ):
            sys.exit(1)
        
        return
    
    # Original mode (CIPS/BepiPred)
    if args.all:
        # Process all PDBs
        success_count = 0
        fail_count = 0
        
        for pdb_id_full in PDB_LIST:
            pdb_id_base = pdb_id_full.split('_')[0]  # e.g., "1s78" from "1s78_DCA"
            antigen_chain = pdb_id_full[-1]  # Last character is typically antigen chain
            
            pdb_file = os.path.join(PDB_DIR, f"{pdb_id_base}.pdb")
            output_dir = os.path.join(OUTPUT_DIR, pdb_id_full)
            
            os.makedirs(output_dir, exist_ok=True)
            
            if generate_labels(pdb_file, output_dir, args.cache_dir, antigen_chain, pdb_id_full):
                success_count += 1
            else:
                fail_count += 1
        
        print(f"\n{'='*60}")
        print(f"Summary: {success_count} succeeded, {fail_count} failed")
        print(f"{'='*60}")
        
    elif args.pdb_file and args.output_dir and args.antigen_chain:
        # Process single PDB
        pdb_id = args.pdb_id or os.path.basename(args.output_dir)
        if not generate_labels(args.pdb_file, args.output_dir, args.cache_dir, args.antigen_chain, pdb_id):
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

