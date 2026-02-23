#!/usr/bin/env python3
"""
Truncate sequences and node features to fit within model's max_antigen_len.

This script truncates:
1. node_feature.parquet (keeps first N residues)
2. node_label_pi.parquet (keeps first N residues)
3. edge_index.parquet (removes edges involving truncated residues)
4. edge_attribute files (removed corresponding edges)
5. antigen_sequence.json (truncates sequence)
"""

import pandas as pd
import numpy as np
import json
import os
import sys
import argparse
from pathlib import Path
from typing import Optional


def truncate_node_features(pdb_id: str, nodes_edges_dir: str, max_len: int, backup: bool = True):
    """Truncate node_feature.parquet to max_len residues."""
    node_file = os.path.join(nodes_edges_dir, pdb_id, 'node_feature.parquet')
    
    if not os.path.exists(node_file):
        print(f"Error: {node_file} not found")
        return False
    
    df = pd.read_parquet(node_file)
    original_len = len(df)
    
    if original_len <= max_len:
        print(f"  {pdb_id}: No truncation needed ({original_len} <= {max_len})")
        return True
    
    # Backup original
    if backup:
        backup_file = node_file + '.backup'
        df.to_parquet(backup_file)
        print(f"  Backed up to {backup_file}")
    
    # Truncate
    df_truncated = df.head(max_len).copy()
    df_truncated.to_parquet(node_file)
    
    print(f"  {pdb_id}: Truncated node_feature from {original_len} to {max_len} residues")
    return True


def truncate_labels(pdb_id: str, nodes_edges_dir: str, max_len: int, backup: bool = True):
    """Truncate node_label_pi.parquet to max_len residues."""
    label_file = os.path.join(nodes_edges_dir, pdb_id, 'node_label_pi.parquet')
    
    if not os.path.exists(label_file):
        print(f"  Warning: {label_file} not found, skipping")
        return True
    
    df = pd.read_parquet(label_file)
    original_len = len(df)
    
    if original_len <= max_len:
        return True
    
    # Backup original
    if backup:
        backup_file = label_file + '.backup'
        df.to_parquet(backup_file)
    
    # Truncate
    df_truncated = df.head(max_len).copy()
    df_truncated.to_parquet(label_file)
    
    print(f"  {pdb_id}: Truncated labels from {original_len} to {max_len} residues")
    return True


def truncate_edges(pdb_id: str, nodes_edges_dir: str, max_len: int, backup: bool = True):
    """Truncate edge files to only include edges within first max_len nodes."""
    edge_index_file = os.path.join(nodes_edges_dir, pdb_id, 'edge_index.parquet')
    edge_dist_file = os.path.join(nodes_edges_dir, pdb_id, 'edge_attribute_dist.parquet')
    edge_charge_file = os.path.join(nodes_edges_dir, pdb_id, 'edge_attribute_charge.parquet')
    
    if not os.path.exists(edge_index_file):
        print(f"Error: {edge_index_file} not found")
        return False
    
    # Load edge files
    df_index = pd.read_parquet(edge_index_file)
    df_dist = pd.read_parquet(edge_dist_file) if os.path.exists(edge_dist_file) else None
    df_charge = pd.read_parquet(edge_charge_file) if os.path.exists(edge_charge_file) else None
    
    original_edge_count = len(df_index)
    
    # Filter edges: keep only edges where both source and target are < max_len
    mask = (df_index['source'] < max_len) & (df_index['target'] < max_len)
    mask_indices = df_index.index[mask]
    
    df_index_truncated = df_index.loc[mask_indices].copy().reset_index(drop=True)
    
    if df_dist is not None:
        df_dist_truncated = df_dist.loc[mask_indices].copy().reset_index(drop=True)
    if df_charge is not None:
        df_charge_truncated = df_charge.loc[mask_indices].copy().reset_index(drop=True)
    
    # Backup originals
    if backup:
        if os.path.exists(edge_index_file):
            pd.read_parquet(edge_index_file).to_parquet(edge_index_file + '.backup')
        if df_dist is not None and os.path.exists(edge_dist_file):
            pd.read_parquet(edge_dist_file).to_parquet(edge_dist_file + '.backup')
        if df_charge is not None and os.path.exists(edge_charge_file):
            pd.read_parquet(edge_charge_file).to_parquet(edge_charge_file + '.backup')
    
    # Save truncated files
    df_index_truncated.to_parquet(edge_index_file)
    if df_dist is not None:
        df_dist_truncated.to_parquet(edge_dist_file)
    if df_charge is not None:
        df_charge_truncated.to_parquet(edge_charge_file)
    
    print(f"  {pdb_id}: Truncated edges from {original_edge_count} to {len(df_index_truncated)} edges")
    return True


def truncate_sequence(pdb_id: str, processed_dir: str, max_len: int, backup: bool = True):
    """Truncate antigen_sequence.json to max_len residues."""
    seq_file = os.path.join(processed_dir, pdb_id, 'sequence', 'antigen_sequence.json')
    
    if not os.path.exists(seq_file):
        print(f"  Warning: {seq_file} not found, skipping")
        return True
    
    # Load sequence
    with open(seq_file, 'r') as f:
        data = json.load(f)
    
    sequence = data['pdb_sequence'].replace('gap', '').replace('x', '')
    original_len = len(sequence)
    
    if original_len <= max_len:
        return True
    
    # Backup original
    if backup:
        backup_file = seq_file + '.backup'
        with open(backup_file, 'w') as f:
            json.dump(data, f)
    
    # Truncate sequence
    truncated_seq = sequence[:max_len]
    data['pdb_sequence'] = truncated_seq
    
    # Save truncated sequence
    with open(seq_file, 'w') as f:
        json.dump(data, f)
    
    print(f"  {pdb_id}: Truncated sequence from {original_len} to {max_len} residues")
    return True


def truncate_pdb(pdb_id: str, nodes_edges_dir: str, processed_dir: str, 
                 max_len: int, backup: bool = True):
    """Truncate all files for a PDB to max_len."""
    print(f"\nTruncating {pdb_id} to {max_len} residues...")
    
    success = True
    success &= truncate_node_features(pdb_id, nodes_edges_dir, max_len, backup)
    success &= truncate_labels(pdb_id, nodes_edges_dir, max_len, backup)
    success &= truncate_edges(pdb_id, nodes_edges_dir, max_len, backup)
    success &= truncate_sequence(pdb_id, processed_dir, max_len, backup)
    
    return success


def get_max_antigen_len(model_dir: str) -> Optional[int]:
    """Extract max_antigen_len from model log.md."""
    log_file = os.path.join(model_dir, 'log.md')
    
    if not os.path.exists(log_file):
        return None
    
    import re
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Look for max_antigen_len
    match = re.search(r'max_antigen_len[:\s=]+(\d+)', content, re.IGNORECASE)
    if match:
        return int(match.group(1))
    
    # Try alternative patterns
    match = re.search(r'max.*antigen.*len[:\s=]+(\d+)', content, re.IGNORECASE)
    if match:
        return int(match.group(1))
    
    return None


def main():
    parser = argparse.ArgumentParser(description="Truncate sequences to fit model's max_antigen_len")
    parser.add_argument('--pdb_id', type=str, help='Single PDB ID to truncate')
    parser.add_argument('--metadata', type=str, help='Metadata CSV with PDB IDs')
    parser.add_argument('--nodes_edges_dir', type=str, required=True, help='Path to nodes_edges directory')
    parser.add_argument('--processed_dir', type=str, help='Path to processed_data directory')
    parser.add_argument('--max_len', type=int, help='Maximum length (overrides model config)')
    parser.add_argument('--model_dir', type=str, help='Model directory to extract max_antigen_len from')
    parser.add_argument('--no-backup', action='store_true', help='Do not create backup files')
    
    args = parser.parse_args()
    
    # Get max_len
    max_len = args.max_len
    if max_len is None and args.model_dir:
        max_len = get_max_antigen_len(args.model_dir)
        if max_len:
            print(f"Found max_antigen_len from model: {max_len}")
        else:
            print("Error: Could not find max_antigen_len in model config. Use --max_len to specify.")
            return 1
    
    if max_len is None:
        print("Error: Must provide --max_len or --model_dir")
        return 1
    
    # Get PDB IDs
    if args.pdb_id:
        pdb_ids = [args.pdb_id]
    elif args.metadata:
        df = pd.read_csv(args.metadata)
        pdb_ids = df['pdbID'].tolist() if 'pdbID' in df.columns else df.iloc[:, 0].tolist()
    else:
        print("Error: Must provide --pdb_id or --metadata")
        return 1
    
    processed_dir = args.processed_dir or args.nodes_edges_dir.replace('/nodes_edges', '/processed_data')
    backup = not args.no_backup
    
    # Truncate each PDB
    all_success = True
    for pdb_id in pdb_ids:
        success = truncate_pdb(pdb_id, args.nodes_edges_dir, processed_dir, max_len, backup)
        all_success &= success
    
    if all_success:
        print("\n✓ Truncation complete!")
        return 0
    else:
        print("\n✗ Some truncations failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())

