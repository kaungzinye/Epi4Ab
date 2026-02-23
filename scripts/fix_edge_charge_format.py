#!/usr/bin/env python3
"""
Fix edge_attribute_charge.parquet format.

Some PDBs have the new format with ['source', 'target', 'charge'] columns,
but the code expects ['qi*qj'] column.

This script converts the format by:
1. Computing qi*qj from node charges if charge is not already qi*qj
2. Or renaming charge to qi*qj if it matches
3. Removing source/target columns (edge_index already has those)
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

NODES_EDGES_DIR = "/leonardo_scratch/fast/EUHPC_D29_035/epi4ab/upstream_preprocess/nodes_edges"

# 19 PDBs that were newly processed (excluding 1n8z_BAC which already has correct format)
PDB_LIST = [
    "1s78_DCA", "3be1_HLA", "3n85_HLA", "3wlw_CDA", "3wsq_HLA",
    "4lst_HLG", "4mwf_ABD", "4ywg_HLG", "5o4g_BAC",
    "6att_HLA", "6j6y_EFD", "6mug_HLG", "6nms_HLS", "6nmu_BAC",
    "6urm_DEC", "6wo5_HLE", "7l7r_DCG", "7lf7_ABM", "7mn8_DCB"
]


def fix_edge_charge(pdb_id: str) -> bool:
    """Fix edge_attribute_charge.parquet format for a PDB."""
    pdb_dir = os.path.join(NODES_EDGES_DIR, pdb_id)
    edge_charge_file = os.path.join(pdb_dir, "edge_attribute_charge.parquet")
    edge_index_file = os.path.join(pdb_dir, "edge_index.parquet")
    node_feature_file = os.path.join(pdb_dir, "node_feature.parquet")
    
    if not os.path.exists(edge_charge_file):
        print(f"  {pdb_id}: edge_attribute_charge.parquet not found")
        return False
    
    # Load files
    df_charge = pd.read_parquet(edge_charge_file)
    df_index = pd.read_parquet(edge_index_file)
    df_node = pd.read_parquet(node_feature_file)
    
    # Check current format
    if 'qi*qj' in df_charge.columns:
        print(f"  {pdb_id}: Already has qi*qj column, skipping")
        return True
    
    if 'charge' not in df_charge.columns:
        print(f"  {pdb_id}: ERROR: No 'charge' column found")
        return False
    
    # Compute qi*qj from node charges
    node_charges = df_node['charge'].values
    qi_qj = node_charges[df_index['source']] * node_charges[df_index['target']]
    
    # Create new dataframe with just qi*qj
    df_new = pd.DataFrame({'qi*qj': qi_qj})
    
    # Save backup
    backup_file = edge_charge_file + '.backup'
    if not os.path.exists(backup_file):
        df_charge.to_parquet(backup_file)
        print(f"  {pdb_id}: Created backup: {backup_file}")
    
    # Save new format
    df_new.to_parquet(edge_charge_file)
    print(f"  {pdb_id}: ✓ Fixed (computed qi*qj from node charges)")
    
    return True


def main():
    print("=" * 60)
    print("Fix Edge Charge Format")
    print("=" * 60)
    print()
    
    success_count = 0
    fail_count = 0
    
    for pdb_id in PDB_LIST:
        if fix_edge_charge(pdb_id):
            success_count += 1
        else:
            fail_count += 1
    
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Success: {success_count}")
    print(f"  Failed: {fail_count}")
    print()


if __name__ == "__main__":
    main()

