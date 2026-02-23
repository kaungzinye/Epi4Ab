#!/usr/bin/env python3
"""
Extract sequences from parquet files for local BepiPred caching.

This script reads node_feature.parquet files and outputs sequences
in a format suitable for cache_bepipred_local.py

Usage:
    python extract_sequences_for_local.py > sequences.txt
    # Then copy sequences.txt to your local machine and run:
    # python cache_bepipred_local.py --sequences sequences.txt
"""

import os
import sys
import pandas as pd

# Configuration
NODES_EDGES_DIR = "/leonardo_scratch/fast/EUHPC_D29_035/epi4ab/upstream_preprocess/nodes_edges"

# 19 PDBs that need labels
PDB_LIST = [
    "1s78_DCA", "3be1_HLA", "3n85_HLA", "3wlw_CDA", "3wsq_HLA",
    "4lst_HLG", "4mwf_ABD", "4ywg_HLG", "5o4g_BAC",
    "6att_HLA", "6j6y_EFD", "6mug_HLG", "6nms_HLS", "6nmu_BAC",
    "6urm_DEC", "6wo5_HLE", "7l7r_DCG", "7lf7_ABM", "7mn8_DCB"
]


def extract_sequence(pdb_id: str) -> str:
    """Extract sequence from node_feature.parquet."""
    # Try full ID first, then base ID (e.g., 6wo5_HLE -> 6wo5)
    node_feature_file = os.path.join(NODES_EDGES_DIR, pdb_id, "node_feature.parquet")
    
    if not os.path.exists(node_feature_file):
        # Try base ID (extract before underscore)
        base_id = pdb_id.split('_')[0]
        node_feature_file = os.path.join(NODES_EDGES_DIR, base_id, "node_feature.parquet")
        
        if not os.path.exists(node_feature_file):
            print(f"# ERROR: {pdb_id} not found (tried {pdb_id} and {base_id})", file=sys.stderr)
            return None
    
    try:
        df = pd.read_parquet(node_feature_file)
        
        if 'resShort' not in df.columns:
            print(f"# ERROR: 'resShort' column not found in {pdb_id}", file=sys.stderr)
            return None
        
        sequence = ''.join(df['resShort'].tolist())
        return sequence
    except Exception as e:
        print(f"# ERROR: Failed to read {pdb_id}: {e}", file=sys.stderr)
        return None


def main():
    print("# BepiPred sequences extracted from parquet files")
    print("# Format: PDB_ID:SEQUENCE")
    print("# Copy this file to your local machine and run:")
    print("#   python cache_bepipred_local.py --sequences sequences.txt")
    print()
    
    for pdb_id in PDB_LIST:
        sequence = extract_sequence(pdb_id)
        if sequence:
            print(f"{pdb_id}:{sequence}")
        else:
            print(f"# {pdb_id}: FAILED", file=sys.stderr)


if __name__ == "__main__":
    main()

