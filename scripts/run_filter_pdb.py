#!/usr/bin/env python3
"""
Standalone script to run filter_pdb when PDB files are pre-downloaded.

This is needed because main.py only runs filter_pdb when DOWNLOAD_PDB=true,
but we need it to run even when .cif files are pre-downloaded.
"""

import os
import sys
import pandas as pd
from pathlib import Path

# Add project root to path
PROJ_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJ_ROOT))

from preprocess.scripts.extract_structure import filter_pdb
from preprocess.scripts.run_setup.logging import DataLogging

# Create minimal args object
class Args:
    def __init__(self):
        self.directory_metadata = os.environ.get('DIRECTORY_METADATA')
        self.directory_output = os.environ.get('DIRECTORY_PROCESSED')
        self.download_pdb = False

if __name__ == '__main__':
    args = Args()
    if not args.directory_metadata or not args.directory_output:
        print("ERROR: DIRECTORY_METADATA and DIRECTORY_PROCESSED must be set")
        sys.exit(1)
    
    logging = DataLogging(args)
    meta_df = pd.read_csv(logging.directory_metadata)
    print(f"Running filter_pdb for {len(meta_df)} PDBs...")
    filter_pdb(meta_df, logging)
    print("filter_pdb completed.")

