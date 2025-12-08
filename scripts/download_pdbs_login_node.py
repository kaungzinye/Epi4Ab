#!/usr/bin/env python3
"""
Download PDB files on a login node (with internet access).

This script should be run on a login node (not compute node) because it requires
internet access to download PDB files from RCSB.

Usage:
    python scripts/download_pdbs_login_node.py

The script will:
1. Read pdb_info.csv from input/
2. Download PDB files to DIRECTORY_PROCESSED/{pdb_id}/{pdb}.cif
3. Create necessary directories
"""

import os
import sys
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from Bio.PDB.PDBList import PDBList

# Add project root to path
PROJ_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJ_ROOT))

def load_env_vars():
    """Load environment variables from .env file."""
    import re
    env_vars = {}
    env_file = PROJ_ROOT / '.env'
    
    if not env_file.exists():
        raise FileNotFoundError(f".env file not found at {env_file}")
    
    # First pass: collect all variables
    with open(env_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                env_vars[key] = value
    
    # Second pass: expand variables recursively
    max_iterations = 10
    for iteration in range(max_iterations):
        changed = False
        for key, value in list(env_vars.items()):
            # Expand ${VAR} references
            def expand_var(match):
                var_name = match.group(1)
                # Check env_vars first, then os.environ
                expanded = env_vars.get(var_name) or os.environ.get(var_name, match.group(0))
                return expanded
            
            new_value = re.sub(r'\$\{([^}]+)\}', expand_var, value)
            if new_value != value:
                env_vars[key] = new_value
                changed = True
        
        if not changed:
            break
    
    return env_vars

def download_pdb_files(pdb_info_path, output_base_dir):
    """
    Download PDB files from RCSB.
    
    Args:
        pdb_info_path: Path to pdb_info.csv
        output_base_dir: Base directory where processed data will be stored
    """
    # Read pdb_info.csv
    print(f"Reading PDB info from: {pdb_info_path}")
    pdb_df = pd.read_csv(pdb_info_path)
    print(f"Found {len(pdb_df)} PDB entries")
    
    # Initialize PDB downloader
    pdb_downloader = PDBList(verbose=False)
    
    # Track errors
    errors = []
    downloaded = []
    skipped = []
    
    # Download each PDB file
    for idx, row in tqdm(pdb_df.iterrows(), total=len(pdb_df), desc="Downloading PDBs"):
        pdb_id = row['pdbID']  # e.g., "1n8z_BAC"
        pdb_code = row['pdb']  # e.g., "1n8z"
        
        # Create directory for this PDB
        pdb_dir = os.path.join(output_base_dir, pdb_id)
        os.makedirs(pdb_dir, exist_ok=True)
        
        # Check if file already exists
        cif_file = os.path.join(pdb_dir, f'{pdb_code}.cif')
        if os.path.exists(cif_file):
            skipped.append(pdb_id)
            continue
        
        try:
            # Download PDB file as mmCIF format
            pdb_downloader.retrieve_pdb_file(
                pdb_code,
                pdir=pdb_dir,
                file_format='mmCif'
            )
            
            # Verify download
            if os.path.exists(cif_file):
                downloaded.append(pdb_id)
                print(f"✓ Downloaded {pdb_id} -> {cif_file}")
            else:
                errors.append(pdb_id)
                print(f"✗ Failed to download {pdb_id}: file not found after download")
                
        except Exception as e:
            errors.append(pdb_id)
            print(f"✗ Error downloading {pdb_id}: {e}")
    
    # Summary
    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)
    print(f"Total entries: {len(pdb_df)}")
    print(f"Successfully downloaded: {len(downloaded)}")
    print(f"Skipped (already exists): {len(skipped)}")
    print(f"Errors: {len(errors)}")
    
    if downloaded:
        print(f"\nDownloaded PDBs: {', '.join(downloaded[:10])}")
        if len(downloaded) > 10:
            print(f"... and {len(downloaded) - 10} more")
    
    if skipped:
        print(f"\nSkipped PDBs (already exist): {', '.join(skipped[:10])}")
        if len(skipped) > 10:
            print(f"... and {len(skipped) - 10} more")
    
    if errors:
        print(f"\n✗ Failed PDBs: {', '.join(errors)}")
        return False
    
    return True

def main():
    """Main function."""
    print("="*60)
    print("PDB Download Script (Login Node)")
    print("="*60)
    print()
    
    # Check if running on login node (heuristic: check hostname or environment)
    # Note: This is a simple check, adjust based on your HPC system
    hostname = os.environ.get('HOSTNAME', '')
    if 'compute' in hostname.lower() or 'node' in hostname.lower():
        print("WARNING: This script should be run on a login node, not a compute node!")
        print("Compute nodes typically don't have internet access.")
        response = input("Continue anyway? (yes/no): ")
        if response.lower() != 'yes':
            print("Aborted.")
            sys.exit(1)
    
    # Load environment variables
    try:
        env = load_env_vars()
    except Exception as e:
        print(f"Error loading .env file: {e}")
        sys.exit(1)
    
    # Get paths from environment
    pdb_info_path = env.get('DIRECTORY_PDB_INFO')
    processed_dir = env.get('DIRECTORY_PROCESSED')
    
    if not pdb_info_path:
        print("ERROR: DIRECTORY_PDB_INFO not set in .env file")
        sys.exit(1)
    
    if not processed_dir:
        print("ERROR: DIRECTORY_PROCESSED not set in .env file")
        sys.exit(1)
    
    # Expand paths
    pdb_info_path = os.path.expanduser(pdb_info_path)
    processed_dir = os.path.expanduser(processed_dir)
    
    # Check if pdb_info.csv exists
    if not os.path.exists(pdb_info_path):
        print(f"ERROR: PDB info file not found: {pdb_info_path}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(processed_dir, exist_ok=True)
    print(f"Output directory: {processed_dir}")
    print()
    
    # Download PDB files
    success = download_pdb_files(pdb_info_path, processed_dir)
    
    if success:
        print("\n✓ All PDB files downloaded successfully!")
        sys.exit(0)
    else:
        print("\n✗ Some PDB files failed to download. Check errors above.")
        sys.exit(1)

if __name__ == '__main__':
    main()

