#!/usr/bin/env python3
"""
Download PDB files from RCSB PDB for HER2 structures.
"""

import argparse
import sys
from pathlib import Path
import urllib.request
import urllib.error

def download_pdb(pdb_id: str, output_dir: Path) -> bool:
    """
    Download a PDB file from RCSB PDB.
    
    Args:
        pdb_id: PDB identifier (e.g., '1N8Z')
        output_dir: Directory to save the PDB file
        
    Returns:
        True if successful, False otherwise
    """
    pdb_id = pdb_id.upper().strip()
    output_file = output_dir / f"{pdb_id}.pdb"
    
    # Skip if already exists
    if output_file.exists():
        print(f"✓ {pdb_id}.pdb already exists, skipping...")
        return True
    
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    
    try:
        print(f"Downloading {pdb_id}.pdb from RCSB PDB...")
        urllib.request.urlretrieve(url, output_file)
        
        # Verify file was downloaded (check size)
        if output_file.stat().st_size < 100:
            print(f"⚠️  Warning: {pdb_id}.pdb is very small ({output_file.stat().st_size} bytes), may be empty or error page")
            return False
        
        print(f"✅ Successfully downloaded {pdb_id}.pdb ({output_file.stat().st_size} bytes)")
        return True
        
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print(f"❌ {pdb_id}.pdb not found on RCSB PDB (404)")
        else:
            print(f"❌ HTTP error {e.code} downloading {pdb_id}.pdb: {e}")
        return False
    except Exception as e:
        print(f"❌ Error downloading {pdb_id}.pdb: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Download PDB files from RCSB PDB')
    parser.add_argument('--pdb_list', help='File containing PDB IDs (one per line or CSV)')
    parser.add_argument('--pdb_id', help='Single PDB ID to download')
    parser.add_argument('--output_dir', default='/leonardo_scratch/fast/AIFAC_F01_302/epi4ab/pdb_files',
                       help='Output directory for PDB files')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pdb_ids = []
    
    if args.pdb_id:
        pdb_ids = [args.pdb_id]
    elif args.pdb_list:
        pdb_list_file = Path(args.pdb_list)
        if not pdb_list_file.exists():
            print(f"Error: PDB list file not found: {pdb_list_file}")
            sys.exit(1)
        
        # Read PDB IDs from file (handle CSV or plain text)
        with open(pdb_list_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                # Handle CSV: take first column
                pdb_id = line.split(',')[0].strip()
                if pdb_id and len(pdb_id) == 4:
                    pdb_ids.append(pdb_id)
    else:
        print("Error: Please provide either --pdb_list or --pdb_id")
        sys.exit(1)
    
    if not pdb_ids:
        print("Error: No PDB IDs found")
        sys.exit(1)
    
    print(f"Downloading {len(pdb_ids)} PDB files to {output_dir}")
    print("=" * 70)
    
    successful = 0
    failed = 0
    
    for pdb_id in pdb_ids:
        if download_pdb(pdb_id, output_dir):
            successful += 1
        else:
            failed += 1
        print()
    
    print("=" * 70)
    print(f"Download complete: {successful} successful, {failed} failed")
    
    if failed > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()

