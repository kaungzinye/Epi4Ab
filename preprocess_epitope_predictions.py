#!/usr/bin/env python3
"""
Pre-processing script for BepiPred 3.0 predictions on login node.

This script should be run on the LOGIN NODE (has internet for ESM model download) 
BEFORE submitting processing jobs. It fetches BepiPred 3.0 predictions and caches 
them so compute nodes can use cached results without needing internet access.

Workflow:
1. Run this script on login node → Generates BepiPred predictions and caches them
2. Submit SLURM jobs on compute nodes → Uses cached predictions (no internet needed)

Usage:
    python preprocess_epitope_predictions.py --pdb_list pdb_list.txt --output_cache /path/to/cache/
    python preprocess_epitope_predictions.py --pdb_id 1N8Z --pdb_file /path/to/1N8Z.pdb --output_cache /path/to/cache/
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from data_processing.epi4ab_pipeline import Epi4AbDataProcessor


def preprocess_single_pdb(pdb_id: str, pdb_file: str, cache_dir: str, fasta_dir: str = None):
    """
    Pre-process a single PDB to fetch and cache BepiPred/Ellipro predictions.
    
    Args:
        pdb_id: PDB identifier
        pdb_file: Path to PDB file
        cache_dir: Directory to store cached predictions
        fasta_dir: Optional directory containing FASTA files
    """
    print(f"\n{'='*70}")
    print(f"Pre-processing {pdb_id}")
    print(f"{'='*70}")
    
    try:
        # Create temporary output directory (we only need predictions, not full processing)
        import tempfile
        temp_output = tempfile.mkdtemp(prefix=f'preprocess_{pdb_id}_')
        
        processor = Epi4AbDataProcessor(
            pdb_file=pdb_file,
            output_dir=temp_output,
            fasta_dir=fasta_dir,
            antigen_chain='A'
        )
        
        # Get antigen sequence for BepiPred
        try:
            antigen_sequence, source = processor.extract_antigen_sequence()
            print(f"  Antigen sequence ({source}): {len(antigen_sequence)} residues")
            
            # Fetch BepiPred predictions (will cache automatically)
            print(f"  Fetching BepiPred predictions...")
            bepipred_scores = processor.get_bepipred_predictions(antigen_sequence, cache_dir=cache_dir)
            if bepipred_scores:
                print(f"  ✓ BepiPred: {len(bepipred_scores)} predictions cached")
            else:
                print(f"  ⚠️  BepiPred: Failed to fetch predictions")
        except Exception as e:
            print(f"  ⚠️  BepiPred error: {e}")
        
        # Fetch Ellipro predictions (will cache automatically)
        # NOTE: Ellipro REQUIRES preprocessing on login node (web API only)
        # Compute nodes will use cached Ellipro predictions
        try:
            print(f"  Fetching Ellipro predictions (web API - login node only)...")
            ellipro_scores = processor.get_ellipro_predictions(pdb_file, cache_dir=cache_dir)
            if ellipro_scores:
                print(f"  ✓ Ellipro: {len(ellipro_scores)} predictions cached")
            else:
                print(f"  ⚠️  Ellipro: Failed to fetch predictions (may require internet)")
        except Exception as e:
            print(f"  ⚠️  Ellipro error: {e}")
        
        # Clean up temp directory
        import shutil
        shutil.rmtree(temp_output, ignore_errors=True)
        
        print(f"  ✓ Pre-processing complete for {pdb_id}")
        
    except Exception as e:
        print(f"  ✗ ERROR processing {pdb_id}: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description='Pre-process PDB files to fetch BepiPred 3.0 predictions on login node'
    )
    parser.add_argument('--pdb_list', type=str, help='Path to file containing list of PDB IDs (one per line)')
    parser.add_argument('--pdb_id', type=str, help='Single PDB ID to process')
    parser.add_argument('--pdb_file', type=str, help='Path to PDB file (required if --pdb_id)')
    parser.add_argument('--pdb_dir', type=str, default='/leonardo_scratch/fast/AIFAC_F01_302/epi4ab/pdb_files',
                       help='Directory containing PDB files (default: /leonardo_scratch/fast/AIFAC_F01_302/epi4ab/pdb_files)')
    parser.add_argument('--output_cache', type=str, required=True,
                       help='Directory to store cached predictions')
    parser.add_argument('--fasta_dir', type=str, default=None,
                       help='Optional directory containing FASTA files')
    
    args = parser.parse_args()
    
    # Create cache directory
    cache_path = Path(args.output_cache)
    cache_path.mkdir(parents=True, exist_ok=True)
    print(f"Cache directory: {cache_path}")
    
    # Process single PDB or list
    if args.pdb_id:
        if not args.pdb_file:
            pdb_file = Path(args.pdb_dir) / f"{args.pdb_id}.pdb"
        else:
            pdb_file = args.pdb_file
        
        if not Path(pdb_file).exists():
            print(f"ERROR: PDB file not found: {pdb_file}")
            sys.exit(1)
        
        preprocess_single_pdb(args.pdb_id, str(pdb_file), str(cache_path), args.fasta_dir)
    
    elif args.pdb_list:
        pdb_list_path = Path(args.pdb_list)
        if not pdb_list_path.exists():
            print(f"ERROR: PDB list file not found: {pdb_list_path}")
            sys.exit(1)
        
        # Read PDB list
        pdb_df = pd.read_csv(pdb_list_path, header=None, names=['pdb_id'])
        pdb_ids = pdb_df['pdb_id'].tolist()
        
        print(f"\nProcessing {len(pdb_ids)} PDB files...")
        
        for pdb_id in pdb_ids:
            pdb_file = Path(args.pdb_dir) / f"{pdb_id}.pdb"
            if pdb_file.exists():
                preprocess_single_pdb(pdb_id, str(pdb_file), str(cache_path), args.fasta_dir)
            else:
                print(f"  ⚠️  Skipping {pdb_id}: PDB file not found at {pdb_file}")
    
    else:
        parser.print_help()
        sys.exit(1)
    
    print(f"\n{'='*70}")
    print("Pre-processing complete!")
    print(f"Predictions cached in: {cache_path}")
    print(f"You can now run processing jobs on compute nodes.")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()

