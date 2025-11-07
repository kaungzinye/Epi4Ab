#!/usr/bin/env python3
"""
Standalone script to extract CDR sequences from PDB files and add them to existing processed data.
This allows updating processed data with CDR sequences without reprocessing all features.
"""

import sys
import json
from pathlib import Path
import MDAnalysis as mda

# Add the Epi4Ab directory to the path
sys.path.append('/leonardo_work/AIFAC_F01_302/Epi4Ab')

from data_processing.epi4ab_pipeline import Epi4AbDataProcessor, AA_MAP

def extract_cdr_for_pdb(pdb_id: str, pdb_file: str, processed_dir: Path):
    """
    Extract CDR sequences from a PDB and add to processed data directory.
    
    Args:
        pdb_id: PDB ID (e.g., '1N8Z')
        pdb_file: Path to PDB file
        processed_dir: Path to processed data directory (e.g., processed/1N8Z/)
    """
    print(f"\nProcessing {pdb_id}...")
    
    try:
        # Create a minimal processor just for CDR extraction
        # We only need the antibody chain detection and CDR extraction
        processor = Epi4AbDataProcessor(
            pdb_file=pdb_file,
            output_dir=str(processed_dir),  # Not used, but required
            antigen_chain='A',  # Default, will be auto-detected
            sequence_source='pdb'
        )
        
        # Extract CDR sequences
        cdr_sequences = processor.extract_cdr_sequences()
        
        # Save to sequence directory
        sequence_dir = processed_dir / 'sequence'
        sequence_dir.mkdir(exist_ok=True)
        
        cdr_file = sequence_dir / 'cdr_sequence.json'
        with open(cdr_file, 'w') as f:
            json.dump(cdr_sequences, f, indent=2)
        
        print(f"  ✓ Saved CDR sequences to {cdr_file}")
        print(f"    H1={len(cdr_sequences['H1_seq'])}, H2={len(cdr_sequences['H2_seq'])}, H3={len(cdr_sequences['H3_seq'])}")
        print(f"    L1={len(cdr_sequences['L1_seq'])}, L2={len(cdr_sequences['L2_seq'])}, L3={len(cdr_sequences['L3_seq'])}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error processing {pdb_id}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function to process all 12 HER2 structures."""
    
    pdb_dir = Path('/leonardo_scratch/fast/AIFAC_F01_302/epi4ab/pdb_files')
    processed_dir = Path('/leonardo_scratch/fast/AIFAC_F01_302/epi4ab/processed')
    
    # List of all 12 HER2 PDB IDs
    her2_pdbs = ['1N8Z', '1S78', '3N85', '4K5Y', '5F8B', '6B0J', '6B0K', '6B0L', '6B0N', '6B0O', '6B0P', '6B0Q']
    
    print("=" * 60)
    print("Extracting CDR sequences for all HER2 structures")
    print("=" * 60)
    
    success_count = 0
    failed = []
    
    for pdb_id in her2_pdbs:
        pdb_file = pdb_dir / f"{pdb_id}.pdb"
        processed_pdb_dir = processed_dir / pdb_id
        
        if not pdb_file.exists():
            print(f"\n⚠️  PDB file not found: {pdb_file}")
            failed.append(pdb_id)
            continue
            
        if not processed_pdb_dir.exists():
            print(f"\n⚠️  Processed directory not found: {processed_pdb_dir}")
            failed.append(pdb_id)
            continue
        
        if extract_cdr_for_pdb(pdb_id, str(pdb_file), processed_pdb_dir):
            success_count += 1
        else:
            failed.append(pdb_id)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"✅ Successfully processed: {success_count}/{len(her2_pdbs)}")
    if failed:
        print(f"❌ Failed: {', '.join(failed)}")
    print("=" * 60)


if __name__ == '__main__':
    main()

