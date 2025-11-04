#!/usr/bin/env python3
"""Quick test script for label generation"""
import sys
sys.path.insert(0, '.')
from data_processing.epi4ab_pipeline import Epi4AbDataProcessor
from pathlib import Path
import tempfile
import pandas as pd

# Test on problematic PDBs
test_pdbs = ['3N85', '6B0N', '1N8Z']
pdb_dir = '/leonardo_scratch/fast/AIFAC_F01_302/epi4ab/pdb_files'

print('='*70)
print('TESTING LABEL GENERATION')
print('='*70)
print()

for pdb_id in test_pdbs:
    pdb_file = f'{pdb_dir}/{pdb_id}.pdb'
    output_dir = tempfile.mkdtemp(prefix=f'test_{pdb_id}_')
    
    try:
        if Path(pdb_file).exists():
            print(f'Testing {pdb_id}...')
            processor = Epi4AbDataProcessor(
                pdb_file=pdb_file,
                output_dir=output_dir,
                antigen_chain='A'
            )
            
            # Generate labels
            labels_df = processor.extract_epitope_labels()
            
            # Analyze label distribution
            total = len(labels_df)
            label_counts = labels_df['isInterface'].value_counts().sort_index()
            
            print(f'  Total residues: {total}')
            for label, count in label_counts.items():
                pct = (count/total)*100
                label_name = {0: 'Non-epitope', 1: 'CIPS (5Å)', 2: 'Ellipro/BepiPred'}.get(label, f'Unknown({label})')
                print(f'    Label {label} ({label_name}): {count:4d} ({pct:5.2f}%)')
            
            # Check if Label 1 exists
            has_label_1 = 1 in label_counts.index
            has_label_0 = 0 in label_counts.index
            
            if has_label_1:
                print(f'  ✓ Label 1 (CIPS) successfully generated!')
            else:
                print(f'  ⚠️  WARNING: No Label 1 (CIPS) generated')
            
            if not has_label_0:
                print(f'  ⚠️  WARNING: No Label 0 (background)')
            
            print()
        else:
            print(f'{pdb_id}: PDB file not found')
            print()
    except Exception as e:
        print(f'{pdb_id}: ERROR - {e}')
        import traceback
        traceback.print_exc()
        print()

