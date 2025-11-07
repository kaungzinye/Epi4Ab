#!/usr/bin/env python3
"""
Test script for label generation - Single antigen test
Run via SLURM: sbatch slurm/test_pipeline.sbatch
"""
import sys
import os
sys.path.insert(0, '.')
from data_processing.epi4ab_pipeline import Epi4AbDataProcessor
from pathlib import Path
import tempfile
import pandas as pd

# Test single antigen (can be run standalone or via SLURM)
test_pdb = '1N8Z'
pdb_dir = '/leonardo_scratch/fast/AIFAC_F01_302/epi4ab/pdb_files'
pdb_file = f'{pdb_dir}/{test_pdb}.pdb'

print('='*70)
print('LABEL GENERATION TEST: Single Antigen')
print('='*70)
print(f'Running on: {"SLURM/Compute Node" if os.environ.get("SLURM_JOB_ID") else "Interactive/Login Node"}')
print(f'Test PDB: {test_pdb}')
print('='*70)
print()

if not Path(pdb_file).exists():
    print(f'ERROR: PDB file not found: {pdb_file}')
    sys.exit(1)

try:
    output_dir = tempfile.mkdtemp(prefix=f'test_{test_pdb}_')
    print(f'Output directory: {output_dir}')
    print()
    
    print('Creating processor...')
    import time
    start_time = time.time()
    
    processor = Epi4AbDataProcessor(
        pdb_file=pdb_file,
        output_dir=output_dir,
        antigen_chain='A'
    )
    
    elapsed = time.time() - start_time
    print(f'✓ Processor created in {elapsed:.2f}s')
    print()
    
    print('Generating labels...')
    labels_df = processor.extract_epitope_labels()
    
    # Analyze label distribution
    total = len(labels_df)
    label_counts = labels_df['isInterface'].value_counts().sort_index()
    
    print(f'\nLabel Distribution:')
    print(f'  Total residues: {total}')
    for label, count in label_counts.items():
        pct = (count/total)*100
        label_name = {0: 'Non-epitope', 1: 'CIPS (5Å)', 2: 'Ellipro/BepiPred'}.get(label, f'Unknown({label})')
        print(f'    Label {label} ({label_name}): {count:4d} ({pct:5.2f}%)')
    
    # Check labels
    has_label_0 = 0 in label_counts.index
    has_label_1 = 1 in label_counts.index
    has_label_2 = 2 in label_counts.index
    
    print(f'\nVerification:')
    print(f'  Label 0 (Non-epitope): {"✓" if has_label_0 else "✗"}')
    print(f'  Label 1 (CIPS): {"✓" if has_label_1 else "✗"}')
    print(f'  Label 2 (BepiPred/Ellipro): {"✓" if has_label_2 else "✗"}')
    
    # Validate label distribution makes sense
    print(f'\nLabel Distribution Validation:')
    if has_label_2:
        label2_pct = label_counts[2] / total * 100
        if label2_pct > 50.0:
            print(f'  ⚠️  WARNING: Label 2 is {label2_pct:.1f}% - suspiciously high!')
            print(f'     Expected <30% typically. Threshold may be too low or bug in label extraction.')
        elif label2_pct > 30.0:
            print(f'  ⚠️  WARNING: Label 2 is {label2_pct:.1f}% - higher than expected (typically <30%)')
        else:
            print(f'  ✓ Label 2 percentage ({label2_pct:.1f}%) is reasonable')
    
    if not has_label_0:
        print(f'  ⚠️  WARNING: No Label 0 (non-epitopes) - all residues are predicted epitopes (unlikely!)')
    
    if has_label_1:
        print(f'\n✓ Label 1 (CIPS) successfully generated!')
    else:
        print(f'\n⚠️  WARNING: No Label 1 (CIPS) generated')
    
    if has_all_labels := (has_label_0 and has_label_1 and has_label_2):
        print(f'\n✅ All three labels present!')
    else:
        missing = [i for i in [0, 1, 2] if i not in label_counts.index]
        print(f'\n⚠️  Missing labels: {missing}')
    
    print()
    
except Exception as e:
    print(f'ERROR: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)

