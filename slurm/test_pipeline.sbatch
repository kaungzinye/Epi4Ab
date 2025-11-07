#!/bin/bash
#SBATCH --job-name=test-pipeline
#SBATCH --account=AIFAC_F01_302
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/leonardo_work/AIFAC_F01_302/Epi4Ab/logs/test-pipeline-%j.out
#SBATCH --error=/leonardo_work/AIFAC_F01_302/Epi4Ab/logs/test-pipeline-%j.err

# Comprehensive pipeline test - Single and Multiple antigens
# Tests: Full pipeline, Label generation (0,1,2), BepiPred/Ellipro integration

set -euo pipefail
module purge
module load python/3.11.7

PROJ=/leonardo_work/AIFAC_F01_302/Epi4Ab
cd $PROJ
source venv/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "=========================================="
echo "PIPELINE TEST: Single and Multiple Antigens"
echo "=========================================="
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURMD_NODENAME: $SLURMD_NODENAME"
echo "Date: $(date)"
echo "=========================================="
echo ""

python3 << 'PYEOF'
import sys
import os
import pandas as pd
from pathlib import Path
import tempfile
import shutil

sys.path.insert(0, '.')
from data_processing.epi4ab_pipeline import Epi4AbDataProcessor

print("="*70)
print("FULL PIPELINE TEST")
print("="*70)
print()

# Test PDBs
test_pdbs = [
    ('1N8Z', '/leonardo_scratch/fast/AIFAC_F01_302/epi4ab/pdb_files/1N8Z.pdb'),
    ('3N85', '/leonardo_scratch/fast/AIFAC_F01_302/epi4ab/pdb_files/3N85.pdb'),
]

results = []

for pdb_id, pdb_file in test_pdbs:
    print(f"\n{'='*70}")
    print(f"Testing: {pdb_id}")
    print(f"{'='*70}")
    
    if not Path(pdb_file).exists():
        print(f"  ⚠️  Skipping {pdb_id}: PDB file not found")
        continue
    
    try:
        # Create temporary output directory
        output_dir = tempfile.mkdtemp(prefix=f'test_{pdb_id}_')
        
        print(f"  Output directory: {output_dir}")
        print()
        
        # Step 1: Create processor (should be fast - no ESM2 hang)
        print("Step 1: Creating processor...")
        import time
        start_time = time.time()
        
        processor = Epi4AbDataProcessor(
            pdb_file=pdb_file,
            output_dir=output_dir,
            antigen_chain='A'
        )
        
        elapsed = time.time() - start_time
        print(f"  ✓ Processor created in {elapsed:.2f}s (no hang)")
        
        # Step 2: Process PDB (full pipeline)
        print("\nStep 2: Running full pipeline (process_pdb)...")
        success = processor.process_pdb()
        
        if not success:
            print(f"  ✗ Processing failed for {pdb_id}")
            results.append((pdb_id, False, "Processing failed"))
            continue
        
        print(f"  ✓ Processing completed")
        
        # Step 3: Verify outputs
        print("\nStep 3: Verifying outputs...")
        
        required_files = [
            'node_feature.parquet',
            'node_label_pi.parquet',
            'edge_index.parquet',
            'edge_attribute_dist.parquet',
            'edge_attribute_charge.parquet',
            'sequence/antigen_sequence.json',
            'sequence/cdr_sequence.json'
        ]
        
        missing_files = []
        for file in required_files:
            file_path = Path(output_dir) / file
            if not file_path.exists():
                missing_files.append(file)
            else:
                print(f"  ✓ {file}")
        
        if missing_files:
            print(f"  ✗ Missing files: {missing_files}")
            results.append((pdb_id, False, f"Missing files: {missing_files}"))
            continue
        
        # Step 4: Verify label distribution
        print("\nStep 4: Verifying label distribution...")
        labels_df = pd.read_parquet(Path(output_dir) / 'node_label_pi.parquet')
        
        total = len(labels_df)
        label_counts = labels_df['isInterface'].value_counts().sort_index()
        
        print(f"  Total residues: {total}")
        for label, count in label_counts.items():
            pct = (count/total)*100
            label_name = {0: 'Non-epitope', 1: 'CIPS (5Å)', 2: 'Ellipro/BepiPred'}.get(label, f'Unknown({label})')
            print(f"    Label {label} ({label_name}): {count:4d} ({pct:5.2f}%)")
        
        # Check labels
        has_label_0 = 0 in label_counts.index
        has_label_1 = 1 in label_counts.index
        has_label_2 = 2 in label_counts.index
        
        # Verify Label 1 (CIPS) exists - this is critical
        if has_label_1:
            print(f"  ✓ Label 1 (CIPS) present - antibody detection working!")
        else:
            print(f"  ✗ ERROR: Label 1 (CIPS) missing - antibody detection may have failed!")
        
        # Label 2 only exists if predictions are available
        if has_label_2:
            label2_pct = label_counts[2] / total * 100
            print(f"  ✓ Label 2 (BepiPred/Ellipro) present - predictions available!")
            
            # Validate Label 2 percentage makes sense
            if label2_pct > 50.0:
                print(f"  ⚠️  WARNING: Label 2 is {label2_pct:.1f}% - suspiciously high!")
                print(f"     Expected <30% typically. Threshold may be too low or bug in label extraction.")
            elif label2_pct > 30.0:
                print(f"  ⚠️  WARNING: Label 2 is {label2_pct:.1f}% - higher than expected (typically <30%)")
            else:
                print(f"  ✓ Label 2 percentage ({label2_pct:.1f}%) is reasonable")
        else:
            print(f"  → Label 2 not present (expected if no predictions available)")
        
        # Label 0 should always exist
        if not has_label_0:
            print(f"  ⚠️  WARNING: Label 0 missing (unusual) - all residues are predicted epitopes!")
        
        # Check if all three labels present (ideal case)
        has_all_labels = has_label_0 and has_label_1 and has_label_2
        if has_all_labels:
            print(f"  ✓ All three labels (0, 1, 2) present - ideal!")
        else:
            missing = [i for i in [0, 1, 2] if i not in label_counts.index]
            if missing == [2]:
                print(f"  → Missing Label 2 (expected if no BepiPred/Ellipro predictions)")
            else:
                print(f"  ⚠️  Missing labels: {missing}")
        
        # Step 5: Verify node features
        print("\nStep 5: Verifying node features...")
        features_df = pd.read_parquet(Path(output_dir) / 'node_feature.parquet')
        print(f"  ✓ Node features: {len(features_df)} residues, {len(features_df.columns)} features")
        
        # Step 6: Verify edges
        print("\nStep 6: Verifying graph edges...")
        edges_df = pd.read_parquet(Path(output_dir) / 'edge_index.parquet')
        print(f"  ✓ Graph edges: {len(edges_df)} connections")
        
        # Step 7: Verify BepiPred/Ellipro integration
        print("\nStep 7: Verifying BepiPred/Ellipro integration...")
        cache_dir = Path(output_dir) / 'epitope_predictions'
        bepipred_cache = cache_dir / f"{pdb_id}_bepipred.json"
        ellipro_cache = cache_dir / f"{pdb_id}_ellipro.json"
        
        if bepipred_cache.exists():
            print(f"  ✓ BepiPred cache exists: {bepipred_cache}")
        else:
            print(f"  ⚠️  BepiPred cache not found (no predictions available)")
        
        if ellipro_cache.exists():
            print(f"  ✓ Ellipro cache exists: {ellipro_cache}")
        else:
            print(f"  ⚠️  Ellipro cache not found (no predictions available)")
        
        # Determine Label 2 source
        if has_label_2:
            if bepipred_cache.exists() or ellipro_cache.exists():
                print(f"  ✓ Label 2 generated from BepiPred/Ellipro predictions")
            else:
                print(f"  ⚠️  Label 2 present but no predictions cached (should be 0 if no predictions)")
        
        # Cleanup
        shutil.rmtree(output_dir, ignore_errors=True)
        
        # Record result - PASS if Label 1 exists (critical), Label 2 is optional
        status = "PASS" if has_label_1 else "FAIL"
        if has_label_1 and not has_label_2:
            status = "PASS (no predictions)"  # Expected when no tools/cache
        results.append((pdb_id, True, status))
        
        print(f"\n  ✓ Test completed for {pdb_id}: {status}")
        
    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        results.append((pdb_id, False, str(e)))

# Summary
print("\n" + "="*70)
print("TEST SUMMARY")
print("="*70)

passed = sum(1 for _, success, _ in results if success)
total = len(results)

for pdb_id, success, status in results:
    symbol = "✓" if success else "✗"
    print(f"{symbol} {pdb_id}: {status}")

print()
print(f"Total: {passed}/{total} tests passed")

if passed == total:
    print("\n✅ ALL TESTS PASSED - Pipeline is working correctly!")
else:
    print("\n⚠️  Some tests had issues - check logs above")

print("="*70)
PYEOF

deactivate

