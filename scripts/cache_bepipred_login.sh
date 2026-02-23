#!/bin/bash
# =============================================================================
# Pre-cache BepiPred predictions on LOGIN NODE (requires internet)
# Run this BEFORE submitting the SLURM job for label generation
# =============================================================================

set -euo pipefail

PROJ=/leonardo_work/EUHPC_D29_035/Epi4Ab
cd "$PROJ"

echo "=========================================="
echo "BepiPred Cache Script (Login Node)"
echo "Started: $(date)"
echo "=========================================="
echo ""
echo "This script must run on LOGIN NODE (requires internet for BepiPred API)"
echo ""

# Activate venv
source venv/bin/activate

# Directories
PDB_DIR=/leonardo_scratch/fast/EUHPC_D29_035/epi4ab/pdb_complexes
CACHE_DIR=/leonardo_scratch/fast/EUHPC_D29_035/epi4ab/bepipred_cache
mkdir -p "$CACHE_DIR"

echo "PDB_DIR: $PDB_DIR"
echo "CACHE_DIR: $CACHE_DIR"
echo ""

# Check internet connectivity
echo "Checking internet connectivity..."
if curl -s --connect-timeout 5 https://api.iedb.org > /dev/null 2>&1; then
    echo "  ✓ Internet available - can fetch BepiPred predictions"
else
    echo "  ✗ No internet - are you on a LOGIN node?"
    echo "    Run this script on login01 or login02, not on compute nodes"
    exit 1
fi
echo ""

# List of PDBs to cache (skip 1n8z which already has labels)
declare -a PDBS=("1s78" "3be1" "3n85" "3wlw" "3wsq" "4lst" "4mwf" "4ywg" "5o4g" "6att" "6j6y" "6mug" "6nms" "6nmu" "6urm" "6wo5" "7l7r" "7lf7" "7mn8")

echo "Caching BepiPred predictions for ${#PDBS[@]} PDBs..."
echo ""

# Use a minimal Python script to extract sequences and cache BepiPred
python3 << 'PYTHON_SCRIPT'
import os
import sys
sys.path.insert(0, '/leonardo_work/EUHPC_D29_035/Epi4Ab')

from pathlib import Path
import MDAnalysis as mda

# Import the pipeline to use its BepiPred caching
from preprocess.scripts.label_generation.epi4ab_pipeline import Epi4AbDataProcessor

pdb_dir = Path('/leonardo_scratch/fast/EUHPC_D29_035/epi4ab/pdb_complexes')
cache_dir = Path('/leonardo_scratch/fast/EUHPC_D29_035/epi4ab/bepipred_cache')
output_dir = Path('/leonardo_scratch/fast/EUHPC_D29_035/epi4ab/upstream_preprocess/nodes_edges')

pdbs = ["1s78", "3be1", "3n85", "3wlw", "3wsq", "4lst", "4mwf", "4ywg", "5o4g", "6att", "6j6y", "6mug", "6nms", "6nmu", "6urm", "6wo5", "7l7r", "7lf7", "7mn8"]

for pdb in pdbs:
    pdb_file = pdb_dir / f"{pdb}.pdb"
    if not pdb_file.exists():
        print(f"  ✗ {pdb}: PDB file not found")
        continue
    
    cache_file = cache_dir / f"{pdb}_bepipred.json"
    if cache_file.exists():
        print(f"  ✓ {pdb}: Already cached")
        continue
    
    print(f"  → {pdb}: Caching BepiPred predictions...")
    try:
        # Create a temporary processor just to get the sequence and cache BepiPred
        processor = Epi4AbDataProcessor(
            str(pdb_file),
            str(output_dir / pdb),
            antigen_chain='A'
        )
        
        # Get antigen sequence
        sequence, _ = processor.extract_antigen_sequence()
        
        # Cache BepiPred predictions
        predictions = processor.get_bepipred_predictions(sequence, cache_dir=str(cache_dir))
        
        if predictions:
            print(f"  ✓ {pdb}: Cached {len(predictions)} residue predictions")
        else:
            print(f"  ⚠ {pdb}: No BepiPred predictions (will use Label 0 for potential epitopes)")
            
    except Exception as e:
        print(f"  ✗ {pdb}: Error - {e}")

print("\nBepiPred caching complete!")
PYTHON_SCRIPT

echo ""
echo "=========================================="
echo "Caching complete"
echo "Finished: $(date)"
echo "=========================================="
echo ""
echo "Cached files:"
ls -la "$CACHE_DIR"/*.json 2>/dev/null || echo "  (no cache files yet)"
echo ""
echo "Next step: Submit SLURM job for label generation:"
echo "  sbatch slurm/generate_labels.sbatch"

deactivate

