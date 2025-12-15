#!/bin/bash
# Download PDBs and create lig.pdb files (LOGIN NODE ONLY - requires internet)
# This script MUST be run on login node before submitting SLURM preprocessing job

set -euo pipefail

PROJ_DIR="/leonardo_work/AIFAC_F01_302/Epi4Ab"
cd "$PROJ_DIR"

# Load environment variables
set -o allexport
source .env
set +o allexport

# Check if metadata file exists
if [ ! -f "$DIRECTORY_METADATA" ]; then
    echo "Error: Metadata file not found: $DIRECTORY_METADATA"
    exit 1
fi

echo "=================================================="
echo "PDB Download Script (Login Node Only)"
echo "=================================================="
echo "Metadata: $DIRECTORY_METADATA"
echo "Output: $DIRECTORY_PROCESSED"
echo "$(date)"
echo "=================================================="

# Count total PDBs
TOTAL_PDBS=$(tail -n +2 "$DIRECTORY_METADATA" | wc -l)
echo "Total PDBs to process: $TOTAL_PDBS"

# Create output directories
mkdir -p "$DIRECTORY_PROCESSED"
mkdir -p logs

# Step 1: Download PDB files
echo ""
echo "Step 1: Downloading PDB .cif files..."
./venv/bin/python preprocess/scripts/download_pdb.py \
    --metadata "$DIRECTORY_METADATA" \
    --output_dir "$DIRECTORY_PROCESSED"

# Step 2: Run filter_pdb to create lig.pdb files
echo ""
echo "Step 2: Running filter_pdb to create lig.pdb files..."
./venv/bin/python scripts/run_filter_pdb.py

echo ""
echo "=================================================="
echo "Download complete at $(date)"
echo "=================================================="
echo ""
echo "Verify downloaded files:"
for dir in "$DIRECTORY_PROCESSED"/*/; do
    pdb=$(basename "$dir")
    if [ -f "$dir/lig.pdb" ]; then
        echo "  ✓ $pdb"
    else
        echo "  ✗ $pdb (missing lig.pdb)"
    fi
done

echo ""
echo "=================================================="
echo "Next steps:"
echo "  1. Verify all PDBs have lig.pdb files above"
echo "  2. Submit preprocessing SLURM job:"
echo "     sbatch slurm/preprocess_test3A.sbatch"
echo "=================================================="
