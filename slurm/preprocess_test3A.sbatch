#!/bin/bash
#SBATCH --job-name=epi4ab-preprocess-test3A
#SBATCH --account=AIFAC_F01_302
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/leonardo_work/AIFAC_F01_302/Epi4Ab/logs/%x-%j.out
#SBATCH --error=/leonardo_work/AIFAC_F01_302/Epi4Ab/logs/%x-%j.err

set -euo pipefail

module purge
module load python/3.11.7

PROJ=/leonardo_work/AIFAC_F01_302/Epi4Ab
cd "$PROJ"

# Use the virtualenv for dependencies
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Load environment variables
if [ -f ".env" ]; then
    set -a
    source .env
    set +a
fi

# Use metadata_complete.csv for 12 complete PDBs
METADATA_FILE="/leonardo_scratch/fast/AIFAC_F01_302/epi4ab/upstream_preprocess/metadata_complete.csv"

# Override DIRECTORY_METADATA to use complete PDBs only
export DIRECTORY_METADATA="$METADATA_FILE"

echo "=========================================="
echo "Epi4Ab Preprocessing - Test3A Dataset"
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo "Working directory: $(pwd)"
echo "METADATA: $DIRECTORY_METADATA"
echo "PROCESSED: $DIRECTORY_PROCESSED"
echo "NODES_EDGES: $DIRECTORY_NODES_EDGES"
echo "PYMOL: $PYMOL_EXECUTABLE"
echo "=========================================="

# Count PDBs
TOTAL_PDBS=$(tail -n +2 "$DIRECTORY_METADATA" | wc -l)
echo "Total PDBs to process: $TOTAL_PDBS"
echo ""

# Ensure output directories exist
mkdir -p "$DIRECTORY_PROCESSED"
mkdir -p "$DIRECTORY_NODES_EDGES"
mkdir -p logs

# Ensure venv/bin is in PATH for pdb2pqr and other tools
export PATH="$(pwd)/venv/bin:$PATH"

# Step 2: Main preprocessing (NO --download_pdb, files already downloaded)
echo "=========================================="
echo "Step 2: Main preprocessing (PDB2PQR, features)"
echo "Started: $(date)"
echo "=========================================="
./venv/bin/python preprocess/main.py "$DIRECTORY_METADATA" "$DIRECTORY_PROCESSED"

# Step 3a: Create nodes/edges (CB)
echo ""
echo "=========================================="
echo "Step 3a: Creating nodes/edges (CB)"
echo "Started: $(date)"
echo "=========================================="
./venv/bin/python preprocess/nodes_edges.py \
    "$DIRECTORY_METADATA" \
    "$DIRECTORY_PROCESSED" \
    "$DIRECTORY_NODES_EDGES" \
    "$PYMOL_EXECUTABLE" \
    'CB'

# Step 3b: Create nodes/edges (CA)
echo ""
echo "=========================================="
echo "Step 3b: Creating nodes/edges (CA)"
echo "Started: $(date)"
echo "=========================================="
./venv/bin/python preprocess/nodes_edges.py \
    "$DIRECTORY_METADATA" \
    "$DIRECTORY_PROCESSED" \
    "$DIRECTORY_NODES_EDGES" \
    "$PYMOL_EXECUTABLE" \
    'CA'

# Step 4: Fill edges
echo ""
echo "=========================================="
echo "Step 4: Filling edges"
echo "Started: $(date)"
echo "=========================================="
./venv/bin/python preprocess/fill_edge.py \
    "$DIRECTORY_METADATA" \
    "$DIRECTORY_NODES_EDGES"

# Verify completion
echo ""
echo "=========================================="
echo "Verification: Checking preprocessed data"
echo "=========================================="
for dir in "$DIRECTORY_NODES_EDGES"/*/; do
    pdb=$(basename "$dir")
    filecount=$(ls "$dir"/*.parquet 2>/dev/null | wc -l)
    if [ "$filecount" -ge 8 ]; then
        echo "  ✓ $pdb ($filecount files)"
    else
        echo "  ✗ $pdb (only $filecount files, expected 8+)"
    fi
done

echo ""
echo "=========================================="
echo "Preprocessing complete"
echo "Finished: $(date)"
echo "=========================================="
echo ""
echo "Next step: Submit inference job"
echo "  sbatch slurm/inference_test3A.sbatch"
echo "=========================================="

if command -v deactivate >/dev/null 2>&1; then
    deactivate
fi
