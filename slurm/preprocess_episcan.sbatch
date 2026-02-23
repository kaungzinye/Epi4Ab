#!/bin/bash
#SBATCH --job-name=epi4ab-preprocess-episcan
#SBATCH --account=EUHPC_D29_035
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/leonardo_work/EUHPC_D29_035/Epi4Ab/logs/%x-%j.out
#SBATCH --error=/leonardo_work/EUHPC_D29_035/Epi4Ab/logs/%x-%j.err

set -euo pipefail

module purge
module load python/3.11.7

PROJ=/leonardo_work/EUHPC_D29_035/Epi4Ab
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

# EpiScan-specific paths
METADATA_FILE="/leonardo_scratch/fast/EUHPC_D29_035/epi4ab/episcan_preprocess/episcan_metadata.csv"
PROCESSED_DIR="/leonardo_scratch/fast/EUHPC_D29_035/epi4ab/episcan_preprocess/processed_data"
NODES_EDGES_DIR="/leonardo_scratch/fast/EUHPC_D29_035/epi4ab/episcan_preprocess/nodes_edges"

# Override environment variables for EpiScan
export DIRECTORY_METADATA="$METADATA_FILE"
export DIRECTORY_PROCESSED="$PROCESSED_DIR"
export DIRECTORY_NODES_EDGES="$NODES_EDGES_DIR"

echo "=========================================="
echo "Epi4Ab Preprocessing - EpiScan Dataset"
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

# Step 1: Main preprocessing (SKIPPED - already completed on login node)
echo "=========================================="
echo "Step 1: Main preprocessing (SKIPPED - already completed)"
echo "Skipping preprocess/main.py as pdb_profile.parquet files already exist"
echo "=========================================="

# Step 2a: Create nodes/edges (CB)
echo ""
echo "=========================================="
echo "Step 2a: Creating nodes/edges (CB)"
echo "Started: $(date)"
echo "=========================================="
./venv/bin/python preprocess/nodes_edges.py \
    "$DIRECTORY_METADATA" \
    "$DIRECTORY_PROCESSED" \
    "$DIRECTORY_NODES_EDGES" \
    "$PYMOL_EXECUTABLE" \
    'CB'

# Step 2b: Create nodes/edges (CA)
echo ""
echo "=========================================="
echo "Step 2b: Creating nodes/edges (CA)"
echo "Started: $(date)"
echo "=========================================="
./venv/bin/python preprocess/nodes_edges.py \
    "$DIRECTORY_METADATA" \
    "$DIRECTORY_PROCESSED" \
    "$DIRECTORY_NODES_EDGES" \
    "$PYMOL_EXECUTABLE" \
    'CA'

# Step 3: Fill edges
echo ""
echo "=========================================="
echo "Step 3: Filling edges"
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
    filecount=$(ls "$dir"/node_feature.parquet "$dir"/edge_attribute_dist.parquet "$dir"/edge_attribute_charge.parquet "$dir"/edge_index.parquet 2>/dev/null | wc -l)
    if [ "$filecount" -ge 4 ]; then
        echo "  ✓ $pdb ($filecount core files)"
    else
        echo "  ✗ $pdb (only $filecount files, expected 4+)"
    fi
done

echo ""
echo "=========================================="
echo "Preprocessing complete"
echo "Finished: $(date)"
echo "=========================================="
echo ""
echo "Next step: Generate labels and run inference"
echo "  sbatch slurm/generate_labels_episcan.sbatch"
echo "=========================================="

if command -v deactivate >/dev/null 2>&1; then
    deactivate
fi


