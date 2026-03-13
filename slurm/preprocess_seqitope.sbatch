#!/bin/bash
# Phase 2 only; expects node_label_seqitope.parquet under nodes_edges/<pdb_id>.
# Offline safety: export TRANSFORMERS_OFFLINE=1 and HF_HUB_OFFLINE=1 (optional HF_HOME).
#SBATCH --job-name=seqitope-preprocess
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

# Seqitope-specific paths
METADATA_FILE="/leonardo_scratch/fast/EUHPC_D29_035/epi4ab/seqitope_ground_truth/seqitope_metadata.csv"
PROCESSED_DIR="/leonardo_scratch/fast/EUHPC_D29_035/epi4ab/seqitope_ground_truth/processed_data"
NODES_EDGES_DIR="/leonardo_scratch/fast/EUHPC_D29_035/epi4ab/seqitope_ground_truth/nodes_edges"

# Override environment variables for Seqitope
export DIRECTORY_METADATA="$METADATA_FILE"
export DIRECTORY_PROCESSED="$PROCESSED_DIR"
export DIRECTORY_NODES_EDGES="$NODES_EDGES_DIR"

echo "=========================================="
echo "Epi4Ab Preprocessing - Seqitope Dataset"
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo "Working directory: $(pwd)"
echo "METADATA: $DIRECTORY_METADATA"
echo "PROCESSED: $DIRECTORY_PROCESSED"
echo "NODES_EDGES: $DIRECTORY_NODES_EDGES"
echo "PYMOL: $PYMOL_EXECUTABLE"
echo "=========================================="

# Check if metadata file exists
if [ ! -f "$METADATA_FILE" ]; then
    echo "WARNING: Metadata file not found: $METADATA_FILE"
    echo "Preprocessing may require metadata file. Continuing anyway..."
fi

# Count PDBs if metadata exists
if [ -f "$METADATA_FILE" ]; then
    TOTAL_PDBS=$(tail -n +2 "$METADATA_FILE" | wc -l)
    echo "Total PDBs to process: $TOTAL_PDBS"
else
    echo "Processing complexes: Ty1_RBD, mAb159_RBD, mAb311_RBD"
    TOTAL_PDBS=3
fi
echo ""

# Ensure output directories exist
mkdir -p "$DIRECTORY_PROCESSED"
mkdir -p "$DIRECTORY_NODES_EDGES"
mkdir -p logs

# Ensure venv/bin is in PATH for pdb2pqr and other tools
export PATH="$(pwd)/venv/bin:$PATH"

# Step 1: Main preprocessing (SKIPPED - may already be completed)
echo "=========================================="
echo "Step 1: Main preprocessing"
echo "Checking if pdb_profile.parquet files exist..."
echo "=========================================="

# Check if preprocessing is needed
NEED_PREPROCESS=false
for complex_id in Ty1_RBD mAb159_RBD mAb311_RBD; do
    profile_file="$DIRECTORY_PROCESSED/$complex_id/pdb_profile.parquet"
    if [ ! -f "$profile_file" ]; then
        echo "  Missing: $profile_file"
        NEED_PREPROCESS=true
    else
        echo "  Found: $profile_file"
    fi
done

if [ "$NEED_PREPROCESS" = true ] && [ -f "$METADATA_FILE" ]; then
    echo "Running main preprocessing..."
    ./venv/bin/python preprocess/main.py
else
    echo "Skipping main preprocessing (files exist or metadata missing)"
fi

# Step 2a: Create nodes/edges (CB)
echo ""
echo "=========================================="
echo "Step 2a: Creating nodes/edges (CB)"
echo "Started: $(date)"
echo "=========================================="

if [ -f "$METADATA_FILE" ]; then
    ./venv/bin/python preprocess/nodes_edges.py \
        "$DIRECTORY_METADATA" \
        "$DIRECTORY_PROCESSED" \
        "$DIRECTORY_NODES_EDGES" \
        "$PYMOL_EXECUTABLE" \
        'CB'
else
    echo "WARNING: Cannot run nodes_edges.py without metadata file"
    echo "Skipping nodes/edges generation. Ensure node_feature.parquet files exist."
fi

# Step 2b: Create nodes/edges (CA)
echo ""
echo "=========================================="
echo "Step 2b: Creating nodes/edges (CA)"
echo "Started: $(date)"
echo "=========================================="

if [ -f "$METADATA_FILE" ]; then
    ./venv/bin/python preprocess/nodes_edges.py \
        "$DIRECTORY_METADATA" \
        "$DIRECTORY_PROCESSED" \
        "$DIRECTORY_NODES_EDGES" \
        "$PYMOL_EXECUTABLE" \
        'CA'
else
    echo "WARNING: Cannot run nodes_edges.py without metadata file"
    echo "Skipping nodes/edges generation. Ensure edge files exist."
fi

# Step 3: Fill edges
echo ""
echo "=========================================="
echo "Step 3: Filling edges"
echo "Started: $(date)"
echo "=========================================="

if [ -f "$METADATA_FILE" ]; then
    ./venv/bin/python preprocess/fill_edge.py \
        "$DIRECTORY_METADATA" \
        "$DIRECTORY_NODES_EDGES"
else
    echo "WARNING: Cannot run fill_edge.py without metadata file"
    echo "Skipping edge filling. Ensure merged edge files exist."
fi

# Verify completion
echo ""
echo "=========================================="
echo "Verification: Checking preprocessed data"
echo "=========================================="
for complex_id in Ty1_RBD mAb159_RBD mAb311_RBD; do
    dir="$DIRECTORY_NODES_EDGES/$complex_id"
    if [ -d "$dir" ]; then
        filecount=$(ls "$dir"/node_feature.parquet "$dir"/edge_attribute_dist.parquet "$dir"/edge_attribute_charge.parquet "$dir"/edge_index.parquet 2>/dev/null | wc -l)
        if [ "$filecount" -ge 4 ]; then
            echo "  ✓ $complex_id ($filecount core files)"
        else
            echo "  ✗ $complex_id (only $filecount files, expected 4+)"
        fi
    else
        echo "  ✗ $complex_id (directory not found)"
    fi
done

echo ""
echo "=========================================="
echo "Preprocessing complete"
echo "Finished: $(date)"
echo "=========================================="
echo ""
echo "Next step: Generate labels and run inference"
echo "  sbatch slurm/generate_labels_seqitope.sbatch"
echo "  sbatch slurm/validate_seqitope.sbatch"
echo "  sbatch slurm/inference_seqitope.sbatch"
echo "=========================================="

if command -v deactivate >/dev/null 2>&1; then
    deactivate
fi

