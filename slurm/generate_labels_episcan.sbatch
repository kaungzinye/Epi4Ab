#!/bin/bash
#SBATCH --job-name=epi4ab-gen-labels-episcan
#SBATCH --account=EUHPC_D29_035
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=02:00:00
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

# Activate venv
if [ -d "venv" ]; then
    source venv/bin/activate
fi

echo "=========================================="
echo "Epi4Ab Label Generation - EpiScan Dataset"
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo "Working directory: $(pwd)"
echo "=========================================="

# Directories
BASE_DIR="/leonardo_scratch/fast/EUHPC_D29_035/epi4ab/episcan_preprocess"
NODES_EDGES_DIR="$BASE_DIR/nodes_edges"

echo "BASE_DIR: $BASE_DIR"
echo "NODES_EDGES_DIR: $NODES_EDGES_DIR"
echo ""

# Run the CIPS label generation script
# The script already contains the logic for all 3 complexes
cd "$BASE_DIR"
python generate_cips_labels.py

echo "=========================================="
echo "Label generation complete"
echo "Finished: $(date)"
echo "=========================================="

# Verify results
echo ""
echo "Verification: Checking generated labels"
echo "=========================================="
for pdb_id in "${!COMPLEXES[@]}"; do
    label_file="$NODES_EDGES_DIR/$pdb_id/node_label_pi.parquet"
    
    if [ -f "$label_file" ]; then
        echo "  ✓ $pdb_id: node_label_pi.parquet exists"
    else
        echo "  ✗ $pdb_id: MISSING node_label_pi.parquet"
    fi
done

if command -v deactivate >/dev/null 2>&1; then
    deactivate
fi

