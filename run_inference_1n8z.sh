#!/bin/bash

# ============================================================
# WARNING: Direct execution without SLURM is NOT recommended
# ============================================================
echo "⚠️  WARNING: This script runs directly without SLURM"
echo "⚠️  For production use, submit via SLURM instead:"
echo "    sbatch slurm/inference_test3A.sbatch"
echo ""
echo "Press Ctrl+C to cancel or wait 5 seconds to continue..."
sleep 5
echo ""
# ============================================================

set -o allexport && source .env && set +o allexport
export PDB_LIST_OVERRIDE="inference_list_1n8z.csv"
export OMP_NUM_THREADS=1
./venv/bin/python epi_prediction.py $DIRECTORY_NODES_EDGES $DIRECTORY_PROCESSED_DATA \
    $PDB_LIST_OVERRIDE $FINAL_MODEL_FOLDER \
    $DIRECTORY_INFERENCE_OUTPUT
