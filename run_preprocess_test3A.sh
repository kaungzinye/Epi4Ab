#!/bin/bash

# ============================================================
# WARNING: Direct execution without SLURM is NOT recommended
# ============================================================
echo "⚠️  WARNING: This script runs directly without SLURM"
echo "⚠️  For production use, submit via SLURM instead:"
echo "    sbatch slurm/preprocess_test3A.sbatch"
echo ""
echo "Press Ctrl+C to cancel or wait 5 seconds to continue..."
sleep 5
echo ""
# ============================================================

# Set environment variables
set -o allexport && source .env && set +o allexport

# Define input/output specific for this task
# We will use the main preprocessing output directories but a specific PDB list
export DIRECTORY_PDB_INFO="input/pdb_info_test3A.csv"
export DOWNLOAD_PDB=true # We need to download these

# Ensure venv/bin is in PATH for pdb2pqr and others
export PATH="$(pwd)/venv/bin:$PATH"

# 1. Create Metadata
echo "Step 1: creating metadata..."
./venv/bin/python preprocess/create_metadata.py $DIRECTORY_PDB_INFO $DIRECTORY_PREPROCESS_OUTPUT $MAFFT_EXECUTABLE

# 2. Main Preprocessing (PDB2PQR, MSMS, Features)
# This script handles downloading PDBs if --download_pdb is set
echo "Step 2: main preprocessing (downloading + features)..."
./venv/bin/python preprocess/main.py $DIRECTORY_METADATA $DIRECTORY_PROCESSED --download_pdb

# 3. Create Graph Nodes/Edges
echo "Step 3: creating nodes and edges..."
./venv/bin/python preprocess/nodes_edges.py $DIRECTORY_METADATA $DIRECTORY_PROCESSED $DIRECTORY_NODES_EDGES $PYMOL_EXECUTABLE 'CB'
./venv/bin/python preprocess/nodes_edges.py $DIRECTORY_METADATA $DIRECTORY_PROCESSED $DIRECTORY_NODES_EDGES $PYMOL_EXECUTABLE 'CA'

# 4. Fill Edges
echo "Step 4: filling edges..."
./venv/bin/python preprocess/fill_edge.py $DIRECTORY_METADATA $DIRECTORY_NODES_EDGES

echo "Preprocessing Complete!"
