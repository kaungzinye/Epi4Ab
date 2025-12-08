#!/bin/bash

# Set environment variables
set -o allexport && source .env && set +o allexport

DOWNLOAD_PDB=false

PARAMS=""
[ "$DOWNLOAD_PDB" = true ] && PARAMS+="--download_pdb " || PARAMS+=""


python preprocess/create_metadata.py $DIRECTORY_PDB_INFO $DIRECTORY_PREPROCESS_OUTPUT $MAFFT_EXECUTABLE

# If DOWNLOAD_PDB=false but .cif files exist, we still need to run filter_pdb
# to create lig.pdb files BEFORE main.py runs (main.py needs lig.pdb for pdb2pqr).
# This is a workaround for pre-downloaded PDB files.
if [ "$DOWNLOAD_PDB" = false ]; then
    echo "DOWNLOAD_PDB=false, running filter_pdb for pre-downloaded .cif files..."
    python scripts/run_filter_pdb.py
fi

python preprocess/main.py $DIRECTORY_METADATA $DIRECTORY_PROCESSED $PARAMS

python preprocess/nodes_edges.py $DIRECTORY_METADATA $DIRECTORY_PROCESSED $DIRECTORY_NODES_EDGES $PYMOL_EXECUTABLE 'CB'
python preprocess/nodes_edges.py $DIRECTORY_METADATA $DIRECTORY_PROCESSED $DIRECTORY_NODES_EDGES $PYMOL_EXECUTABLE 'CA'

python preprocess/fill_edge.py $DIRECTORY_METADATA $DIRECTORY_NODES_EDGES