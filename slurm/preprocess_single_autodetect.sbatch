#!/bin/bash
#SBATCH --job-name=epi4ab-preprocess-single-autodetect
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

if [ -d "venv" ]; then
    source venv/bin/activate
fi

if [ -f ".env" ]; then
    set -a
    source .env
    set +a
fi

METADATA_FILE="$PROJ/input/metadata_single_5o4g_BAC.csv"

OUT_BASE="/leonardo_scratch/fast/EUHPC_D29_035/epi4ab/autodetect_single_5o4g"
DIRECTORY_PROCESSED="$OUT_BASE/processed_data"
DIRECTORY_NODES_EDGES="$OUT_BASE/nodes_edges"

echo "=========================================="
echo "Epi4Ab Preprocessing - Single PDB (autodetect antigen chain)"
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo "METADATA: $METADATA_FILE"
echo "PROCESSED: $DIRECTORY_PROCESSED"
echo "NODES_EDGES: $DIRECTORY_NODES_EDGES"
echo "PYMOL: ${PYMOL_EXECUTABLE:-<unset>}"
echo "=========================================="

mkdir -p "$DIRECTORY_PROCESSED" "$DIRECTORY_NODES_EDGES" logs

# Seed with existing CIF so chain extraction can run offline
mkdir -p "$DIRECTORY_PROCESSED/5o4g_BAC"
cp -f /leonardo_scratch/fast/EUHPC_D29_035/epi4ab/upstream_preprocess/processed_data/5o4g_BAC/5o4g.cif \
    "$DIRECTORY_PROCESSED/5o4g_BAC/5o4g.cif"

export PATH="$(pwd)/venv/bin:$PATH"

echo "Step 1: preprocessing (lig.pdb + features)"
python preprocess/main.py "$METADATA_FILE" "$DIRECTORY_PROCESSED" --autodetect_antigen_chain

echo "Step 2: nodes/edges (CB)"
python preprocess/nodes_edges.py "$METADATA_FILE" "$DIRECTORY_PROCESSED" "$DIRECTORY_NODES_EDGES" "$PYMOL_EXECUTABLE" 'CB'

echo "Step 3: nodes/edges (CA)"
python preprocess/nodes_edges.py "$METADATA_FILE" "$DIRECTORY_PROCESSED" "$DIRECTORY_NODES_EDGES" "$PYMOL_EXECUTABLE" 'CA'

echo "Step 4: fill edges"
python preprocess/fill_edge.py "$METADATA_FILE" "$DIRECTORY_NODES_EDGES"

echo "DONE"

if command -v deactivate >/dev/null 2>&1; then
    deactivate
fi
