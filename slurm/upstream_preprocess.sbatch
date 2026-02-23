#!/bin/bash
#SBATCH --job-name=epi4ab-upstream-preprocess
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

# Use the upstream virtualenv for dependencies
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Load environment variables for upstream preprocess (paths, executables, etc.)
if [ -f ".env" ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

echo "=========================================="
echo "Running upstream Epi4Ab preprocessing (run_preprocess.sh)"
echo "Branch: $(git rev-parse --abbrev-ref HEAD)"
echo "Working directory: $(pwd)"
echo "DIRECTORY_PDB_INFO=${DIRECTORY_PDB_INFO:-unset}"
echo "DIRECTORY_PREPROCESS_OUTPUT=${DIRECTORY_PREPROCESS_OUTPUT:-unset}"
echo "DIRECTORY_PROCESSED_DATA=${DIRECTORY_PROCESSED_DATA:-unset}"
echo "DIRECTORY_NODES_EDGES=${DIRECTORY_NODES_EDGES:-unset}"
echo "DIRECTORY_METADATA=${DIRECTORY_METADATA:-unset}"
echo "PYMOL_EXECUTABLE=${PYMOL_EXECUTABLE:-unset}"
echo "MAFFT_EXECUTABLE=${MAFFT_EXECUTABLE:-unset}"
echo "=========================================="

# Ensure output directories exist (fallback if env vars not set)
mkdir -p "${DIRECTORY_PREPROCESS_OUTPUT:-output_preprocess}"
mkdir -p logs

# Run the unchanged upstream preprocessing pipeline
bash run_preprocess.sh

echo "Upstream preprocessing finished."

if command -v deactivate >/dev/null 2>&1; then
    deactivate
fi

