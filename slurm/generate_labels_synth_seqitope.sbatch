#!/bin/bash
#SBATCH --job-name=epi4ab-synthlabels
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

# Phase 1.5 only; generates node_label_synthseq.parquet
# Offline safety: set TRANSFORMERS_OFFLINE=1 and HF_HUB_OFFLINE=1 if using HF models elsewhere.

set -euo pipefail

module purge
module load python/3.11.7

PROJ=/leonardo_work/EUHPC_D29_035/Epi4Ab
cd "$PROJ"

if [ -d "venv" ]; then
  source venv/bin/activate
fi

: "${OUT_BASE:?Set OUT_BASE to preprocessing root (…/upstream_preprocess_autodetect/<RUN>)}"
: "${PDB_LIST:?Set PDB_LIST to CSV with a pdbID column (e.g., metadata.ok.fill_edge.proteinmpnn190.csv)}"

NODES_EDGES_DIR="$OUT_BASE/nodes_edges"

python scripts/synthetic_labels/generate_synth_seqitope.py \
  --nodes_edges_dir "$NODES_EDGES_DIR" \
  --pdb_list "$PDB_LIST"

