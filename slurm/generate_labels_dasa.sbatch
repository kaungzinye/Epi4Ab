#!/bin/bash
#SBATCH --job-name=epi4ab-dasa-labels
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

# Generate Delta-ASA labels (ASA_BACKEND=dssp|freesasa|biopython_sr; see OUTPUT_FILE):
#   nodes_edges/<pdb_id>/<OUTPUT_FILE>  (default node_label_dasa.parquet)

set -euo pipefail

module purge
module load python/3.11.7

PROJ=/leonardo_work/EUHPC_D29_035/Epi4Ab
cd "$PROJ"

if [ -d "venv" ]; then
  source venv/bin/activate
fi

: "${OUT_BASE:?Set OUT_BASE to preprocessing root (.../upstream_preprocess_autodetect/<RUN>)}"
: "${METADATA_FILE:?Set METADATA_FILE to metadata CSV used for preprocessing (must include pdbID,pdb,antigen)}"
: "${PDB_LIST:?Set PDB_LIST to CSV with a pdbID column (for example gates/fill_edge/metadata.ok.fill_edge.csv)}"

NODES_EDGES_DIR="$OUT_BASE/nodes_edges"
PROCESSED_DIR="$OUT_BASE/processed_data"

# ASA_BACKEND=dssp (default) | freesasa (pip install freesasa) | biopython_sr (biopython only)
ASA_BACKEND="${ASA_BACKEND:-dssp}"
# Optional override: DSSP binary when ASA_BACKEND=dssp (default mkdssp)
DSSP_BIN="${DSSP_BIN:-mkdssp}"
# Optional: non-default output name (e.g. node_label_dasa_freesasa.parquet for FreeSASA runs)
OUTPUT_FILE="${OUTPUT_FILE:-node_label_dasa.parquet}"

python scripts/generate_dasa_labels.py \
  --nodes_edges_dir "$NODES_EDGES_DIR" \
  --processed_dir "$PROCESSED_DIR" \
  --metadata "$METADATA_FILE" \
  --pdb_list "$PDB_LIST" \
  --asa_backend "$ASA_BACKEND" \
  --dssp_bin "$DSSP_BIN" \
  --output_file "$OUTPUT_FILE"

