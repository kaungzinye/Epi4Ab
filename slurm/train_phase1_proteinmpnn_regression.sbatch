#!/bin/bash
#SBATCH --job-name=epi4ab-train-phase1-proteinmpnn
#SBATCH --account=EUHPC_D29_035
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
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

# Offline safety for HF models (optional). If your cache is populated, this
# avoids accidental network attempts on compute nodes.
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

# Optional: point HF cache to a shared location.
# Recommended to set HF_HOME in your environment or .env.
if [ -n "${HF_HOME:-}" ]; then
  export HF_HOME
fi

OUT_BASE="${OUT_BASE:?Set OUT_BASE to preprocessing output (e.g. /.../upstream_preprocess_autodetect/<RUN_TAG>)}"
NODES_EDGES_DIR="$OUT_BASE/nodes_edges"
PROCESSED_DIR="$OUT_BASE/processed_data"
METADATA_OK="${METADATA_OK:-$OUT_BASE/gates/fill_edge/metadata.ok.fill_edge.csv}"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-/leonardo_scratch/fast/EUHPC_D29_035/epi4ab/training_phase1/${RUN_TAG}}"

PARAMETERS_FILE="${PARAMETERS_FILE:-$PROJ/parameters_phase1_proteinmpnn_regression.txt}"

PDB_LIST="$OUT_DIR/pdb_list.csv"

# Export for embedded Python snippets
export METADATA_OK
export PDB_LIST

echo "=========================================="
echo "Epi4Ab Phase 1 Training (ProteinMPNN regression)"
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo "OUT_BASE: $OUT_BASE"
echo "NODES_EDGES_DIR: $NODES_EDGES_DIR"
echo "PROCESSED_DIR: $PROCESSED_DIR"
echo "METADATA_OK: $METADATA_OK"
echo "PARAMETERS_FILE: $PARAMETERS_FILE"
echo "OUT_DIR: $OUT_DIR"
echo "=========================================="

mkdir -p "$OUT_DIR"

echo "Generating training pdb list..."
python - <<'PY'
import os
import pandas as pd

meta = os.environ['METADATA_OK']
out = os.environ['PDB_LIST']

df = pd.read_csv(meta)
col = None
for c in ('pdbID', 'pdbId', 'pdb_id'):
    if c in df.columns:
        col = c
        break
if col is None:
    col = df.columns[0]

out_df = pd.DataFrame({'pdbID': df[col].astype(str).tolist()})
out_df.to_csv(out, index=False)
print(f"Wrote {out} with {len(out_df)} PDBs")
PY

echo "Building CLI args from $PARAMETERS_FILE"
PARAMS=""
while IFS='=' read -r key value || [ -n "$key" ]; do
  [[ ! $key =~ ^\ *# && -n $key ]] && PARAMS+="--${key// /} $value "
done < "$PARAMETERS_FILE"

SAVED_PARAMS="$(<"$PARAMETERS_FILE")"
SAVED_PARAMS=$(echo "$SAVED_PARAMS" | tr '\n' ';')
SAVED_PARAMS=$(echo "$SAVED_PARAMS" | sed 's/ /+/g')

echo "Starting training..."
python epi_graph.py \
  "$SAVED_PARAMS" \
  "$NODES_EDGES_DIR" \
  "$PROCESSED_DIR" \
  "$PDB_LIST" \
  "$PDB_LIST" \
  "$PDB_LIST" \
  "$PDB_LIST" \
  "$OUT_DIR" \
  $PARAMS

echo "DONE"

if command -v deactivate >/dev/null 2>&1; then
  deactivate
fi
