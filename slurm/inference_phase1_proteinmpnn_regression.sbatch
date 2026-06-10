#!/bin/bash
#SBATCH --job-name=epi4ab-infer-phase1-reg
#SBATCH --account=EUHPC_D29_035
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=04:00:00
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

export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

OUT_BASE="${OUT_BASE:?Set OUT_BASE to preprocessing output (e.g. /.../upstream_preprocess_autodetect/<RUN_TAG>)}"
NODES_EDGES_DIR="$OUT_BASE/nodes_edges"
PROCESSED_DIR="$OUT_BASE/processed_data"

# Use gated PDB list by default
METADATA_OK="${METADATA_OK:-$OUT_BASE/gates/fill_edge/metadata.ok.fill_edge.csv}"

# Trained model folder (the folder that contains model.pt + log.md)
MODEL_DIR="${MODEL_DIR:?Set MODEL_DIR to the trained model output folder (contains model.pt)}"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-/leonardo_scratch/fast/EUHPC_D29_035/epi4ab/output_inference_phase1/${RUN_TAG}}"

PDB_LIST="$OUT_DIR/pdb_list.csv"
export METADATA_OK PDB_LIST

mkdir -p "$OUT_DIR"

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
pd.DataFrame({'pdbID': df[col].astype(str)}).to_csv(out, index=False)
print(f"Wrote {out} with {len(df)} PDBs")
PY

echo "Running regression inference..."
python epi_prediction.py \
  "$NODES_EDGES_DIR" \
  "$PROCESSED_DIR" \
  "$PDB_LIST" \
  "$MODEL_DIR" \
  "$OUT_DIR"

echo "Inference outputs: $OUT_DIR"
echo "DONE"
