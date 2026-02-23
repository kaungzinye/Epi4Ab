#!/bin/bash
#SBATCH --job-name=epi4ab-preprocess-full-autodetect
#SBATCH --account=EUHPC_D29_035
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
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

METADATA_FILE="${METADATA_FILE:-/leonardo_scratch/fast/EUHPC_D29_035/epi4ab/upstream_preprocess/metadata_complete.csv}"
SRC_PROCESSED_DIR="${SRC_PROCESSED_DIR:-/leonardo_scratch/fast/EUHPC_D29_035/epi4ab/upstream_preprocess/processed_data}"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUT_BASE="${OUT_BASE:-/leonardo_scratch/fast/EUHPC_D29_035/epi4ab/upstream_preprocess_autodetect/${RUN_TAG}}"
DIRECTORY_PROCESSED="$OUT_BASE/processed_data"
DIRECTORY_NODES_EDGES="$OUT_BASE/nodes_edges"

# Export for embedded Python snippets
export METADATA_FILE SRC_PROCESSED_DIR DIRECTORY_PROCESSED

echo "=========================================="
echo "Epi4Ab Preprocessing - Full set (autodetect antigen chain)"
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo "METADATA: $METADATA_FILE"
echo "SRC_PROCESSED_DIR: $SRC_PROCESSED_DIR"
echo "OUT_BASE: $OUT_BASE"
echo "PROCESSED: $DIRECTORY_PROCESSED"
echo "NODES_EDGES: $DIRECTORY_NODES_EDGES"
echo "PYMOL: ${PYMOL_EXECUTABLE:-<unset>}"
echo "=========================================="

mkdir -p "$DIRECTORY_PROCESSED" "$DIRECTORY_NODES_EDGES" logs

echo "Seeding CIFs from existing processed_data..."
python - <<'PY'
import os, shutil
import pandas as pd

meta=os.environ['METADATA_FILE']
src=os.environ['SRC_PROCESSED_DIR']
out=os.environ['DIRECTORY_PROCESSED']

df=pd.read_csv(meta)
need=['pdbID','pdb']
for c in need:
    if c not in df.columns:
        raise SystemExit(f"Metadata missing required column: {c}")

missing=[]
for _,r in df.iterrows():
    pdb_id=str(r['pdbID'])
    pdb=str(r['pdb'])
    src_cif=os.path.join(src, pdb_id, f"{pdb}.cif")
    dst_dir=os.path.join(out, pdb_id)
    os.makedirs(dst_dir, exist_ok=True)
    dst_cif=os.path.join(dst_dir, f"{pdb}.cif")
    if os.path.exists(src_cif):
        shutil.copy2(src_cif, dst_cif)
    else:
        missing.append((pdb_id, src_cif))

print(f"Seeded CIFs for {len(df)-len(missing)}/{len(df)} PDBs")
if missing:
    print("Missing CIFs (first 10):")
    for pdb_id, path in missing[:10]:
        print(" ", pdb_id, path)
PY

echo "Step 1: preprocessing (extract lig.pdb + features)"
python preprocess/main.py "$METADATA_FILE" "$DIRECTORY_PROCESSED" --autodetect_antigen_chain

echo "Gate 1: verify preprocessing outputs + filter metadata"
GATE_PRE_DIR="$OUT_BASE/gates/preprocess"
python scripts/pipeline_gate.py \
  --stage preprocess \
  --metadata "$METADATA_FILE" \
  --processed_dir "$DIRECTORY_PROCESSED" \
  --out_dir "$GATE_PRE_DIR"
METADATA_OK_PRE="$GATE_PRE_DIR/metadata.ok.preprocess.csv"

echo "Using filtered metadata: $METADATA_OK_PRE"

echo "Step 2: nodes/edges (CB)"
python preprocess/nodes_edges.py "$METADATA_OK_PRE" "$DIRECTORY_PROCESSED" "$DIRECTORY_NODES_EDGES" "$PYMOL_EXECUTABLE" 'CB'

echo "Step 3: nodes/edges (CA)"
python preprocess/nodes_edges.py "$METADATA_OK_PRE" "$DIRECTORY_PROCESSED" "$DIRECTORY_NODES_EDGES" "$PYMOL_EXECUTABLE" 'CA'

echo "Gate 2: verify nodes/edges outputs + filter metadata"
GATE_NE_DIR="$OUT_BASE/gates/nodes_edges"
python scripts/pipeline_gate.py \
  --stage nodes_edges \
  --metadata "$METADATA_OK_PRE" \
  --nodes_edges_dir "$DIRECTORY_NODES_EDGES" \
  --out_dir "$GATE_NE_DIR"
METADATA_OK_NE="$GATE_NE_DIR/metadata.ok.nodes_edges.csv"

echo "Using filtered metadata: $METADATA_OK_NE"

echo "Step 4: fill edges"
python preprocess/fill_edge.py "$METADATA_OK_NE" "$DIRECTORY_NODES_EDGES"

echo "Gate 3: verify filled edge outputs"
GATE_FE_DIR="$OUT_BASE/gates/fill_edge"
python scripts/pipeline_gate.py \
  --stage fill_edge \
  --metadata "$METADATA_OK_NE" \
  --nodes_edges_dir "$DIRECTORY_NODES_EDGES" \
  --out_dir "$GATE_FE_DIR"

echo "DONE: Full preprocessing complete"
echo "OUT_BASE=$OUT_BASE"

if command -v deactivate >/dev/null 2>&1; then
  deactivate
fi
