#!/bin/bash
#SBATCH --job-name=proteinmpnn-scores-full-autodetect
#SBATCH --account=EUHPC_D29_035
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=24:00:00
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

PROTEINMPNN_DIR="$PROJ/tools/proteinmpnn/ProteinMPNN"

OUT_BASE="${OUT_BASE:?Set OUT_BASE to the preprocessing output folder (e.g. /.../upstream_preprocess_autodetect/<RUN_TAG>)}"
NODES_EDGES_DIR="$OUT_BASE/nodes_edges"
PROCESSED_DIR="$OUT_BASE/processed_data"
OUT_ROOT="${OUT_ROOT:-/leonardo_scratch/fast/EUHPC_D29_035/epi4ab/proteinmpnn_scores_autodetect_full}"

# Make OUT_BASE visible to embedded Python snippets
export OUT_BASE

# Prefer gated/filtered metadata if available
METADATA_OK_DEFAULT="$OUT_BASE/gates/fill_edge/metadata.ok.fill_edge.csv"
METADATA_OK="${METADATA_OK:-$METADATA_OK_DEFAULT}"
export METADATA_OK

MANIFEST="$OUT_BASE/proteinmpnn_manifest_autogen.csv"

echo "=========================================="
echo "ProteinMPNN Scoring - Full set (autodetect-preprocessed lig.pdb)"
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo "OUT_BASE: $OUT_BASE"
echo "Nodes/edges: $NODES_EDGES_DIR"
echo "Processed: $PROCESSED_DIR"
echo "Out root: $OUT_ROOT"
echo "=========================================="

mkdir -p "$OUT_ROOT"

echo "Generating manifest from processed_data lig.pdb paths..."
python - <<'PY'
import os, csv

out_base = os.environ['OUT_BASE']
processed = os.path.join(out_base, 'processed_data')
manifest = os.path.join(out_base, 'proteinmpnn_manifest_autogen.csv')
metadata_ok = os.environ.get('METADATA_OK', '')

rows=[]
if metadata_ok and os.path.exists(metadata_ok):
    # Use filtered list of PDBs from gate output.
    with open(metadata_ok, newline='') as f:
        r = csv.DictReader(f)
        for row in r:
            pdb_id = row.get('pdbID') or row.get('pdbId') or row.get('pdb_id')
            if not pdb_id:
                continue
            lig = os.path.join(processed, str(pdb_id), 'lig.pdb')
            if os.path.exists(lig):
                rows.append((str(pdb_id), lig, ''))
else:
    for pdb_id in sorted(os.listdir(processed)):
        lig=os.path.join(processed,pdb_id,'lig.pdb')
        if os.path.exists(lig):
            rows.append((pdb_id, lig, ''))

with open(manifest,'w',newline='') as f:
    w=csv.writer(f)
    w.writerow(['pdb_id','pdb_path','chain_id'])
    for r in rows:
        w.writerow(r)

print(f"Wrote manifest {manifest} with {len(rows)} rows")
PY

python scripts/run_proteinmpnn_score_only_and_prepare_targets.py \
  --manifest "$MANIFEST" \
  --nodes_edges_dir "$NODES_EDGES_DIR" \
  --proteinmpnn_run "$PROTEINMPNN_DIR/protein_mpnn_run.py" \
  --out_root "$OUT_ROOT" \
  --target_file proteinmpnn_scores.parquet \
  --skip_existing

python scripts/validate_pipeline.py \
  --metadata "$MANIFEST" \
  --nodes_edges_dir "$NODES_EDGES_DIR" \
  --processed_dir "$PROCESSED_DIR" \
  --step labels \
  --target_type proteinmpnn \
  --target_file proteinmpnn_scores.parquet \
  --target_column score

if command -v deactivate >/dev/null 2>&1; then
  deactivate
fi
