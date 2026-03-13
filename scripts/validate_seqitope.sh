#!/usr/bin/env bash
set -euo pipefail
# Validate Seqitope labels against node_feature per PDB.
# Usage:
#   METADATA=<csv> OUT_BASE=<.../upstream_preprocess_autodetect/<RUN>} \
#   ./scripts/validate_seqitope.sh
: "${METADATA:?Set METADATA to a CSV with pdbID column}"
: "${OUT_BASE:?Set OUT_BASE to preprocessing output root}"

NODES_EDGES_DIR="$OUT_BASE/nodes_edges"
python scripts/validate_pipeline.py \
  --metadata "$METADATA" \
  --nodes_edges_dir "$NODES_EDGES_DIR" \
  --step labels \
  --target_type seqitope \
  --target_file node_label_seqitope.parquet \
  --target_column score
