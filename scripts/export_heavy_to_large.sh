#!/usr/bin/env bash
# Archive superseded heavy scratch dirs to large scratch (tar.gz).
set -euo pipefail

SCRATCH="${EPI4AB_SCRATCH:-/leonardo_scratch/fast/EUHPC_D29_035/epi4ab}"
ARCHIVE="${EPI4AB_LARGE_ARCHIVE:-/leonardo_scratch/large/userexternal/knaung00/epi4ab_archive}"
LABEL="${LABEL:-$(date +%Y%m%d)}"
DRY=1

usage() {
  echo "Usage: $0 --dry-run | --apply [--label YYYYMMDD]"
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY=1; shift ;;
    --apply) DRY=0; shift ;;
    --label) LABEL="$2"; shift 2 ;;
    *) usage ;;
  esac
done

DEST="$ARCHIVE/$LABEL"
mkdir -p "$ARCHIVE"

CANDIDATES=(
  "$SCRATCH/upstream_preprocess_autodetect/20260204_134431"
  "$SCRATCH/upstream_preprocess_autodetect/20260204_134700"
  "$SCRATCH/upstream_preprocess_autodetect/20260204_134749"
  "$SCRATCH/upstream_preprocess_autodetect/20260204_155222"
  "$SCRATCH/upstream_preprocess_autodetect/20260205_152916"
  "$SCRATCH/upstream_preprocess_autodetect/20260205_154827"
  "$SCRATCH/upstream_preprocess_autodetect/20260205_160147"
  "$SCRATCH/upstream_preprocess_autodetect/20260205_162222"
  "$SCRATCH/upstream_preprocess_autodetect/20260305_133247"
  "$SCRATCH/training_dasa_matrix"
  "$SCRATCH/training_dasa_matrix_augmented_epitope_20260410"
  "$SCRATCH/training_dasa_matrix_loao_antigen"
  "$SCRATCH/training_dasa_ab_rsa_mse_vs_pearson_20260529_0950"
  "$SCRATCH/run_model_output"
)

echo "=== Export to $DEST (DRY=$DRY) ==="
for src in "${CANDIDATES[@]}"; do
  [[ -d "$src" ]] || continue
  name="$(basename "$src")"
  parent="$(basename "$(dirname "$src")")"
  tarball="$DEST/${parent}_${name}.tar.gz"
  if [[ $DRY -eq 1 ]]; then
    echo "TAR $src -> $tarball && rm -rf $src"
    continue
  fi
  mkdir -p "$DEST"
  manifest="$DEST/${parent}_${name}.MANIFEST.txt"
  {
    echo "original_path=$src"
    echo "archived_at=$(date -Iseconds)"
    echo "reason=superseded by plots hub migration"
  } > "$manifest"
  tar -czf "$tarball" -C "$(dirname "$src")" "$name"
  rm -rf "$src"
  echo "ARCHIVED $src -> $tarball (see $manifest)"
done

echo "=== Done ==="
