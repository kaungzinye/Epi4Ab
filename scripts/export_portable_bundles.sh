#!/usr/bin/env bash
# Create tar.gz bundles on Leonardo for download to a local hard drive.
# Does NOT delete source data. See docs/LOCAL_BACKUP.md
set -euo pipefail

SCRATCH="${EPI4AB_SCRATCH:-/leonardo_scratch/fast/EUHPC_D29_035/epi4ab}"
OUT_TAG="${EPI4AB_OUT_BASE_TAG:-20260305_143201}"
LARGE_ARCHIVE="${EPI4AB_LARGE_ARCHIVE:-/leonardo_scratch/large/userexternal/knaung00/epi4ab_archive}"
LABEL="${LABEL:-$(date +%Y%m%d)}"
DEST="${BACKUP_DEST:-$SCRATCH/portable_bundles/$LABEL}"
DRY=1
INCLUDE_PHASE1=0
INCLUDE_LARGE_ARCHIVE=0

usage() {
  cat <<EOF
Usage: $0 --dry-run | --apply [options]

Creates tar.gz files under BACKUP_DEST (default: \$SCRATCH/portable_bundles/<LABEL>/).

Bundles (always when paths exist):
  epi4ab_plots.tar.gz              — plots hub (HTML/PNG/CSV summaries)
  epi4ab_outbase_<tag>.tar.gz       — canonical preprocess (nodes_edges, gates, cache)
  epi4ab_training_dasa_v103.tar.gz  — active DASA training (model.pt, test_record, splits)

Options:
  --label YYYYMMDD       Output folder name (default: today)
  --dest PATH            Override BACKUP_DEST
  --include-phase1       Also bundle training_phase1/
  --include-large-archive  Also bundle latest epi4ab_archive/<label> from large scratch
  --dry-run              Print actions only (default)
  --apply                Write tarballs

After --apply, download from your laptop:
  rsync -avP USER@login.leonardo.cineca.it:$DEST/ ~/Backups/Epi4Ab/
EOF
  exit 1
}

tar_bundle() {
  local name="$1"
  local parent="$2"
  local dir="$3"
  local out="$DEST/${name}.tar.gz"
  if [[ ! -d "$dir" ]]; then
    echo "SKIP $name (missing $dir)"
    return 0
  fi
  if [[ $DRY -eq 1 ]]; then
    echo "TAR $out <- $dir"
    return 0
  fi
  mkdir -p "$DEST"
  tar -czf "$out" -C "$parent" "$(basename "$dir")"
  echo "WROTE $out ($(du -h "$out" | cut -f1))"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY=1; shift ;;
    --apply) DRY=0; shift ;;
    --label) LABEL="$2"; DEST="${BACKUP_DEST:-$SCRATCH/portable_bundles/$LABEL}"; shift 2 ;;
    --dest) DEST="$2"; shift 2 ;;
    --include-phase1) INCLUDE_PHASE1=1; shift ;;
    --include-large-archive) INCLUDE_LARGE_ARCHIVE=1; shift ;;
    -h|--help) usage ;;
    *) usage ;;
  esac
done

DEST="${BACKUP_DEST:-$SCRATCH/portable_bundles/$LABEL}"

echo "=== Portable bundles -> $DEST (DRY=$DRY) ==="

tar_bundle "epi4ab_plots" "$SCRATCH" "$SCRATCH/plots"
tar_bundle "epi4ab_outbase_${OUT_TAG}" \
  "$SCRATCH/upstream_preprocess_autodetect" \
  "$SCRATCH/upstream_preprocess_autodetect/$OUT_TAG"
tar_bundle "epi4ab_training_dasa_v103" "$SCRATCH" "$SCRATCH/training_dasa_v103"

if [[ $INCLUDE_PHASE1 -eq 1 ]]; then
  tar_bundle "epi4ab_training_phase1" "$SCRATCH" "$SCRATCH/training_phase1"
fi

if [[ $INCLUDE_LARGE_ARCHIVE -eq 1 ]]; then
  latest="$(ls -1d "$LARGE_ARCHIVE"/*/ 2>/dev/null | sort | tail -1 || true)"
  if [[ -n "$latest" ]]; then
    tar_bundle "epi4ab_large_archive_$(basename "${latest%/}")" \
      "$LARGE_ARCHIVE" "${latest%/}"
  else
    echo "SKIP large archive (none under $LARGE_ARCHIVE)"
  fi
fi

if [[ $DRY -eq 0 ]]; then
  cat > "$DEST/MANIFEST.txt" <<EOF
created=$(date -Iseconds)
host=$(hostname)
out_base_tag=$OUT_TAG
bundles=$(ls -1 "$DEST"/*.tar.gz 2>/dev/null | xargs -n1 basename | tr '\n' ' ')
download_example=rsync -avP USER@login.leonardo.cineca.it:$DEST/ ~/Backups/Epi4Ab/
EOF
  echo "Wrote $DEST/MANIFEST.txt"
fi

echo "=== Done ==="
