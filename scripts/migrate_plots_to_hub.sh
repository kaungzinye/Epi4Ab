#!/usr/bin/env bash
# Move existing plot artifacts into epi4ab/plots/ hub (see docs/ARTIFACT_LAYOUT.md).
set -euo pipefail

PROJ="${PROJ:-/leonardo_work/EUHPC_D29_035/Epi4Ab}"
SCRATCH="${EPI4AB_SCRATCH:-/leonardo_scratch/fast/EUHPC_D29_035/epi4ab}"
PLOTS="${EPI4AB_PLOTS_ROOT:-$SCRATCH/plots}"
OUT_TAG="${EPI4AB_OUT_BASE_TAG:-20260305_143201}"
OUT_BASE="${EPI4AB_OUT_BASE:-$SCRATCH/upstream_preprocess_autodetect/$OUT_TAG}"
REPORTS="$OUT_BASE/reports"
COHORT="$PLOTS/cohort/$OUT_TAG"
DRY=1

usage() {
  echo "Usage: $0 --dry-run | --apply"
  exit 1
}

mv1() {
  local src="$1" dst="$2"
  if [[ ! -e "$src" ]]; then
    return 0
  fi
  mkdir -p "$(dirname "$dst")"
  if [[ $DRY -eq 1 ]]; then
    echo "MV $src -> $dst"
  else
    mv "$src" "$dst"
    echo "MOVED $src -> $dst"
  fi
}

mv_html_dedupe() {
  local src="$1" dst_dir="$2"
  [[ -f "$src" ]] || return 0
  local base dst
  base="$(basename "$src")"
  dst="$dst_dir/$base"
  mkdir -p "$dst_dir"
  if [[ -f "$dst" ]] && [[ "$(stat -c%s "$src")" == "$(stat -c%s "$dst")" ]]; then
    if [[ $DRY -eq 1 ]]; then
      echo "SKIP duplicate $src"
    else
      rm -f "$src"
      echo "REMOVED duplicate $src (already at $dst)"
    fi
    return 0
  fi
  mv1 "$src" "$dst"
}

[[ $# -eq 1 ]] || usage
case "$1" in
  --dry-run) DRY=1 ;;
  --apply) DRY=0 ;;
  *) usage ;;
esac

echo "=== Plots hub migration (DRY=$DRY) ==="
echo "PLOTS=$PLOTS"
echo "OUT_BASE=$OUT_BASE"

# Cohort HTML at reports root
for f in "$REPORTS"/*.html; do
  [[ -f "$f" ]] || continue
  mv_html_dedupe "$f" "$COHORT/html"
done

# literature_dual_by_run
if [[ -d "$REPORTS/literature_dual_by_run" ]]; then
  mv1 "$REPORTS/literature_dual_by_run" "$COHORT/literature_dual_by_run"
fi

# plots_archive
if [[ -d "$REPORTS/plots_archive_20260522" ]]; then
  for f in "$REPORTS/plots_archive_20260522"/*.html; do
    [[ -f "$f" ]] || continue
    mv_html_dedupe "$f" "$COHORT/html"
  done
  if [[ $DRY -eq 0 ]]; then
    rmdir "$REPORTS/plots_archive_20260522" 2>/dev/null || true
  fi
fi

# Viz CSVs
for pat in dasa_runs_*.csv dasa_backend_comparison*.csv dasa_epitope_confusion*.csv \
  lit_subset_eval*.csv dasa_noise_*.csv cohort_uniprots_*.csv; do
  for f in "$REPORTS"/$pat; do
    [[ -f "$f" ]] || continue
    mv1 "$f" "$COHORT/csv/$(basename "$f")"
  done
done

# Cache moves
mkdir -p "$OUT_BASE/cache/covabdab" "$OUT_BASE/cache/pdbe" "$OUT_BASE/cache/iedb"
for f in "$REPORTS"/CoV-AbDab*.csv; do
  [[ -f "$f" ]] || continue
  mv1 "$f" "$OUT_BASE/cache/covabdab/$(basename "$f")"
done
if [[ -d "$REPORTS/pdbe_cache_cov" ]]; then
  mv1 "$REPORTS/pdbe_cache_cov" "$OUT_BASE/cache/pdbe/pdbe_cache_cov"
fi
for f in "$REPORTS"/iedb_*.csv "$REPORTS"/*manifest*.csv; do
  [[ -f "$f" ]] || continue
  base="$(basename "$f")"
  case "$base" in
    dasa_runs*|dasa_backend*) continue ;;
  esac
  mv1 "$f" "$OUT_BASE/cache/iedb/$base"
done

# v103 run
V103="$SCRATCH/training_dasa_v103/biopython_sr_seed42/2026-06-05_GNNResNet_1"
if [[ -d "$V103/plots" ]]; then
  for f in "$V103/plots"/*; do
    [[ -f "$f" ]] || continue
    mv1 "$f" "$PLOTS/runs/v103_biopy_s42/png/$(basename "$f")"
  done
  rmdir "$V103/plots" 2>/dev/null || true
fi
if [[ -d "$V103/reports" ]]; then
  for f in "$V103/reports"/*; do
    [[ -f "$f" ]] || continue
    ext="${f##*.}"
    if [[ "$ext" == "html" ]]; then
      mv1 "$f" "$PLOTS/runs/v103_biopy_s42/html/$(basename "$f")"
    elif [[ "$ext" == "csv" ]]; then
      mv1 "$f" "$PLOTS/runs/v103_biopy_s42/csv/$(basename "$f")"
    fi
  done
fi

# Phase1 scale190 dashboard
P1="$SCRATCH/training_phase1/phase1_scale190_noab_seed42/2026-03-10_GNNResNet_1/dashboard.html"
mv_html_dedupe "$P1" "$PLOTS/runs/phase1_scale190_s42/html"

# output_inference_phase1 dashboards
for f in "$SCRATCH/output_inference_phase1"/*/*/dashboard.html; do
  [[ -f "$f" ]] || continue
  parent="$(basename "$(dirname "$f")")"
  infer_parent="$(basename "$(dirname "$(dirname "$f")")")"
  rid="infer_${infer_parent}_${parent}"
  mv_html_dedupe "$f" "$PLOTS/runs/$rid/html"
done

# Legacy classification HTML
if [[ -d "$SCRATCH/run_model_output" ]]; then
  for run_dir in "$SCRATCH/run_model_output"/*; do
    [[ -d "$run_dir" ]] || continue
    name="$(basename "$run_dir")"
    for f in "$run_dir"/*.html; do
      [[ -f "$f" ]] || continue
      mv1 "$f" "$PLOTS/legacy/classification/$name/$(basename "$f")"
    done
  done
fi

# Repo reports -> legacy rsa
REPO_REPORTS="$PROJ/reports"
if [[ -d "$REPO_REPORTS" ]]; then
  mkdir -p "$PLOTS/legacy/rsa_ab_20260529"
  for f in "$REPO_REPORTS"/*; do
    [[ -e "$f" ]] || continue
    mv1 "$f" "$PLOTS/legacy/rsa_ab_20260529/$(basename "$f")"
  done
  if [[ $DRY -eq 0 ]]; then
    rmdir "$REPO_REPORTS" 2>/dev/null || true
  fi
fi

# Pointer in OUT_BASE/reports
README_TXT="$REPORTS/README.txt"
if [[ $DRY -eq 1 ]]; then
  echo "WRITE $README_TXT (pointer to plots hub)"
else
  mkdir -p "$REPORTS"
  cat > "$README_TXT" <<EOF
Plot and dashboard HTML/PNG/CSV summaries moved to:
  $PLOTS/cohort/$OUT_TAG/
Master index:
  $PLOTS/index.html
Bulk caches (IEDB, CoV-AbDab, PDBe) are under:
  $OUT_BASE/cache/
EOF
  echo "Wrote $README_TXT"
fi

if [[ $DRY -eq 0 ]]; then
  python "$PROJ/scripts/consolidate_dasa_runs_csv.py" \
    --reports_dir "$COHORT/csv" || true
  python "$PROJ/scripts/build_plots_index.py" || true
fi
echo "=== Done ==="
