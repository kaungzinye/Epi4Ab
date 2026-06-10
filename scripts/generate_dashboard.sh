#!/bin/bash
# Generate Epi4Ab Results Dashboard
# This script can run on login node (no heavy compute required)

set -euo pipefail

PROJ_DIR="/leonardo_work/EUHPC_D29_035/Epi4Ab"
cd "$PROJ_DIR"

# Check arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <test_record_dir> [pdb_id]"
    echo ""
    echo "Set RUN_ID to write dashboard under epi4ab/plots/runs/<RUN_ID>/html/"
    echo ""
    echo "Examples:"
    echo "  RUN_ID=v103_biopy_s42 $0 /path/to/test_record"
    echo "  $0 output_inference/2025-12-12_GNNResNet_7/test_record 1n8z_BAC"
    echo ""
    exit 1
fi

TEST_RECORD_DIR="$1"
PDB_ID="${2:-}"
RUN_ID="${RUN_ID:-}"
PLOTS_ROOT="${EPI4AB_PLOTS_ROOT:-/leonardo_scratch/fast/EUHPC_D29_035/epi4ab/plots}"

# Verify test_record directory exists
if [ ! -d "$TEST_RECORD_DIR" ]; then
    echo "Error: Test record directory not found: $TEST_RECORD_DIR"
    exit 1
fi

# Count result files
RESULT_COUNT=$(ls "$TEST_RECORD_DIR"/*_final_result.txt 2>/dev/null | wc -l)
if [ "$RESULT_COUNT" -eq 0 ]; then
    echo "Error: No result files found in $TEST_RECORD_DIR"
    exit 1
fi

echo "=================================================="
echo "Epi4Ab Dashboard Generation"
echo "=================================================="
echo "Test record directory: $TEST_RECORD_DIR"
echo "Found $RESULT_COUNT PDB result(s)"
if [ -n "$PDB_ID" ]; then
    echo "Generating dashboard for: $PDB_ID"
else
    echo "Generating dashboard for all PDBs"
fi
if [ -n "$RUN_ID" ]; then
    echo "RUN_ID: $RUN_ID"
fi
echo "Started: $(date)"
echo "=================================================="
echo ""

VIZ_ARGS=(--test_record_dir "$TEST_RECORD_DIR" --build-index)
if [ -n "$RUN_ID" ]; then
    VIZ_ARGS+=(--run_id "$RUN_ID")
fi
if [ -n "$PDB_ID" ]; then
    VIZ_ARGS+=(--pdb_id "$PDB_ID")
fi

if [ -x "./venv/bin/python" ]; then
    ./venv/bin/python scripts/visualize_results.py "${VIZ_ARGS[@]}"
else
    python scripts/visualize_results.py "${VIZ_ARGS[@]}"
fi

if [ -n "$RUN_ID" ]; then
    DASHBOARD_FILE="$PLOTS_ROOT/runs/$RUN_ID/html/dashboard.html"
else
    DASHBOARD_FILE="$(dirname "$TEST_RECORD_DIR")/dashboard.html"
fi

if [ -f "$DASHBOARD_FILE" ]; then
    DASHBOARD_SIZE=$(du -h "$DASHBOARD_FILE" | cut -f1)
    echo ""
    echo "=================================================="
    echo "Dashboard generation complete!"
    echo "=================================================="
    echo "Dashboard file: $DASHBOARD_FILE"
    echo "File size: $DASHBOARD_SIZE"
    echo "Finished: $(date)"
    echo "=================================================="
    echo ""
    echo "Master plots index: $PLOTS_ROOT/index.html"
    echo "=================================================="
else
    echo "Error: Dashboard file not created"
    exit 1
fi
