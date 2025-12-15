#!/bin/bash
# Generate Epi4Ab Results Dashboard
# This script can run on login node (no heavy compute required)

set -euo pipefail

PROJ_DIR="/leonardo_work/AIFAC_F01_302/Epi4Ab"
cd "$PROJ_DIR"

# Check arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <test_record_dir> [pdb_id]"
    echo ""
    echo "Examples:"
    echo "  $0 output_inference/2025-12-12_GNNResNet_7/test_record"
    echo "  $0 output_inference/2025-12-12_GNNResNet_7/test_record 1n8z_BAC"
    echo ""
    exit 1
fi

TEST_RECORD_DIR="$1"
PDB_ID="${2:-}"

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
echo "Started: $(date)"
echo "=================================================="
echo ""

# Run visualization script
if [ -n "$PDB_ID" ]; then
    ./venv/bin/python scripts/visualize_results.py \
        --test_record_dir "$TEST_RECORD_DIR" \
        --pdb_id "$PDB_ID"
else
    ./venv/bin/python scripts/visualize_results.py \
        --test_record_dir "$TEST_RECORD_DIR"
fi

# Find the dashboard file
DASHBOARD_FILE="$(dirname "$TEST_RECORD_DIR")/dashboard.html"

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
    echo "To view the dashboard:"
    echo "  1. Download the file to your local machine"
    echo "  2. Open in a web browser"
    echo ""
    echo "Or use VSCode to open and preview:"
    echo "  code $DASHBOARD_FILE"
    echo "=================================================="
else
    echo "Error: Dashboard file not created"
    exit 1
fi
