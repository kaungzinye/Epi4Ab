#!/bin/bash
# Installation script for BepiPred 3.0 and Ellipro on Leonardo HPC
# Install in project directory so tools are available on compute nodes

set -euo pipefail

PROJ=/leonardo_work/AIFAC_F01_302/Epi4Ab
cd $PROJ

# Activate virtual environment
source venv/bin/activate

# Create tools directory
TOOLS_DIR="$PROJ/tools"
mkdir -p "$TOOLS_DIR"
cd "$TOOLS_DIR"

echo "=========================================="
echo "Installing BepiPred 3.0 and Ellipro"
echo "=========================================="
echo "Installation directory: $TOOLS_DIR"
echo ""

# Check Python version
python3 --version

# Install BepiPred 3.0
echo ""
echo "=== Installing BepiPred 3.0 ==="
echo ""

# Try multiple installation methods
if pip install bepipred3 2>&1 | tee bepipred_install.log; then
    echo "✓ BepiPred 3.0 installed via pip (bepipred3)"
elif pip install bepipred-3.0 2>&1 | tee bepipred_install.log; then
    echo "✓ BepiPred 3.0 installed via pip (bepipred-3.0)"
elif pip install git+https://github.com/cathalobrien/bepipred-3.0.git 2>&1 | tee bepipred_install.log; then
    echo "✓ BepiPred 3.0 installed from GitHub"
else
    echo "⚠️  Standard pip installation failed, checking alternatives..."
    # Check if it's available as a Python package
    python3 -c "import bepipred3; print('BepiPred 3.0 available as Python package')" 2>&1 || echo "Not available as Python package"
fi

# Check if bepipred command is now available
echo ""
echo "Checking BepiPred installation..."
for cmd in bepipred bepipred3 bepipred-3.0 bepipred3.0; do
    if command -v $cmd &> /dev/null; then
        echo "  ✓ Found command: $cmd -> $(which $cmd)"
    fi
done

# Try to find bepipred in Python packages
python3 -c "
import sys
import subprocess
try:
    import bepipred3
    print(f'  ✓ BepiPred 3.0 Python package found: {bepipred3.__file__}')
except ImportError:
    try:
        import bepipred
        print(f'  ✓ BepiPred Python package found: {bepipred.__file__}')
    except ImportError:
        print('  ⚠️  BepiPred not found as Python package')
" 2>&1 || echo "  ⚠️  Could not check Python package"

# Install Ellipro
echo ""
echo "=== Installing Ellipro ==="
echo ""

# Ellipro is typically available via IEDB API or as standalone tool
# Check if there's a Python package
if pip install ellipro 2>&1 | tee ellipro_install.log; then
    echo "✓ Ellipro installed via pip"
elif pip install git+https://github.com/iedb/ellipro.git 2>&1 | tee ellipro_install.log; then
    echo "✓ Ellipro installed from GitHub"
else
    echo "⚠️  Standard pip installation failed"
    echo "  Note: Ellipro may need to be downloaded manually from IEDB"
    echo "  Or use web API (already implemented in code)"
fi

# Check if ellipro command is now available
echo ""
echo "Checking Ellipro installation..."
for cmd in ellipro ellipro-web ellipro-standalone; do
    if command -v $cmd &> /dev/null; then
        echo "  ✓ Found command: $cmd -> $(which $cmd)"
    fi
done

# Try to find ellipro in Python packages
python3 -c "
try:
    import ellipro
    print(f'  ✓ Ellipro Python package found: {ellipro.__file__}')
except ImportError:
    print('  ⚠️  Ellipro not found as Python package')
" 2>&1 || echo "  ⚠️  Could not check Python package"

echo ""
echo "=========================================="
echo "Installation Summary"
echo "=========================================="
echo ""
echo "Tools directory: $TOOLS_DIR"
echo ""
echo "To use these tools, ensure PATH includes:"
echo "  export PATH=\"$TOOLS_DIR/bin:\$PATH\""
echo ""
echo "Or install in venv bin:"
echo "  export PATH=\"$PROJ/venv/bin:\$PATH\""
echo ""

deactivate


