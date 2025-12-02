#!/bin/bash
# Install FreeSASA (Python) and PDB2PQR inside the project virtual environment
# Also verifies installations and prints follow-up instructions.

set -euo pipefail

PROJ=/leonardo_work/AIFAC_F01_302/Epi4Ab
VENV=$PROJ/venv
TOOLS_DIR=$PROJ/tools

if [ ! -d "$VENV" ]; then
    echo "ERROR: Virtual environment not found at $VENV"
    echo "Create it first, e.g. python3 -m venv venv && source venv/bin/activate"
    exit 1
fi

source "$VENV/bin/activate"
mkdir -p "$TOOLS_DIR/bin"

printf '\n==========================================\n'
printf 'Installing structural analysis tools\n'
printf '==========================================\n\n'

# Install/upgrade PDB2PQR
printf '=== Installing PDB2PQR (pip) ===\n'
pip install --upgrade pdb2pqr >/tmp/pdb2pqr_install.log 2>&1 && tail -n 5 /tmp/pdb2pqr_install.log || cat /tmp/pdb2pqr_install.log
if command -v pdb2pqr >/dev/null 2>&1; then
    printf '  ✓ pdb2pqr command available: %s\n' "$(which pdb2pqr)"
    pdb2pqr --version 2>&1 | head -3 || true
else
    printf '  ⚠️  pdb2pqr command not on PATH, but Python package should be available.\n'
    python -c "import pdb2pqr; print('  ✓ pdb2pqr module path:', pdb2pqr.__file__)" || printf '  ✗ Unable to import pdb2pqr module\n'
fi

# Install/upgrade FreeSASA (pure Python wheel includes native library)
printf '\n=== Installing FreeSASA (pip) ===\n'
pip install --upgrade freesasa >/tmp/freesasa_install.log 2>&1 && tail -n 5 /tmp/freesasa_install.log || cat /tmp/freesasa_install.log
python -c "import freesasa; print('  ✓ freesasa version:', freesasa.__version__);" || {
    printf '  ✗ FreeSASA import failed. Ensure build dependencies are available.\n'
    deactivate
    exit 1
}

# Summary and PATH guidance
printf '\n==========================================\n'
printf 'Installation summary\n'
printf '==========================================\n'
printf '  FreeSASA  : available via python -m freesasa (module path checked above)\n'
printf '  PDB2PQR   : %s\n' "$(command -v pdb2pqr >/dev/null 2>&1 && which pdb2pqr || echo 'Python module only (use python -m pdb2pqr)')"
printf '\nTo ensure commands are accessible, add to ~/.bashrc:\n'
printf '  export PATH="%s/bin:$PATH"\n' "$TOOLS_DIR"
printf '  source %s/bin/activate\n' "$VENV"

printf '\nVerification steps:\n'
printf '  python -c "import freesasa"\n'
printf '  pdb2pqr --version (or python -m pdb2pqr --help)\n'

printf '\nDone.\n'

deactivate
