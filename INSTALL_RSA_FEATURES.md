# Installing RSA Feature Dependencies

## Overview
- **FreeSASA** (Python package) → computes absolute SASA values; we normalise using Tien et al. (2013)
- **PDB2PQR** (Python package) → generates per-residue charges (already used in pipeline)
- External tools such as Naccess are no longer required.

## Quick Start
```bash
cd /leonardo_work/AIFAC_F01_302/Epi4Ab
source venv/bin/activate
bash install_structural_tools.sh
```

The script installs/updates both packages inside the project `venv/` and prints verification steps.

## Manual Installation
If you prefer to install manually:
```bash
cd /leonardo_work/AIFAC_F01_302/Epi4Ab
source venv/bin/activate
pip install --upgrade freesasa pdb2pqr
```

## Verification
```bash
source venv/bin/activate
python -c "import freesasa; print(f'FreeSASA {freesasa.__version__}')"
pdb2pqr --version  # or: python -m pdb2pqr --help
```

## Environment Configuration
Add the following to `~/.bashrc` (optional but recommended):
```bash
export PATH="/leonardo_work/AIFAC_F01_302/Epi4Ab/tools/bin:$PATH"
source /leonardo_work/AIFAC_F01_302/Epi4Ab/venv/bin/activate
```

## Troubleshooting
- **FreeSASA build errors**: ensure compiler toolchain (gcc) and Python headers are available. On Leonardo login nodes this is typically satisfied; otherwise load the appropriate module.
- **Import errors on compute nodes**: activate the same virtual environment inside batch scripts (`source venv/bin/activate`).
- **Legacy caches**: older runs may contain `naccess_cache/`. The pipeline now writes to `rsa_cache/` but still cleans the legacy directory automatically.

