# Installing PyMOL and MAFFT for Upstream Epi4Ab Pipeline

## Current Status
PyMOL and MAFFT are **NOT currently installed** on Leonardo HPC. They are required for the upstream preprocessing pipeline (`run_preprocess.sh`).

## Installation Options

### Option 1: Conda Installation (Recommended for HPC)

```bash
# Activate your conda environment or create a new one
conda create -n epi4ab-upstream python=3.11
conda activate epi4ab-upstream

# Install PyMOL (open-source version)
conda install -c conda-forge pymol-open-source

# Install MAFFT
conda install -c bioconda mafft

# Find the executable paths
conda info --base
# Paths will be: <conda_base>/envs/epi4ab-upstream/bin/pymol
#                <conda_base>/envs/epi4ab-upstream/bin/mafft
```

### Option 2: Install in Project venv (if compatible)

```bash
cd /leonardo_work/AIFAC_F01_302/Epi4Ab
source venv/bin/activate

# Try installing PyMOL (may fail on HPC due to dependencies)
pip install pymol-open-source

# MAFFT Python wrapper (still needs system MAFFT binary)
pip install mafft-python
```

### Option 3: Request System Installation

Contact Leonardo HPC support to request PyMOL and MAFFT as system modules:
- Email: support@cineca.it
- Request: Add PyMOL and MAFFT to available modules

### Option 4: Download and Compile (Advanced)

**PyMOL:**
```bash
# Download from: https://github.com/schrodinger/pymol-open-source
# Follow build instructions for Linux
# Install to: /leonardo_work/AIFAC_F01_302/Epi4Ab/tools/pymol/
```

**MAFFT:**
```bash
# Download from: https://mafft.cbrc.jp/alignment/software/
# Extract and compile
# Install to: /leonardo_work/AIFAC_F01_302/Epi4Ab/tools/mafft/
```

## After Installation

1. Find the executable paths:
   ```bash
   which pymol
   which mafft
   ```

2. Update `.env` file:
   ```bash
   PYMOL_EXECUTABLE="/path/to/pymol"
   MAFFT_EXECUTABLE="/path/to/mafft"
   ```

3. Test the executables:
   ```bash
   pymol --version
   mafft --version
   ```

## Alternative: Skip Upstream Pipeline

If PyMOL/MAFFT installation is difficult, you can:
- Continue using your **fork pipeline** (`data_processing/epi4ab_pipeline.py`) which doesn't require PyMOL/MAFFT
- Your fork uses MDAnalysis for edge construction (faster and no external dependencies)
