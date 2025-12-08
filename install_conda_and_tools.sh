#!/bin/bash
# Install Miniconda and then PyMOL/MAFFT for upstream Epi4Ab pipeline
# Run this from: /leonardo_work/AIFAC_F01_302/Epi4Ab

set -euo pipefail

PROJ_DIR="/leonardo_work/AIFAC_F01_302/Epi4Ab"
CONDA_DIR="${PROJ_DIR}/tools/conda"
CONDA_ENV_NAME="epi4ab-upstream"

echo "=========================================="
echo "Installing Miniconda and PyMOL/MAFFT"
echo "=========================================="
echo "Project directory: ${PROJ_DIR}"
echo "Conda will be installed to: ${CONDA_DIR}"
echo "Environment name: ${CONDA_ENV_NAME}"
echo ""

# Step 1: Download and install Miniconda
if [ ! -d "${CONDA_DIR}" ]; then
    echo "Step 1: Installing Miniconda..."
    mkdir -p "${PROJ_DIR}/tools"
    cd "${PROJ_DIR}/tools"
    
    # Download Miniconda for Linux x86_64
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    
    # Install Miniconda (silent mode, to tools/conda)
    bash miniconda.sh -b -p "${CONDA_DIR}"
    
    # Initialize conda
    source "${CONDA_DIR}/etc/profile.d/conda.sh"
    conda init bash
    
    echo "✓ Miniconda installed to ${CONDA_DIR}"
else
    echo "✓ Miniconda already exists at ${CONDA_DIR}"
    source "${CONDA_DIR}/etc/profile.d/conda.sh"
fi

# Step 2: Create conda environment
echo ""
echo "Step 2: Creating conda environment '${CONDA_ENV_NAME}'..."
if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
    echo "  Environment ${CONDA_ENV_NAME} already exists, activating..."
    conda activate "${CONDA_ENV_NAME}"
else
    conda create -n "${CONDA_ENV_NAME}" python=3.11 -y
    conda activate "${CONDA_ENV_NAME}"
    echo "✓ Environment created and activated"
fi

# Step 3: Install PyMOL and MAFFT
echo ""
echo "Step 3: Installing PyMOL and MAFFT..."
conda install -c conda-forge pymol-open-source -y
conda install -c bioconda mafft -y

# Step 4: Verify installation
echo ""
echo "Step 4: Verifying installation..."
PYMOL_PATH=$(which pymol)
MAFFT_PATH=$(which mafft)

if [ -n "${PYMOL_PATH}" ] && [ -n "${MAFFT_PATH}" ]; then
    echo "✓ PyMOL found at: ${PYMOL_PATH}"
    echo "✓ MAFFT found at: ${MAFFT_PATH}"
    
    # Test versions
    pymol -c -Q -u <<< "quit" 2>/dev/null && echo "  PyMOL test: OK" || echo "  PyMOL test: Warning (may need X11)"
    mafft --version 2>&1 | head -1 && echo "  MAFFT test: OK"
    
    # Update .env template with paths
    echo ""
    echo "Step 5: Updating .env template with paths..."
    if [ -f "${PROJ_DIR}/.env.upstream_template" ]; then
        sed -i "s|PYMOL_EXECUTABLE=\"/path/to/pymol\"|PYMOL_EXECUTABLE=\"${PYMOL_PATH}\"|" "${PROJ_DIR}/.env.upstream_template"
        sed -i "s|MAFFT_EXECUTABLE=\"/path/to/mafft\"|MAFFT_EXECUTABLE=\"${MAFFT_PATH}\"|" "${PROJ_DIR}/.env.upstream_template"
        echo "✓ Updated .env.upstream_template with paths"
        echo ""
        echo "To use these paths, copy the template:"
        echo "  cp ${PROJ_DIR}/.env.upstream_template ${PROJ_DIR}/.env"
    fi
    
    echo ""
    echo "=========================================="
    echo "Installation Complete!"
    echo "=========================================="
    echo "PyMOL: ${PYMOL_PATH}"
    echo "MAFFT: ${MAFFT_PATH}"
    echo ""
    echo "To activate the environment in future sessions:"
    echo "  source ${CONDA_DIR}/etc/profile.d/conda.sh"
    echo "  conda activate ${CONDA_ENV_NAME}"
    echo ""
else
    echo "✗ Installation verification failed"
    exit 1
fi
