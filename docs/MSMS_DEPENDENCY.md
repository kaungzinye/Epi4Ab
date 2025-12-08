# MSMS Dependency in Upstream Pipeline

## What is MSMS?

**MSMS (Michel Sanner's Molecular Surface)** is a computational tool for generating and triangulating molecular surfaces. It's used to:

- Calculate the **solvent-accessible surface** of biomolecules (proteins, nucleic acids)
- Generate **triangulated surface meshes** for visualization and analysis
- Compute **molecular surface properties** for structural analysis

### Key Details:
- **Full Name**: Michel Sanner's Molecular Surface
- **Purpose**: Molecular surface generation and triangulation
- **Developed by**: Michel Sanner (Scripps Research Institute)
- **Website**: https://ccsb.scripps.edu/msms/
- **Platform**: Linux, macOS, Windows

## Why is MSMS Needed?

In the upstream Epi4Ab pipeline, MSMS is required by **BioPython's `ResidueDepth`** class to calculate:

- **Residue Depth**: The distance from a residue to the molecular surface
- **CA Depth**: The distance from the C-alpha atom to the molecular surface

These depth measurements are used as **structural features** in the machine learning model.

### Code Location:
```python
# preprocess/scripts/extract_depth.py
from Bio.PDB.ResidueDepth import ResidueDepth

rd = ResidueDepth(chain)  # Requires MSMS executable
```

## The Problem

**MSMS is NOT installed** on the HPC system, causing the pipeline to fail at the `extract_depth` step:

```
RuntimeError: Failed to generate surface file using command:
```

### Why MSMS is Hard to Install on HPC:

1. **Binary Executable Required**: MSMS is a compiled C++ program, not a Python package
2. **System Dependencies**: Requires specific system libraries and compilation tools
3. **No Standard Package Manager**: Not available via `pip` or `conda` easily
4. **Platform-Specific**: Must be compiled for the specific HPC architecture
5. **License/Download**: Requires manual download and compilation from source

## Alternative Solutions

### 1. Your Fork's Solution (FreeSASA)
Your fork uses **FreeSASA** instead of MSMS:
- ✅ Available via `pip install freesasa`
- ✅ Pure Python/C library, easier to install
- ✅ Calculates **Relative Solvent Accessibility (RSA)** instead of depth
- ✅ Works well on HPC systems
NOtes from inout:
INstall one package htat allows to run it and run it afterwards
### 2. Other Alternatives
- **DSSP**: Another tool for surface/accessibility calculations
- **NACCESS**: Similar to MSMS but also requires compilation
- **PyMOL**: Has built-in surface calculation (but slower)

## Installation Options (if needed)

If you want to install MSMS for the upstream pipeline:

### Option 1: Download and Compile
```bash
# Download from Scripps Research
wget https://ccsb.scripps.edu/msms/downloads/msms_i86_64Linux2.2.6.1.tar.gz
tar -xzf msms_i86_64Linux2.2.6.1.tar.gz
# Add to PATH or specify full path in code
```

### Option 2: Use System Module (if available)
```bash
module load msms  # Check if available on your HPC
```

### Option 3: Container/Singularity
Package MSMS in a container image.

## Current Status

- **Upstream Pipeline**: ❌ Blocked by missing MSMS
- **Your Fork**: ✅ Uses FreeSASA (no MSMS needed)

## Recommendation

Since MSMS is difficult to install and your fork already uses FreeSASA (which is more HPC-friendly), you have two options:

1. **Document the limitation**: Note that upstream requires MSMS, which is not available
2. **Skip extract_depth**: Modify upstream to skip this step or use an alternative
3. **Use your fork**: Your implementation with FreeSASA is more practical for HPC

## References

- MSMS Official Site: https://ccsb.scripps.edu/msms/
- BioPython ResidueDepth: https://biopython.org/docs/latest/api/Bio.PDB.ResidueDepth.html
- FreeSASA: https://freesasa.github.io/

