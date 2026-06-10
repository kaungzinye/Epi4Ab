# Pipeline Comparison: Fork vs Upstream/Main

## Executive Summary

This document provides a comprehensive comparison between:
- **Fork Implementation** (`fork/her2-selection-and-visualization`): Your custom pipeline with FreeSASA/PDB2PQR, HPC optimizations, and bug fixes
- **Upstream/Main Implementation** (`upstream/master`): The official Epi4Ab pipeline from the original authors

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Key Differences](#key-differences)
3. [Dependencies](#dependencies)
4. [Feature Extraction](#feature-extraction)
5. [HPC Compatibility](#hpc-compatibility)
6. [Bug Fixes](#bug-fixes)
7. [Output Format](#output-format)
8. [Visualization](#visualization)
9. [Migration Guide](#migration-guide)

---

## Architecture Overview

### Fork Implementation

```
data_processing/
├── epi4ab_pipeline.py          # Main pipeline orchestrator
│   ├── calculate_rsa()         # FreeSASA-based RSA calculation
│   ├── calculate_partial_charges()  # PDB2PQR-based charge calculation
│   └── HPC-aware caching        # Cache management for compute nodes
└── sequence_utils.py            # Sequence processing utilities

interface_prediction/
└── data_function/
    └── data_function.py         # Fixed reshape(-1) bug
```

**Key Characteristics:**
- ✅ HPC-optimized with caching
- ✅ FreeSASA for RSA (no MSMS dependency)
- ✅ PDB2PQR for charges
- ✅ Comprehensive error handling
- ✅ Network-aware (handles compute node limitations)

### Upstream/Main Implementation

```
preprocess/
├── main.py                      # Main preprocessing orchestrator
├── create_metadata.py           # Metadata generation
├── nodes_edges.py               # Node/edge construction (uses PyMOL)
├── fill_edge.py                 # Edge filling
└── scripts/
    ├── extract_depth.py         # MSMS-based depth extraction (original)
    └── extract_depth_freesasa.py  # FreeSASA replacement (added)

source_code/
└── data_function/
    └── data_function.py         # Original reshape(-1) bug present
```

**Key Characteristics:**
- ⚠️ Requires MSMS (or FreeSASA replacement)
- ⚠️ Requires PyMOL for edge construction
- ⚠️ Requires MAFFT for CDR alignment
- ⚠️ Network access needed for PDB downloads
- ✅ Modular preprocessing pipeline
- ✅ Comprehensive feature extraction

---

## Key Differences

### 1. RSA/SASA Calculation

| Aspect | Fork | Upstream |
|--------|------|----------|
| **Tool** | FreeSASA (Python library) | MSMS (via BioPython ResidueDepth) |
| **Installation** | `pip install freesasa` | Requires compiled binary |
| **HPC Compatibility** | ✅ Excellent | ❌ Difficult (binary compilation) |
| **Output** | RSA (Relative Solvent Accessibility) | Depth (distance to surface) |
| **Performance** | Fast, pure Python/C | Requires external executable |

**Fork Implementation:**
```python
# data_processing/epi4ab_pipeline.py
def calculate_rsa(pdb_file: str, chain_id: str) -> pd.DataFrame:
    """Calculate RSA using FreeSASA with caching."""
    # Uses freesasa library
    # Implements caching for HPC efficiency
    # Returns RSA values per residue
```

**Upstream Implementation:**
```python
# preprocess/scripts/extract_depth.py (original)
from Bio.PDB.ResidueDepth import ResidueDepth
rd = ResidueDepth(chain)  # Requires MSMS executable
# Returns depth (distance to surface)
```

**Current Status:**
- Upstream has been adapted to use `extract_depth_freesasa.py` (FreeSASA-based) to avoid MSMS dependency

### 2. Charge Calculation

| Aspect | Fork | Upstream |
|--------|------|----------|
| **Tool** | PDB2PQR | PDB2PQR |
| **Integration** | Direct Python API | Command-line execution |
| **Caching** | ✅ Implemented | ❌ No caching |
| **Error Handling** | ✅ Comprehensive | ⚠️ Basic |

**Fork Implementation:**
```python
# data_processing/epi4ab_pipeline.py
def calculate_partial_charges(pdb_file: str) -> pd.DataFrame:
    """Calculate charges using PDB2PQR with caching."""
    # Uses pdb2pqr Python API
    # Implements result caching
    # Handles errors gracefully
```

**Upstream Implementation:**
```python
# preprocess/main.py
os.system(f'pdb2pqr {pdb_file} {output_file}')
# No caching, direct command execution
```

### 3. Edge Construction

| Aspect | Fork | Upstream |
|--------|------|----------|
| **Tool** | NetworkX (Python) | PyMOL (external executable) |
| **Dependency** | Python library | External binary |
| **HPC Compatibility** | ✅ Excellent | ⚠️ Requires PyMOL installation |
| **Performance** | Fast (in-memory) | Slower (process spawning) |

**Fork Implementation:**
- Uses NetworkX for graph construction
- Pure Python, no external dependencies

**Upstream Implementation:**
- Uses PyMOL for edge construction
- Requires PyMOL executable in PATH
- More complex but potentially more accurate

### 4. CDR Alignment

| Aspect | Fork | Upstream |
|--------|------|----------|
| **Tool** | Built-in Python | MAFFT (external) |
| **Dependency** | None | MAFFT executable |
| **HPC Compatibility** | ✅ Excellent | ⚠️ Requires MAFFT installation |

**Upstream Implementation:**
- Uses MAFFT for multiple sequence alignment
- Required for CDR sequence processing

---

## Dependencies

### Fork Dependencies

```python
# Minimal external dependencies
- freesasa          # RSA calculation
- pdb2pqr            # Charge calculation
- networkx           # Graph construction
- pandas, numpy      # Data processing
```

**Installation:**
```bash
pip install freesasa pdb2pqr networkx
```

### Upstream Dependencies

```python
# More complex dependencies
- MSMS (or FreeSASA replacement)  # Depth calculation
- PyMOL                            # Edge construction
- MAFFT                            # CDR alignment
- PDB2PQR                          # Charge calculation
- BioPython                        # Structure parsing
```

**Installation:**
```bash
# Requires system-level tools
conda install -c conda-forge pymol mafft
pip install freesasa pdb2pqr biopython
# MSMS requires manual compilation
```

---

## Feature Extraction

### Node Features

Both implementations extract similar node features, but with differences:

| Feature | Fork | Upstream | Notes |
|---------|------|----------|-------|
| **RSA** | ✅ FreeSASA | ⚠️ MSMS (or FreeSASA) | Different calculation method |
| **Charge** | ✅ PDB2PQR | ✅ PDB2PQR | Same tool, different integration |
| **Depth** | ❌ Not included | ✅ Included | Upstream calculates depth |
| **Dihedral Angles** | ✅ Yes | ✅ Yes | Same calculation |
| **Amino Acid Composition** | ✅ Yes | ✅ Yes | Same calculation |
| **IMGT Gene Families** | ✅ Yes | ✅ Yes | Same calculation |

### Edge Features

| Feature | Fork | Upstream | Notes |
|---------|------|----------|-------|
| **Distance** | ✅ Yes | ✅ Yes | Same calculation |
| **Charge Interaction** | ✅ Yes | ✅ Yes | Same calculation |
| **Construction Method** | NetworkX | PyMOL | Different tools |

---

## HPC Compatibility

### Fork: HPC-Optimized

**Strengths:**
- ✅ **Caching**: Implements result caching to avoid redundant computations
- ✅ **Network Awareness**: Detects compute nodes and handles network limitations
- ✅ **Pure Python**: Minimal external dependencies
- ✅ **Error Recovery**: Comprehensive error handling and retry logic
- ✅ **Resource Efficiency**: Optimized for batch processing

**Example:**
```python
# data_processing/epi4ab_pipeline.py
def _is_compute_node() -> bool:
    """Detect if running on compute node."""
    # Checks for SLURM environment variables
    return 'SLURM_JOB_ID' in os.environ

def _initialize_rsa_cache(cache_dir: str):
    """Initialize RSA cache for HPC efficiency."""
    # Creates cache directory structure
    # Implements cache lookup and storage
```

### Upstream: HPC Challenges

**Challenges:**
- ⚠️ **MSMS Dependency**: Requires compiled binary (difficult on HPC)
- ⚠️ **Network Access**: PDB downloads require internet (compute nodes may not have access)
- ⚠️ **External Tools**: PyMOL, MAFFT require installation
- ⚠️ **No Caching**: Redundant computations for repeated runs

**Solutions Applied:**
- ✅ Replaced MSMS with FreeSASA (`extract_depth_freesasa.py`)
- ✅ Created `download_pdbs_login_node.py` for pre-downloading PDBs
- ✅ Installed PyMOL and MAFFT via conda
- ✅ Fixed syntax errors in `fill_edge.py`

---

## Bug Fixes

### 1. Critical Edge Attribute Bug (Fork Only)

**Location:** `source_code/data_function/data_function.py` (lines 81-82)

**Bug in Upstream:**
```python
# Upstream (BUGGY)
attribute = torch.tensor(edgeAttribute.to_numpy().reshape(-1), dtype = torch.float)
attributeCharge = torch.tensor(edgeCharge.to_numpy().reshape(-1), dtype = torch.int)
```

**Problem:**
- `edgeAttribute` is a DataFrame with columns `['dist']`
- `reshape(-1)` flattens ALL columns, not just the distance values
- Causes shape mismatch and incorrect data

**Fix in Fork:**
```python
# Fork (FIXED)
attribute = torch.tensor(edgeAttribute['dist'].to_numpy().reshape(-1), dtype = torch.float)
attributeCharge = torch.tensor(edgeCharge['charge'].to_numpy().reshape(-1), dtype = torch.int)
```

**Impact:**
- ✅ Prevents assertion errors: "Edge Attribute - Bias_distance should be greater than 0"
- ✅ Ensures correct tensor shapes for model training/inference
- ⚠️ **Status**: Upstream still has this bug (not fixed by authors)

### 2. Syntax Error in fill_edge.py (Fixed in Upstream)

**Location:** `preprocess/fill_edge.py`

**Bug:**
```python
# Original (BUGGY)
os.system(f'rm {os.path.join(pdb_path, '*_CA*')} {os.path.join(pdb_path, '*_CB*')} ...')
# SyntaxError: f-string expression part cannot include a backslash
```

**Fix:**
```python
# Fixed
rm_patterns = [
    os.path.join(pdb_path, '*_CA*'),
    os.path.join(pdb_path, '*_CB*'),
    os.path.join(pdb_path, '*-CA_*'),
    os.path.join(pdb_path, '*-CB_*')
]
os.system('rm ' + ' '.join(rm_patterns))
```

### 3. Model Loading on CPU (Fixed in Upstream)

**Location:** `epi_prediction.py`

**Bug:**
```python
# Original (BUGGY)
model.load_state_dict(torch.load('model.pt'))
# RuntimeError: Attempting to deserialize object on a CUDA device but torch.cuda.is_available() is False
```

**Fix:**
```python
# Fixed
map_location = 'cpu' if not torch.cuda.is_available() else None
model.load_state_dict(torch.load('model.pt', map_location=map_location, weights_only=True))
```

---

## Output Format

### Fork Output Structure

```
{PDB_ID}/
├── node_feature.parquet         # 56 columns (missing some upstream features)
├── node_label_pi.parquet        # Labels (legacy classification)
├── proteinmpnn_scores.parquet   # Labels (Phase 1 regression; raw NLL)
├── node_label_seqitope.parquet  # Labels (Phase 2 regression; score in [0,1])
├── edge_index.parquet           # Edge connections
├── edge_attribute_dist.parquet  # Distance attributes
├── edge_attribute_charge.parquet # Charge attributes
└── sequence/
    ├── antigen_sequence.json
    └── cdr_sequence.json
```

### Upstream Output Structure

```
{PDB_ID}/
├── nodes_edges/
│   ├── node_feature.parquet     # 68 columns (includes depth)
│   ├── node_label_pi.parquet   # Labels
│   ├── node_list.parquet       # Residue-to-chain mapping
│   ├── edge_index.parquet      # Edge connections
│   ├── edge_attribute_dist.parquet  # Distance (shape: N_edges, 1)
│   └── edge_attribute_charge.parquet # Charge (shape: N_edges, 1)
└── preprocess/
    ├── aac/                     # Amino acid composition
    ├── angle/                   # Dihedral angles
    ├── atc/                     # Charge
    ├── cdrs/                    # CDR information
    ├── cips/                    # CIPS interface files
    ├── depth/                   # Depth calculations
    ├── ellipro_bepiPred/        # Epitope predictions
    └── sequence/                # Sequence files
```

**Key Differences:**
- Upstream includes `node_list.parquet` (residue-to-chain mapping)
- Upstream includes `preprocess/` subdirectories (intermediate results)
- Upstream edge attributes have shape `(N_edges, 1)` vs fork's `(N_edges, 3)` with source/target

---

## Visualization

### Fork Visualization

**Tools:**
- `generate_visualizer.py` - Interactive HTML visualizations
- `interface_prediction/evaluation_and_plot/visualizer.py` - Plotly-based plots
- `scripts/review_predictions.py` - Comprehensive quality review

**Features:**
- ✅ Interactive probability plots
- ✅ Per-residue prediction tables
- ✅ Performance metrics (ROC curves, confusion matrices)
- ✅ 3D structure visualization (if py3Dmol available)
- ✅ Summary statistics

**Usage:**
```bash
python generate_visualizer.py --test_record_dir output_inference/.../test_record
```

### Upstream Visualization

**Tools:**
- `source_code/evaluation_and_plot/networkx.py` - NetworkX-based plots
- `source_code/evaluation_and_plot/result_evaluation.py` - Evaluation metrics

**Features:**
- ✅ Network graph visualization (ground truth vs prediction)
- ✅ Evaluation metrics tables
- ⚠️ Less interactive than fork's Plotly-based visualizations

**Usage:**
- Built into `epi_prediction.py` when `plot_network=True`

### Current Status

- ✅ `scripts/visualize_results.py` is now regression-first (Seqitope / ProteinMPNN workflows)
- ⚠️ The legacy classification dashboard is no longer the primary path

---

## Migration Guide

### From Fork to Upstream

If you want to use the upstream pipeline:

1. **Install Dependencies:**
   ```bash
   conda install -c conda-forge pymol mafft
   pip install freesasa pdb2pqr biopython
   ```

2. **Set Environment Variables:**
   ```bash
   # Copy .env.upstream_template to .env
   cp .env.upstream_template .env
   # Edit .env with your paths
   ```

3. **Download PDBs on Login Node:**
   ```bash
   python scripts/download_pdbs_login_node.py
   ```

4. **Run Preprocessing:**
   ```bash
   bash run_preprocess.sh
   ```

5. **Run Inference:**
   ```bash
   python epi_prediction.py
   ```

### From Upstream to Fork

If you want to use the fork's pipeline:

1. **Install Dependencies:**
   ```bash
   pip install freesasa pdb2pqr networkx
   ```

2. **Use Fork's Pipeline:**
   ```python
   from data_processing.epi4ab_pipeline import Epi4AbPipeline
   pipeline = Epi4AbPipeline()
   pipeline.process_pdb(pdb_file, chain_id)
   ```

3. **Benefits:**
   - HPC-optimized caching
   - No external tool dependencies (PyMOL, MAFFT)
   - Better error handling
   - Fixed reshape(-1) bug

---

## Recommendations

### For HPC Environments

**Use Fork Implementation:**
- ✅ Better HPC compatibility
- ✅ No external tool dependencies
- ✅ Comprehensive caching
- ✅ Fixed critical bugs

### For Standard Environments

**Use Upstream Implementation:**
- ✅ More comprehensive feature extraction (includes depth)
- ✅ Modular preprocessing pipeline
- ✅ Well-documented official implementation
- ⚠️ Requires more setup (PyMOL, MAFFT, MSMS replacement)

### Hybrid Approach

**Best of Both Worlds:**
1. Use upstream's preprocessing structure
2. Replace MSMS with FreeSASA (already done)
3. Apply fork's reshape(-1) bug fix
4. Add fork's caching mechanism
5. Use fork's visualization tools

---

## Summary Table

| Feature | Fork | Upstream | Winner |
|---------|------|----------|--------|
| **HPC Compatibility** | ✅ Excellent | ⚠️ Requires setup | Fork |
| **Dependencies** | ✅ Minimal | ⚠️ Multiple external tools | Fork |
| **Bug Fixes** | ✅ Fixed reshape(-1) | ❌ Bug still present | Fork |
| **Feature Completeness** | ⚠️ Missing depth | ✅ Includes depth | Upstream |
| **Caching** | ✅ Implemented | ❌ No caching | Fork |
| **Visualization** | ✅ Interactive (Plotly) | ⚠️ Basic (NetworkX) | Fork |
| **Modularity** | ⚠️ Monolithic | ✅ Modular | Upstream |
| **Documentation** | ⚠️ Limited | ✅ Comprehensive | Upstream |

---

## Conclusion

Both implementations have strengths:

- **Fork**: Better for HPC, fewer dependencies, bug fixes, excellent caching
- **Upstream**: More comprehensive features, modular design, official support

**Current Status:**
- Upstream has been adapted to work on HPC (FreeSASA replacement, PyMOL/MAFFT installation)
- Fork's visualization tools have been ported to upstream
- Critical reshape(-1) bug remains in upstream (should be fixed)

**Recommendation:**
- For production HPC use: **Fork implementation**
- For research/development: **Upstream implementation** (with adaptations)
- For best results: **Hybrid approach** combining strengths of both
