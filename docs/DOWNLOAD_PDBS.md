# Downloading PDB Files for Upstream Pipeline

## Overview

The upstream Epi4Ab preprocessing pipeline requires PDB (Protein Data Bank) files to be downloaded before running on compute nodes. Since compute nodes typically don't have internet access, PDB files must be downloaded on a **login node** first.

## Quick Start

1. **Activate your Python environment** (on login node):
   ```bash
   cd /leonardo_work/AIFAC_F01_302/Epi4Ab
   source venv/bin/activate  # or your conda environment
   ```

2. **Run the download script**:
   ```bash
   python scripts/download_pdbs_login_node.py
   ```

3. **Verify downloads**:
   ```bash
   ls -lh ${DIRECTORY_PROCESSED}/*/*.cif
   ```

## Script Details

### `scripts/download_pdbs_login_node.py`

This script:
- Reads PDB information from `input/pdb_info.csv`
- Downloads PDB files from RCSB in mmCIF format
- Saves files to `DIRECTORY_PROCESSED/{pdb_id}/{pdb}.cif`
- Skips files that already exist
- Reports download status and errors

### Expected File Structure

After downloading, the structure should be:
```
${DIRECTORY_PROCESSED}/
├── 1n8z_BAC/
│   └── 1n8z.cif
├── 1ahw_BAC/
│   └── 1ahw.cif
└── ...
```

Where:
- `{pdb_id}` comes from the `pdbID` column in `pdb_info.csv` (e.g., "1n8z_BAC")
- `{pdb}.cif` comes from the `pdb` column in `pdb_info.csv` (e.g., "1n8z")

## Environment Variables

The script uses these variables from `.env`:
- `DIRECTORY_PDB_INFO`: Path to `input/pdb_info.csv`
- `DIRECTORY_PROCESSED`: Base directory for processed data (where PDB files are stored)

## Troubleshooting

### Error: "Network is unreachable"
- **Cause**: Running on a compute node instead of login node
- **Solution**: Run the script on a login node (where you have internet access)

### Error: "DIRECTORY_PDB_INFO not set"
- **Cause**: `.env` file missing or variable not set
- **Solution**: Ensure `.env` file exists and contains `DIRECTORY_PDB_INFO`

### Error: "Failed to download {pdb_id}"
- **Cause**: PDB code might be invalid or RCSB server issue
- **Solution**: 
  - Verify PDB code is correct in `pdb_info.csv`
  - Check internet connection
  - Try downloading manually: `wget https://files.rcsb.org/download/{pdb}.cif`

### Files already exist
- The script will skip files that already exist
- To re-download, delete the existing `.cif` file first

## Integration with Pipeline

After downloading PDB files:

1. **Run preprocessing on compute nodes**:
   ```bash
   sbatch slurm/upstream_preprocess.sbatch
   ```

2. The pipeline will:
   - Use existing `.cif` files (no download needed)
   - Extract `lig.pdb` from `.cif` files
   - Process features and create parquet files

## Manual Download (Alternative)

If the script doesn't work, you can download manually:

```bash
# For each PDB in pdb_info.csv
PDB_CODE="1n8z"  # from 'pdb' column
PDB_ID="1n8z_BAC"  # from 'pdbID' column
OUTPUT_DIR="${DIRECTORY_PROCESSED}/${PDB_ID}"
mkdir -p "${OUTPUT_DIR}"
wget -O "${OUTPUT_DIR}/${PDB_CODE}.cif" "https://files.rcsb.org/download/${PDB_CODE}.cif"
```

## Notes

- PDB files are downloaded in **mmCIF format** (`.cif` extension)
- The upstream pipeline expects files at: `{DIRECTORY_PROCESSED}/{pdb_id}/{pdb}.cif`
- Downloads are typically fast (few seconds per PDB)
- Total size depends on number of PDBs (typically 100KB-1MB per file)

