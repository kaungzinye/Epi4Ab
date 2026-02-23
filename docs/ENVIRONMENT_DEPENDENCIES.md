# Environment Dependencies (Cross-Checked)

This repo uses a Python 3.11 virtualenv (`venv/`) on Leonardo.

## What the code actually imports/uses

Core ML + graph:
- `torch==2.5.1`
- `torch_geometric==2.5.1`

Data / IO:
- `pandas==2.0.3`
- `pyarrow==22.0.0` (parquet)
- `fastparquet==2024.11.0` (parquet; used in `preprocess/nodes_edges.py`)

Preprocessing:
- `biopython==1.83` (PDB parsing)
- `pdb2pqr==3.6.2` + `propka==3.5.1`
- `MDAnalysis==2.7.0`

NLP embeddings:
- `transformers==4.42.0`
- `fair-esm==2.0.0` (ESM models)
- `antiberty==0.1.3`

Plotting (optional for training/inference, required for dashboards):
- `matplotlib==3.7.5`
- `seaborn==0.13.2`
- `plotly==6.5.0`

## Pinned snapshot from a working venv

See `requirements.lock.txt` (generated from `pip freeze`).

## Non-Python requirements

- PyMOL executable (used by `preprocess/nodes_edges.py`): set `PYMOL_EXECUTABLE` (see `.env` patterns and SLURM scripts).
- Internet access is not available on compute nodes:
  - If `use_pretrained` is enabled, HF/ESM weights must be cached ahead of time.
  - Set `TRANSFORMERS_OFFLINE=1` and `HF_HUB_OFFLINE=1` for safety.

## Docker / container note

Leonardo typically does not allow Docker on compute nodes.
If a container is required, use Apptainer/Singularity with an NVIDIA/CUDA-enabled base image.
