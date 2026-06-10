# Epi4Ab Pipeline Validation Quick Reference

Quick reference guide for using the Epi4Ab pipeline validation system.

## Overview

The validation system checks data integrity and alignment at each pipeline step to prevent errors and ensure consistency.

## Quick Start

### Validate All Steps for a Single PDB

```bash
python scripts/validate_pipeline.py \
    --pdb_id 4ywg_HLG \
    --nodes_edges_dir /leonardo_scratch/fast/EUHPC_D29_035/epi4ab/upstream_preprocess/nodes_edges \
    --processed_dir /leonardo_scratch/fast/EUHPC_D29_035/epi4ab/upstream_preprocess/processed_data \
    --step all
```

### Validate Specific Step

```bash
# Validate labels only
python scripts/validate_pipeline.py \
    --pdb_id 4ywg_HLG \
    --nodes_edges_dir /path/to/nodes_edges \
    --step labels

# Validate inference outputs
python scripts/validate_pipeline.py \
    --output_dir output_inference/2025-12-18_GNNResNet_1 \
    --inference_list inference_list_test3A.csv \
    --nodes_edges_dir /path/to/nodes_edges \
    --step inference_outputs
```

### Validate All PDBs from Metadata

```bash
python scripts/validate_pipeline.py \
    --metadata metadata.csv \
    --nodes_edges_dir /path/to/nodes_edges \
    --step all
```

## Validation Steps

### Step 1: nodes_edges.py (CB/CA)

Validates output after running `nodes_edges.py`:

- File existence and readability
- Required columns present (`resId`, `resShort`, `source`, `target`, `dist`, `qi*qj`)
- Data types correct
- Edge indices within valid range
- No NaN values in critical columns

```bash
python scripts/validate_pipeline.py \
    --pdb_id 4ywg_HLG \
    --nodes_edges_dir /path/to/nodes_edges \
    --step nodes_edges
```

### Step 2: fill_edge.py

Validates merged edge files:

- Merged files exist (`edge_index.parquet`, `edge_attribute_dist.parquet`, `edge_attribute_charge.parquet`)
- Row counts match across all edge files
- Edge indices valid

```bash
python scripts/validate_pipeline.py \
    --pdb_id 4ywg_HLG \
    --nodes_edges_dir /path/to/nodes_edges \
    --step fill_edge
```

### Step 3: Labels

Validates label file alignment:

This repository supports two label modes:

1) Legacy classification labels:
- `node_label_pi.parquet` exists
- Columns: `resId`, `isInterface`
- `resId` matches `node_feature.parquet` exactly

2) Regression labels (Seqitope / ProteinMPNN):
- Per-PDB label parquet exists (configured via CLI):
  - Phase 1: `proteinmpnn_scores.parquet` with `resId, score` (raw NLL)
  - Phase 2: `node_label_seqitope.parquet` with `resId, score` (score in `[0,1]`)
- `resId` matches `node_feature.parquet` exactly

```bash
python scripts/validate_pipeline.py \
    --pdb_id 4ywg_HLG \
    --nodes_edges_dir /path/to/nodes_edges \
    --step labels
```

### Step 4: Sequence Files

Validates sequence JSON files:

- `antigen_sequence.json`: Valid JSON, contains `pdb_sequence` key
- `cdr_sequence.json`: Valid JSON, contains all CDR keys (H1_seq, H2_seq, etc.)
- Sequence characters valid amino acid codes

```bash
python scripts/validate_pipeline.py \
    --pdb_id 4ywg_HLG \
    --processed_dir /path/to/processed_data \
    --step sequences
```

### Step 5: Inference Inputs

Validates all files required for inference:

- All required files exist (based on model config)
- File formats correct
- Alignment verified
- Data types correct

```bash
python scripts/validate_pipeline.py \
    --pdb_id 4ywg_HLG \
    --nodes_edges_dir /path/to/nodes_edges \
    --processed_dir /path/to/processed_data \
    --step inference_inputs \
    --config model_config.json
```

### Step 6: Inference Outputs

Validates inference results:

- Result files exist for all PDBs
- Results align with `node_feature.parquet`

Output columns depend on mode:

- Legacy classification output: `pred_label`, `prob.`, `score`
- Regression output: `pred_score` and optionally `true_score`; if sigmoid is used, `pred_prob` may be present

```bash
python scripts/validate_pipeline.py \
    --output_dir output_inference/2025-12-18_GNNResNet_1 \
    --inference_list inference_list_test3A.csv \
    --nodes_edges_dir /path/to/nodes_edges \
    --step inference_outputs \
    --config model_config.json
```

## Model Configuration

The validation system uses model configuration to determine which files are required:

```bash
# Parse model config
python scripts/parse_model_config.py \
    --model_dir /path/to/model \
    --output config.json
```

Configuration flags:
- `prediction`: If True, `node_label_pi.parquet` not required
- `use_pretrained`: If True, `antigen_sequence.json` required
- `use_antiberty`: If True, `cdr_sequence.json` required

## Common Issues and Fixes

### Issue: Missing `isInterface` Column

**Error**: `AttributeError: 'DataFrame' object has no attribute 'isInterface'`

**Fix**: Regenerate labels:
```bash
python scripts/generate_labels.py --pdb_id 4ywg_HLG --output_dir /path/to/nodes_edges
```

### Issue: Sequence Length Mismatch

**Error**: `AssertionError: size of x_seq X different with Y`

**Fix**: Regenerate `antigen_sequence.json` from `node_feature.parquet`:
```python
import pandas as pd
import json

df = pd.read_parquet('node_feature.parquet')
sequence = ''.join(df['resShort'].tolist())

with open('antigen_sequence.json', 'w') as f:
    json.dump({'pdb_sequence': sequence}, f)
```

### Issue: Edge Index Out of Range

**Error**: Edge indices reference non-existent nodes

**Fix**: Regenerate edges:
```bash
python preprocess/nodes_edges.py metadata.csv processed_data nodes_edges pymol_path CB
python preprocess/nodes_edges.py metadata.csv processed_data nodes_edges pymol_path CA
python preprocess/fill_edge.py metadata.csv nodes_edges
```

## Exit Codes

- `0`: All validations passed
- `1`: One or more validations failed

## Options

- `--strict`: Enable strict mode (treat warnings as errors)
- `--quiet`: Suppress verbose output
- `--config`: Path to model configuration JSON file

## Examples

### Test3A Dataset

```bash
# Validate all preprocessing steps
python scripts/validate_pipeline.py \
    --metadata /leonardo_scratch/fast/EUHPC_D29_035/epi4ab/upstream_preprocess/metadata_complete.csv \
    --nodes_edges_dir /leonardo_scratch/fast/EUHPC_D29_035/epi4ab/upstream_preprocess/nodes_edges \
    --processed_dir /leonardo_scratch/fast/EUHPC_D29_035/epi4ab/upstream_preprocess/processed_data \
    --step all
```

### EpiScan Dataset

```bash
# Validate inference outputs
python scripts/validate_pipeline.py \
    --output_dir output_inference/2025-12-18_GNNResNet_2 \
    --inference_list inference_list_episcan.csv \
    --nodes_edges_dir /leonardo_scratch/fast/EUHPC_D29_035/epi4ab/episcan_preprocess/nodes_edges \
    --step inference_outputs
```

## Integration with SLURM Scripts

Validation is automatically run after inference in SLURM scripts:
- `slurm/inference_test3A.sbatch`
- `slurm/inference_episcan.sbatch`

The validation runs automatically and reports any issues without stopping the pipeline.

## See Also

- [Pipeline Documentation](PIPELINE_DOCUMENTATION.md) - Complete pipeline documentation with diagrams
- [Validation Script](../scripts/validate_pipeline.py) - Full validation implementation
