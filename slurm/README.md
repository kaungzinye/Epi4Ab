# Epi4Ab SLURM Job Scripts

This directory contains SLURM batch scripts for running the Epi4Ab epitope prediction pipeline on the Leonardo supercomputer.

---

## Production Scripts

### Data Processing

**`process_data.sbatch`** - Process PDB files into parquet format
- **Usage:** Standard data processing for all HER2 structures
- **Input:** PDB files from `/leonardo_scratch/fast/AIFAC_F01_302/epi4ab/pdb_files/`
- **Output:** Processed parquet files in `/leonardo_scratch/fast/AIFAC_F01_302/epi4ab/processed/`
- **Time:** ~4-5 minutes per structure
- **Resources:** 8 CPUs, 32GB RAM

**`REPROCESS_ALL.sbatch`** - Reprocess all 12 HER2 structures
- **Usage:** After bug fixes or pipeline updates
- **Features:** 
  - Backs up old processed data
  - Reprocesses all 12 structures
  - Validates output (50 columns check)
- **Time:** ~60 minutes total
- **Resources:** 8 CPUs, 32GB RAM

**`validate.sbatch`** - Comprehensive validation of processed data
- **Usage:** Run BEFORE inference to catch data issues
- **Features:**
  - Validates all 50 columns (including VH/VK/VLa families)
  - Checks data integrity (distances, angles, sequences)
  - Verifies model compatibility
  - Saves detailed JSON report
- **Time:** ~2-5 minutes
- **Resources:** 4 CPUs, 16GB RAM
- **⚠️ Important:** Always run this after processing!

### Inference

**`infer_single.sbatch`** - Run inference on single PDB
- **Usage:** Testing or single-structure prediction
- **Input:** Single PDB ID (1N8Z)
- **Output:** Predictions in `run_model_output/`
- **Time:** ~5 minutes
- **Resources:** 1 GPU (A100), 16GB RAM

**`infer_all.sbatch`** - Run batch inference on all 12 structures
- **Usage:** Production inference on complete dataset
- **Input:** All 12 HER2 PDB IDs
- **Output:** Predictions for each structure
- **Time:** ~50-60 minutes
- **Resources:** 1 GPU (A100), 16GB RAM

**`infer.sbatch`** - Original inference script
- **Usage:** Legacy script, use infer_single or infer_all instead
- **Status:** Kept for reference

### Training

**`train.sbatch`** - Train new Epi4Ab model
- **Usage:** Model training (not typically needed for inference)
- **Time:** Several hours
- **Resources:** 1 GPU, large memory

---

## Pipeline Flow (Overview)

```
Raw PDB → Processing → Validation → Inference → Results
   │           │            │           │           │
   │           ▼            ▼           ▼           ▼
   │    1. Structure Loading & Chain Selection
   │    2. Residue Deduplication
   │    3. Node Feature Extraction (50 columns)
   │    4. Graph Connectivity (10Å cutoff)
   │    5. Edge Attributes (distance + charge)
   │    6. Sequence Extraction (FASTA/API/PDB)
   │    7. CDR Extraction (H1/H2/H3, L1/L2/L3)
   │    8. Epitope Labels (CIPS method)
   │    9. File Output (parquet + JSON)
   │
   └───────────────────────────────────────────────→
              Processed Data (parquet files)
                        │
                        ├─ node_feature.parquet (50 cols)
                        ├─ edge_index.parquet
                        ├─ edge_attribute_dist.parquet
                        ├─ edge_attribute_charge.parquet
                        ├─ node_label_pi.parquet
                        └─ sequence/
                           ├─ antigen_sequence.json
                           └─ cdr_sequence.json
```

**Detailed Flow Diagram:** See `docs/PIPELINE_FLOW.md` for complete step-by-step breakdown with all labeled sub-steps.

---

## Quick Reference

### Submit Jobs
```bash
# Process all HER2 structures
sbatch slurm/process_data.sbatch

# Reprocess all (after bug fixes)
sbatch slurm/REPROCESS_ALL.sbatch

# Run inference on single structure
sbatch slurm/infer_single.sbatch

# Run inference on all structures
sbatch slurm/infer_all.sbatch
```

### Monitor Jobs
```bash
# Check job status
squeue -u $USER

# Watch job progress
watch squeue -u $USER

# Check specific job
squeue -j JOBID

# Check job output
tail -f logs/epi4ab-*-JOBID.out
```

### Cancel Jobs
```bash
# Cancel specific job
scancel JOBID

# Cancel all your jobs
scancel -u $USER
```

---

## File Structure

```
slurm/
├── process_data.sbatch        # Standard processing
├── REPROCESS_ALL.sbatch       # Reprocess all structures
├── validate.sbatch            # Comprehensive validation
├── infer_single.sbatch        # Single PDB inference
├── infer_all.sbatch           # Batch inference
├── infer.sbatch               # Legacy inference
├── train.sbatch               # Model training
├── README.md                  # This file
└── SLURM_shortcuts.md         # Quick command reference
```

---

## Common Workflows

### After Fixing a Bug in Processing Code
```bash
# 1. Reprocess all structures
sbatch slurm/REPROCESS_ALL.sbatch

# 2. Wait for completion (~60 min)
squeue -u $USER

# 3. Validate processed data
sbatch slurm/validate.sbatch
# Or run interactively:
python validate_all.py --processed_dir /leonardo_scratch/fast/AIFAC_F01_302/epi4ab/processed/

# 4. Run inference
sbatch slurm/infer_all.sbatch
```

### Testing on Single Structure
```bash
# 1. Process single PDB (if needed)
python data_processing/epi4ab_pipeline.py \
    --pdb_file /leonardo_scratch/fast/AIFAC_F01_302/epi4ab/pdb_files/1N8Z.pdb \
    --output_dir /leonardo_scratch/fast/AIFAC_F01_302/epi4ab/processed/1N8Z \
    --antigen_chain A --sequence_source auto

# 2. Validate
python validate_all.py \
    --processed_dir /leonardo_scratch/fast/AIFAC_F01_302/epi4ab/processed/ \
    --pdb_id 1N8Z \
    --verbose

# 3. Run inference
sbatch slurm/infer_single.sbatch
```

### Full Production Run
```bash
# 1. Ensure data is processed
sbatch slurm/REPROCESS_ALL.sbatch
# Wait for completion...

# 2. Validate processed data
sbatch slurm/validate.sbatch
# Wait for completion (~5 min)

# 3. Run batch inference
sbatch slurm/infer_all.sbatch
# Wait for completion (~60 min)

# 4. Check results
ls /leonardo_scratch/fast/AIFAC_F01_302/epi4ab/run_model_output/
```

---

## Important Paths

| Purpose | Path |
|---------|------|
| PDB Files | `/leonardo_scratch/fast/AIFAC_F01_302/epi4ab/pdb_files/` |
| Processed Data | `/leonardo_scratch/fast/AIFAC_F01_302/epi4ab/processed/` |
| Model Weights | `/leonardo_work/AIFAC_F01_302/Epi4Ab/final_trained_Epi4Ab/` |
| Inference Output | `/leonardo_scratch/fast/AIFAC_F01_302/epi4ab/run_model_output/` |
| Job Logs | `/leonardo_work/AIFAC_F01_302/Epi4Ab/logs/` |

---

## Resource Requirements

| Job Type | CPUs | Memory | GPU | Typical Time |
|----------|------|--------|-----|--------------|
| Processing (single) | 8 | 32GB | - | 4-5 min |
| Processing (all 12) | 8 | 32GB | - | 50-60 min |
| Inference (single) | 4 | 16GB | 1x A100 | 4-5 min |
| Inference (all 12) | 4 | 16GB | 1x A100 | 50-60 min |
| Training | 8 | 64GB | 1x A100 | Hours |

---

## Troubleshooting

### Job Fails Immediately
- Check error log: `cat logs/epi4ab-*-JOBID.err`
- Common issues:
  - Module load failures → Check python/3.11.7 is available
  - Virtual env not found → Check venv path
  - Permission errors → Check file permissions

### Job Runs But Produces No Output
- Check stdout log: `cat logs/epi4ab-*-JOBID.out`
- Verify input files exist
- Check processed data completeness: `python validate_all.py --processed_dir /leonardo_scratch/fast/AIFAC_F01_302/epi4ab/processed/ --verbose`

### Inference Fails with Assertion Error
- Usually means processed data is incomplete
- Reprocess with: `sbatch slurm/REPROCESS_ALL.sbatch`
- Validate before inference

### Out of Memory
- Increase `--mem` in SLURM script
- Check if multiple jobs running on same node
- Consider using larger partition

---

## Version History

### v1.0 (Oct 23, 2025)
- Initial production scripts
- Added REPROCESS_ALL for bug fixes
- Unified inference scripts (single and batch)
- Added comprehensive validation

---

## Notes

1. **Always validate after processing**: Run `sbatch slurm/validate.sbatch` or `python validate_all.py` to ensure all 50 columns present (including VH/VK/VLa families)
2. **Validation is CRITICAL**: Catches missing columns (Bug #4, #5), edge attribute issues (Bug #2), missing sequences (Bug #3)
3. **Backup before reprocessing**: REPROCESS_ALL automatically backs up old data with timestamps
4. **Monitor GPU usage**: Use `nvidia-smi` on compute node to check utilization
5. **Check logs regularly**: Logs are in `logs/` directory with job ID in filename

---

**Last Updated:** October 29, 2025  
**Maintainer:** Epi4Ab Pipeline Team

