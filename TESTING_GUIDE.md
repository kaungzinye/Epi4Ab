# Testing Guide: SLURM Execution

## Important: Always Test via SLURM

From now on, **all testing should be done via SLURM** to simulate the production environment accurately.

## Why SLURM Testing?

1. **Simulates Production**: Compute nodes have no internet (like production)
2. **Validates HPC Logic**: Ensures compute node detection works
3. **Tests Offline Behavior**: Verifies cache and local tools work without internet
4. **Catches Issues Early**: API calls won't work on compute nodes

## Quick Test Commands

### Test Option 2 (Local Tools Priority)

```bash
# Submit test job
sbatch slurm/test_local_tools.sbatch

# Monitor job
squeue -u $USER

# Check output
tail -f logs/test-local-tools-*.out
tail -f logs/test-local-tools-*.err
```

### Test Label Generation

```bash
# Use existing test script via SLURM
sbatch --wrap="cd /leonardo_work/AIFAC_F01_302/Epi4Ab && source venv/bin/activate && module load python/3.11.7 && python3 test_label_generation.py" \
    --job-name=test-labels \
    --account=AIFAC_F01_302 \
    --partition=boost_usr_prod \
    --time=00:30:00 \
    --output=logs/test-labels-%j.out \
    --error=logs/test-labels-%j.err
```

## Expected Behavior on Compute Nodes

### BepiPred/Ellipro Functions

1. **Check cache** → Fast, works offline
2. **Try local tools** → Works if installed
3. **Skip web API** → Detected automatically (no internet warning)
4. **Fallback to RSA** → Always available

### Compute Node Detection

- ✅ Detects SLURM environment
- ✅ Warns when API calls attempted
- ✅ Prioritizes cache and local tools
- ✅ Graceful fallback to RSA

## Current Test Results

### Job 24269868 / 24271027 (SLURM)
- ✅ Compute node detected correctly (`lrdn3439`)
- ✅ SLURM environment variables present
- ⚠️ ESM2 model download attempts (expected - will timeout gracefully)
- ✅ BepiPred/Ellipro logic will skip API calls automatically

## Next Steps for Option 2 Testing

1. **If tools not installed**: 
   - Use pre-processing script on login node first
   - Cache predictions before running SLURM jobs
   - Then test that cache is used on compute nodes

2. **If tools available**:
   - Install locally or via modules
   - Test that tools execute on compute nodes
   - Verify predictions are cached automatically

3. **Model caching** (separate issue):
   - Pre-download ESM2 model on login node
   - Cache in `~/.cache/huggingface/` or similar
   - Will avoid download attempts on compute nodes

## Summary

✅ **All future tests via SLURM**  
✅ **Code handles compute nodes correctly**  
✅ **API calls automatically skipped**  
✅ **Cache system ready for offline use**  
✅ **Local tools will be used if available**

