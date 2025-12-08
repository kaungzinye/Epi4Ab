# Migration Recommendation: Fork vs Upstream Pipeline

**Date**: 2025-01-XX  
**Context**: Comparison between fork implementation (`fork/her2-selection-and-visualization`) and upstream official pipeline (`upstream/master`)

## Recommendation: **Keep Fork as Primary, Selectively Adopt Upstream**

### Why Keep the Fork?

1. **HPC Environment Requirements**
   - ✅ Fork has compute node detection and offline support
   - ✅ Comprehensive caching system (essential for compute nodes)
   - ❌ Upstream assumes internet access (won't work on compute nodes)

2. **Critical Missing Features in Upstream**
   - ✅ **FreeSASA/RSA**: Required for `sasa` column (68-column format)
   - ✅ **ESM2 Embeddings**: Integrated in fork
   - ✅ **Vectorized Edge Construction**: Much faster than upstream's PyMOL approach
   - ❌ Upstream lacks all of these

3. **Working Infrastructure**
   - ✅ SLURM scripts already configured (`slurm/process_data.sbatch`)
   - ✅ HER2-specific workflows established
   - ✅ Cache directories and verification scripts in place

### What to Adopt from Upstream?

#### ✅ Safe to Adopt (Non-Conflicting)

1. **README Updates**
   - Upstream has updated documentation
   - **Action**: Manually review and cherry-pick useful README sections

2. **Example Data**
   - Upstream has `examples/` folder with sample outputs
   - **Action**: Review for format validation, but don't overwrite your structure

3. **Bug Fixes in Core Scripts**
   - If upstream fixes bugs in shared dependencies
   - **Action**: Cherry-pick specific commits if needed

#### ⚠️ Requires Careful Integration

1. **Preprocess Scripts Structure**
   - Upstream uses modular `preprocess/scripts/` approach
   - **Action**: Keep as reference, but don't replace your unified pipeline
   - **Use Case**: If you need to customize individual steps, upstream scripts are good templates

2. **Feature Extraction Methods**
   - Upstream's dihedral angle calculation (MDAnalysis native) vs fork's custom
   - **Action**: Consider testing upstream's method for accuracy, but keep fork's if it works better

#### ❌ Don't Adopt (Would Break Your Workflow)

1. **Replace `data_processing/epi4ab_pipeline.py`**
   - This would lose all HPC support, caching, FreeSASA, ESM2
   - **Action**: Keep your fork implementation

2. **Remove Caching System**
   - Upstream has no caching
   - **Action**: Keep your caching (essential for compute nodes)

3. **Switch to PyMOL Edge Construction**
   - Upstream uses slow PyMOL scripts
   - **Action**: Keep your fast vectorized MDAnalysis approach

## Recommended Workflow

### Option A: Continue with Fork (Recommended)

**Best for**: Current HER2 research, HPC environment, production use

```bash
# Stay on fork branch
git checkout fork/her2-selection-and-visualization

# Continue using your pipeline
sbatch slurm/process_data.sbatch

# Periodically check upstream for documentation updates
git fetch upstream
git log upstream/master --oneline --since="1 month ago"
```

**Pros**:
- ✅ Everything works now
- ✅ HPC-optimized
- ✅ All features you need
- ✅ No migration risk

**Cons**:
- ⚠️ May miss upstream bug fixes (but you can cherry-pick)
- ⚠️ Not "official" version (but yours is more feature-complete)

### Option B: Hybrid Approach (Advanced)

**Best for**: Wanting to contribute back, long-term maintenance

1. **Keep fork as main working branch**
2. **Create integration branch** to test upstream improvements:
   ```bash
   git checkout -b integration/upstream-improvements
   git merge upstream/master --no-commit
   # Manually resolve conflicts, keeping fork enhancements
   ```

3. **Selectively merge upstream improvements**:
   - Documentation updates
   - Bug fixes in shared code
   - Example data formats

**Pros**:
- ✅ Get upstream improvements
- ✅ Keep your enhancements
- ✅ Can contribute back to upstream

**Cons**:
- ⚠️ More complex maintenance
- ⚠️ Requires careful conflict resolution

### Option C: Contribute Fork Enhancements to Upstream

**Best for**: Long-term collaboration, making your improvements official

1. **Identify contributions**:
   - FreeSASA/RSA implementation
   - HPC awareness and caching
   - Vectorized edge construction

2. **Prepare pull request**:
   - Document FreeSASA integration
   - Add HPC support as optional feature
   - Keep backward compatibility

3. **Maintain fork until upstream adopts**:
   - Continue using fork
   - Once upstream merges, switch to upstream

**Pros**:
- ✅ Benefits entire community
- ✅ Official support for your enhancements
- ✅ Easier long-term maintenance

**Cons**:
- ⚠️ Requires PR preparation and review
- ⚠️ May take time for upstream to merge

## Decision Matrix

| Scenario | Recommended Approach |
|----------|---------------------|
| **Active HER2 research, need results now** | Option A: Continue with fork |
| **Want to stay updated with upstream** | Option B: Hybrid approach |
| **Planning to publish/share pipeline** | Option C: Contribute to upstream |
| **Just starting new project** | Option A: Fork (has all features) |
| **Standard environment (not HPC)** | Could use upstream, but fork still better |

## Immediate Next Steps

### For Current Work (Recommended):

1. **Continue using fork branch**:
   ```bash
   git checkout fork/her2-selection-and-visualization
   ```

2. **Verify your pipeline works**:
   ```bash
   # Test on one PDB
   python data_processing/epi4ab_pipeline.py \
       --pdb_file /path/to/test.pdb \
       --output_dir /path/to/output \
       --sequence_source auto
   ```

3. **Monitor upstream for critical updates**:
   ```bash
   git fetch upstream
   git log upstream/master --oneline --since="2 weeks ago"
   ```

### If You Want to Try Upstream (Not Recommended for HPC):

1. **Create test branch**:
   ```bash
   git checkout -b test/upstream-pipeline
   git merge upstream/master
   ```

2. **Test on login node** (upstream needs internet):
   ```bash
   # On login node (not compute node)
   cd preprocess/
   python main.py --help
   ```

3. **Compare outputs** with fork version

4. **Revert if issues**:
   ```bash
   git checkout fork/her2-selection-and-visualization
   ```

## Summary

**For your current situation (HER2 research on Leonardo HPC):**

✅ **Use the fork** (`fork/her2-selection-and-visualization`)  
✅ **Keep your enhancements** (FreeSASA, caching, HPC support)  
✅ **Continue with existing SLURM workflows**  
⚠️ **Monitor upstream** for documentation/bug fixes, but don't switch  
💡 **Consider contributing** FreeSASA implementation back to upstream later

The fork is **more feature-complete** and **better suited for your HPC environment** than upstream. The upstream pipeline is useful as a reference, but switching would lose critical functionality you need.


