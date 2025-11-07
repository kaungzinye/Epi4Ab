# Pipeline Comparison: Our Implementation vs Official Dataset

## Executive Summary

This document compares our Epi4Ab pipeline implementation with the official dataset structure provided in `final_trained_Epi4Ab/examples/1ahw_BAC/`.

## 1. PDB List Alignment

### Official Dataset
- **Training Set**: 304 PDBs (`final_trained_Epi4Ab/pdb_list/train3A.txt`)
- **Test Set**: 21 PDBs (`final_trained_Epi4Ab/pdb_list/unseen3A.txt`)

### Our HER2 Test Set
- **Total**: 12 HER2 PDBs (from SAbDab filtering)
- **Overlap with Official Test Set**: 8 PDBs
  - 1N8Z, 1S78, 3BE1, 3N85, 3WLW, 3WSQ, 5O4G, 7MN8
- **⚠️ Issue**: 6BGT is in official **training set** (should be removed from our test set)
- **Not in Official Lists**: 3H3B, 6J71, 8Q6J

### Recommendation
- Use corrected HER2 list: `her2_test_set_filtered_corrected.csv` (11 PDBs, excludes 6BGT)
- Consider aligning with official test set PDBs for direct comparison

## 2. Node Feature Comparison

### Column Count
- **Official**: 68 columns
- **Ours**: 56 columns
- **Missing**: 19 columns
- **Extra**: 7 columns

### Missing Columns (19)

#### Structural Properties
- `sasa` - Solvent Accessible Surface Area
- `charge` - Residue charge

#### Physical Properties
- `weight` - Molecular weight (Da)
- `volume` - Residue volume (Å³)
- `hydrophobicity` - Kyte-Doolittle hydrophobicity scale
- `atomNumber` - Number of atoms per residue
- `pI` - Isoelectric point

#### Chemical Property Flags (one-hot encoded)
- `negative` - Negatively charged residues (ASP, GLU)
- `positive` - Positively charged residues (ARG, LYS, HIS)
- `polar` - Polar residues
- `hydrophobic` - Hydrophobic residues
- `neutral` - Neutral residues
- `hydroxyl` - Residues with hydroxyl group (SER, THR, TYR)
- `sulfur` - Sulfur-containing residues (CYS, MET)
- `carboxyl` - Carboxyl group (ASP, GLU)
- `amino` - Amino group (LYS, ARG)
- `heterocyclic` - Heterocyclic residues (HIS, TRP, PRO)
- `benzene` - Aromatic residues (PHE, TYR, TRP)
- `imino` - Imino group (PRO)

### Extra Columns (7)
- `VH10`, `VH11`, `VH12`, `VH13` - Additional VH gene families
- `VK7`, `VK9`, `VK11` - Additional VK gene families

**Note**: These extra columns suggest our IMGT gene family detection may be more comprehensive, but we should align with official 68-column format.

### Common Columns (49)
All structural, dihedral, and IMGT gene family columns match (except for the extra VH/VK families).

## 3. Edge Attributes Comparison

### Structure Differences

**Official Format:**
- `edge_attribute_dist.parquet`: Shape `(N_edges, 1)` with column `['dist']`
- `edge_attribute_charge.parquet`: Shape `(N_edges, 1)` with column `['qi*qj']`
- Only contains the attribute values, no source/target columns

**Our Format:**
- `edge_attribute_dist.parquet`: Shape `(N_edges, 3)` with columns `['source', 'target', 'dist']`
- `edge_attribute_charge.parquet`: Shape `(N_edges, 3)` with columns `['source', 'target', 'charge']`
- Includes source/target columns (redundant since `edge_index.parquet` already has this)

### Recommendation
- **Fix**: Drop `source` and `target` columns from edge attribute files
- Keep only the attribute values: `['dist']` and `['qi*qj']` (or `['charge']`)

## 4. Output Files Comparison

### Official Structure
```
1ahw_BAC/
├── nodes_edges/
│   ├── node_feature.parquet (68 columns)
│   ├── node_label_pi.parquet (resId, isInterface)
│   ├── node_list.parquet (resId, chainId)  ⚠️ Missing in ours
│   ├── edge_index.parquet (source, target)
│   ├── edge_attribute_dist.parquet (dist)
│   └── edge_attribute_charge.parquet (qi*qj)
└── preprocess/
    ├── aac/ (aac_result.parquet)
    ├── angle/ (angle_result.parquet)
    ├── atc/ (cc_result.parquet)
    ├── cdrs/ (cdrs_info.json)
    ├── cips/ (CIPS interface files)
    ├── depth/ (depth_result.parquet)
    ├── ellipro_bepiPred/ (bepiPred_result.txt, overlap_label2.txt)
    └── sequence/ (antigen_sequence.json, cdr_sequence.json, etc.)
```

### Our Structure
```
{PDB_ID}/
├── node_feature.parquet (56 columns) ⚠️ Missing 19 columns
├── node_label_pi.parquet (resId, isInterface) ✓
├── edge_index.parquet (source, target) ✓
├── edge_attribute_dist.parquet (source, target, dist) ⚠️ Extra columns
├── edge_attribute_charge.parquet (source, target, charge) ⚠️ Extra columns
└── sequence/
    ├── antigen_sequence.json ✓
    └── cdr_sequence.json ✓
```

### Missing Files
- `node_list.parquet` - Maps residue IDs to chain IDs
- `preprocess/` subdirectories - Intermediate preprocessing results (optional, for debugging)

## 5. Required Fixes

### High Priority
1. **Add missing 19 columns to `node_feature.parquet`**
   - Implement SASA calculation (using Naccess or MDAnalysis)
   - Add charge calculation
   - Add physical properties (weight, volume, hydrophobicity, atomNumber, pI)
   - Add chemical property flags (one-hot encoding)

2. **Fix edge attribute files**
   - Remove `source` and `target` columns
   - Rename `charge` to `qi*qj` in charge file
   - Keep only attribute values

3. **Generate `node_list.parquet`**
   - Create file with `resId` and `chainId` columns
   - Maps each residue to its chain

4. **Update HER2 test set**
   - Remove 6BGT (it's in official training set)
   - Use `her2_test_set_filtered_corrected.csv` (11 PDBs)

### Medium Priority
5. **Align IMGT gene families**
   - Remove extra VH/VK families (VH10-13, VK7, VK9, VK11)
   - Or verify if these are valid additions

6. **Add preprocess subdirectories** (optional)
   - For debugging and intermediate result storage
   - Not strictly necessary for training/inference

## 6. Column Implementation Guide

### SASA (Solvent Accessible Surface Area)
```python
# Using MDAnalysis or Naccess
from MDAnalysis.analysis import distances
# Or use Naccess tool if available
```

### Charge
```python
# Calculate from residue type
charge_map = {
    'ASP': -1, 'GLU': -1,  # Negative
    'ARG': +1, 'LYS': +1, 'HIS': +0.5  # Positive/partial
}
```

### Physical Properties
- Already defined in pipeline: `residue_weights`, `residue_volumes`, `hydrophobicity`, `atom_counts`, `isoelectric_points`
- Need to add to node features DataFrame

### Chemical Property Flags
```python
# One-hot encoding based on residue type
chemical_properties = {
    'negative': ['ASP', 'GLU'],
    'positive': ['ARG', 'LYS', 'HIS'],
    'polar': ['ASN', 'GLN', 'SER', 'THR', 'TYR', 'CYS'],
    'hydrophobic': ['ALA', 'VAL', 'ILE', 'LEU', 'MET', 'PHE', 'TRP'],
    # ... etc
}
```

## 7. Testing Alignment

After implementing fixes, verify:
1. `node_feature.parquet` has exactly 68 columns matching official order
2. Edge attribute files have correct structure (no source/target)
3. `node_list.parquet` is generated correctly
4. All HER2 test PDBs are processed successfully
5. Output matches official format for at least one test PDB (e.g., 1N8Z)

## 8. Next Steps

1. ✅ Update HER2 list (remove 6BGT)
2. ⏳ Implement missing 19 columns in `extract_node_features()`
3. ⏳ Fix edge attribute file structure
4. ⏳ Add `node_list.parquet` generation
5. ⏳ Test with official test set PDBs
6. ⏳ Verify column order matches official format

