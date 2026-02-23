# Residue Numbering: Sequential vs PDB Residue Numbers

## Overview

When working with protein structures, there are two common ways to number residues:

1. **Sequential Numbering**: Residues numbered 1, 2, 3, 4... starting from the first residue in the structure
2. **PDB Residue Numbers**: Residues numbered according to their original position in the full protein sequence (as stored in the PDB file)

## Sequential Numbering

### What it is:
- Residues are numbered starting from 1, incrementing by 1 for each residue
- The first residue in your structure = 1
- The second residue = 2
- And so on...

### Example:
If you extract residues 380-387 from a protein:
```
Sequential:  1,  2,  3,  4,  5,  6,  7,  8
PDB resIds: 380, 381, 382, 383, 384, 385, 386, 387
```

### When it's used:
- When extracting a fragment/subdomain from a larger structure
- When the original PDB numbering is not preserved
- In some preprocessing pipelines that renumber residues

### Advantages:
- Simple and consistent
- Always starts at 1
- Easy to work with programmatically

### Disadvantages:
- Loses connection to original PDB structure
- Cannot directly reference literature/experimental data that uses PDB numbers
- Makes it hard to map back to full protein sequence

## PDB Residue Numbers

### What it is:
- Residues keep their original numbering from the PDB file
- Reflects the position in the full protein sequence
- May have gaps (e.g., 380, 381, 383, 384 if residue 382 is missing)
- May start at any number (not necessarily 1)

### Example:
From SARS-CoV-2 Spike protein RBD:
```
PDB resIds: 331, 332, 333, ..., 380, 381, 382, 383, 384, ..., 527
           (RBD domain starts around 331, ends around 527)
```

### When it's used:
- When preserving original structure information
- When referencing experimental data (e.g., "residue 384 is an epitope")
- When mapping to full protein sequences
- In most structural biology workflows

### Advantages:
- Preserves original structure information
- Can directly reference literature and experimental data
- Maps correctly to full protein sequences
- Maintains biological context

### Disadvantages:
- May have gaps or non-sequential numbering
- More complex to work with programmatically

## The Problem in Our Seqitope Data

### What Happened:

**Ty1_RBD** (Working correctly):
- Node features use PDB residue numbers: 14-730
- Raw ground truth specifies: 351, 449, 472, 490
- ✅ These resIds exist in node features
- ✅ Labels can be generated correctly

**mAb159_RBD** (Problem):
- Node features use sequential numbering: 1-199
- Raw ground truth specifies: 380, 381, 383, 384, 386, 387
- ❌ These resIds don't exist in node features (only 1-199 exist)
- ❌ Labels cannot match raw_ground_truth.csv

**mAb311_RBD** (Problem):
- Node features use sequential numbering: 1-199
- Raw ground truth specifies: 456, 475, 487, 493, 494
- ❌ These resIds don't exist in node features (only 1-199 exist)
- ❌ Labels cannot match raw_ground_truth.csv

### Why This Matters:

The `raw_ground_truth.csv` file contains experimental data that references **PDB residue numbers**:
- mAb159_RBD epitopes at residues 380, 381, 383, 384, 386, 387
- mAb311_RBD epitopes at residues 456, 475, 487, 493, 494

But the node features were generated with **sequential numbering** (1-199), so:
- There's no residue numbered 380, 384, etc. in the node features
- The label generation script looks for these resIds and can't find them
- Result: All labels become 0 (non-epitope) instead of the correct epitope labels

## Visual Example

### Sequential Numbering (Current mAb159_RBD):
```
Node Feature resIds:  1,  2,  3, ..., 180, 181, 182, ..., 199
Raw Ground Truth:    380, 381, 383, 384, 386, 387  ← NOT FOUND!
Labels Generated:    0,  0,  0, ...,  0,   0,   0, ...,  0  (all wrong!)
```

### PDB Residue Numbers (What we need):
```
Node Feature resIds:  331, 332, ..., 380, 381, 382, 383, 384, ..., 527
Raw Ground Truth:    380, 381, 383, 384, 386, 387  ← FOUND!
Labels Generated:    0,  0, ...,  1,  1,  0,  1,  1, ...,  0  (correct!)
```

## Solution

To fix mAb159_RBD and mAb311_RBD:

1. **Regenerate node features** with PDB residue numbers preserved
   - The preprocessing pipeline needs to extract the RBD domain while keeping original PDB numbering
   - This requires using the correct PDB structure and chain

2. **Then regenerate labels** from raw_ground_truth.csv
   - Once node features have correct PDB resIds, labels will match

## How to Check Residue Numbering

You can check which numbering system is used:

```python
import pandas as pd

# Load node features
df = pd.read_parquet('node_feature.parquet')

# Check resId range
print(f"resId range: {df['resId'].min()} - {df['resId'].max()}")
print(f"First 10 resIds: {df['resId'].head(10).tolist()}")

# Sequential numbering: Usually starts at 1, no gaps
# PDB numbering: May start at any number, may have gaps
```

## Key Takeaway

**Sequential numbering** is convenient for computation but loses biological context.
**PDB residue numbers** preserve biological context and allow mapping to experimental data.

For our use case, we need **PDB residue numbers** because:
- Raw ground truth data uses PDB numbers
- We need to match experimental epitope data
- We want to map results back to full protein structures
