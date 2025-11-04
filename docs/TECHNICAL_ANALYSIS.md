# Technical Deep-Dive: Epi4Ab Pipeline Debugging

**Date:** October 23, 2025  
**Focus:** Critical Edge Attribute Bug Analysis

---

## Table of Contents
1. [The Critical Edge Attribute Bug](#the-critical-edge-attribute-bug)
2. [DataFrame Operations and `.reshape(-1)`](#dataframe-operations-and-reshape-1)
3. [Data Flow Through Pipeline](#data-flow-through-pipeline)
4. [Why the Original Code Was Broken](#why-the-original-code-was-broken)
5. [Impact on Graph Neural Network](#impact-on-graph-neural-network)

---

## The Critical Edge Attribute Bug

### Problem Statement

The most critical bug in the Epi4Ab pipeline was a **silent data corruption** issue caused by improper DataFrame handling. This bug caused:
- Assertion failures during inference
- Incorrect edge potential calculations
- Complete pipeline failure

### The Buggy Code

**Location:** `interface_prediction/data_function/data_function.py:81-82`

```python
# BEFORE (BROKEN):
edgeAttribute = pd.read_parquet(os.path.join(pdb_folder, logging.edge_attribute_file))
edgeCharge = pd.read_parquet(os.path.join(pdb_folder, logging.edge_attribute_charge_file))

attribute = torch.tensor(edgeAttribute.to_numpy().reshape(-1), dtype=torch.float)
attributeCharge = torch.tensor(edgeCharge.to_numpy().reshape(-1), dtype=torch.int)
```

### DataFrame Structure

The parquet files contain edge lists with THREE columns:

```python
# edgeAttribute DataFrame (edge_attribute_dist.parquet):
   source  target   dist
0      0      15   8.52
1      0      23   5.24
2      1      12   7.89
3      1      45   9.13
...

# edgeCharge DataFrame (edge_attribute_charge.parquet):
   source  target  charge
0      0      15      -1
1      0      23       1
2      1      12       0
3      1      45      -1
...
```

**Why three columns?**
- `source`: Index of first node in edge
- `target`: Index of second node in edge
- `dist`/`charge`: The actual attribute value we need

### What `.to_numpy().reshape(-1)` Does

Let's trace through the execution step by step:

#### Step 1: Read DataFrame
```python
edgeAttribute = pd.read_parquet('edge_attribute_dist.parquet')
# Result: DataFrame with shape (3692, 3)
```

#### Step 2: Convert to NumPy
```python
arr = edgeAttribute.to_numpy()
# Result: 2D array with shape (3692, 3)
# Array contents:
# [[  0.  15.   8.52]
#  [  0.  23.   5.24]
#  [  1.  12.   7.89]
#  [  1.  45.   9.13]
#  ...]
```

#### Step 3: Flatten with `.reshape(-1)`
```python
flattened = arr.reshape(-1)
# Result: 1D array with shape (11076,)  # 3692 * 3 = 11076
# Array contents (ROW-MAJOR ORDER):
# [0., 15., 8.52, 0., 23., 5.24, 1., 12., 7.89, 1., 45., 9.13, ...]
#  ↑   ↑    ↑    ↑   ↑    ↑     ↑   ↑    ↑     ↑   ↑    ↑
#  source  dist  source  dist   source  dist   source  dist
#  INDEX!  MIX!  INDEX!  MIX!   INDEX!  MIX!   INDEX!  MIX!
```

### The Data Corruption

The flattened array now contains a **mixture** of:
- Node indices (0, 1, 2, ... 213)
- Distance values (2.84 - 10.00 Å)

**Problem:** Many node indices are **0** (first node in the structure), which gets mixed into what should be a pure distance array!

```
Expected distance array:
[8.52, 5.24, 7.89, 9.13, ...]  # All values 2.84-10.00

Actual (buggy) array:
[0., 15., 8.52, 0., 23., 5.24, 1., 12., 7.89, ...]  # Contains zeros!
      ↑              ↑                        ↑
   ZEROS FROM NODE INDICES!
```

### Impact on Edge Potentials

The corrupted distance array is used to calculate edge potentials:

```python
# In attribute.py:
class CalculateAttribute:
    def forward(self, distance, charge):
        denominator = distance + self.bias_distance  # bias_distance = 0
        
        # Assertion fails here when distance contains zeros!
        assert torch.gt(denominator, 0).all(), \
            "Edge Attribute - Bias_distance should be greater than 0."
        
        # Bond potential: 1/d
        bond_potential = 1.0 / denominator  # Division by zero → Inf!
        
        # Lennard-Jones: (r₀/d)¹² - (r₀/d)⁶
        lj_potential = torch.pow(r0/denominator, 12) - torch.pow(r0/denominator, 6)  # Inf!
        
        # Charge: q₁*q₂/d
        charge_potential = charge / denominator  # Inf or NaN!
```

**Result:** Pipeline crashes with assertion error before even getting to the Inf/NaN values.

---

## DataFrame Operations and `.reshape(-1)`

### Understanding NumPy `.reshape(-1)`

`.reshape(-1)` means "flatten to 1D with automatic size calculation."

**How it flattens:**
- **Row-major order** (C-style): left-to-right, then down
- Takes each row, appends it to result
- Continues to next row

**Visual Example:**
```python
arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])
                
arr.reshape(-1)  # [1, 2, 3, 4, 5, 6, 7, 8, 9]
                 #  --------  --------  --------
                 #   row 0     row 1     row 2
```

### The Correct Approach

**Select the column FIRST, then flatten:**

```python
# AFTER (FIXED):
attribute = torch.tensor(edgeAttribute['dist'].to_numpy().reshape(-1), dtype=torch.float)
attributeCharge = torch.tensor(edgeCharge['charge'].to_numpy().reshape(-1), dtype=torch.int)
```

**What this does:**

#### Step 1: Select Column
```python
dist_series = edgeAttribute['dist']
# Result: Series with shape (3692,) containing ONLY distances
# [8.52, 5.24, 7.89, 9.13, ...]
```

#### Step 2: Convert to NumPy
```python
dist_array = dist_series.to_numpy()
# Result: 1D array with shape (3692,)
# [8.52, 5.24, 7.89, 9.13, ...]
```

#### Step 3: Flatten (redundant but harmless)
```python
dist_flat = dist_array.reshape(-1)
# Result: 1D array with shape (3692,) - unchanged because already 1D
# [8.52, 5.24, 7.89, 9.13, ...]
```

**Key Difference:** Only distance values, no node indices mixed in!

### Alternative (Also Correct) Approaches

```python
# Option 1: Select column by position
attribute = torch.tensor(edgeAttribute.iloc[:, 2].to_numpy(), dtype=torch.float)

# Option 2: Filter columns first
dist_df = edgeAttribute[['dist']]
attribute = torch.tensor(dist_df.to_numpy().reshape(-1), dtype=torch.float)

# Option 3: Direct indexing (most explicit)
attribute = torch.tensor(edgeAttribute['dist'].values, dtype=torch.float)
```

---

## Data Flow Through Pipeline

### Pipeline Architecture

```
┌──────────────┐
│   PDB File   │
│   (1N8Z.pdb) │
└──────┬───────┘
       │
       ▼
┌─────────────────────────────────────┐
│  epi4ab_pipeline.py                 │
│  ─────────────────────────────────  │
│  1. Load PDB with MDAnalysis        │
│  2. Extract node features           │
│     - resId, resShort, depths       │
│     - dihedral angles (φ,ψ,ω,χ)    │
│     - RSA, partial charges          │
│     - VH/VL family features         │
│  3. Extract epitope labels          │
│  4. Build graph connectivity        │
│     - K-d tree for spatial search   │
│     - Find neighbors within 10Å     │
│  5. Calculate edge attributes       │
│     - Distances between CAs         │
│     - Charge interactions           │
│  6. Extract & save sequence         │
└─────────────┬───────────────────────┘
              │
              ▼
┌────────────────────────────────────┐
│  Parquet Files (per PDB)           │
│  ────────────────────────────────  │
│  • node_feature.parquet            │
│    - 50 columns × N residues       │
│    - Includes resShort!            │
│  • node_label_pi.parquet           │
│  • edge_index.parquet              │
│    - [source, target] pairs        │
│  • edge_attribute_dist.parquet     │
│    - [source, target, dist] ← BUG! │
│  • edge_attribute_charge.parquet   │
│    - [source, target, charge]      │
│  • sequence/antigen_sequence.json  │
│    - {"pdb_sequence": "DIQMTQ..."} │
└─────────────┬──────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  data_function.py (batch_list)      │
│  ───────────────────────────────────│
│  1. Load parquet files              │
│  2. Convert to PyTorch tensors      │
│     ❌ BUG WAS HERE (line 81-82)    │
│     ✅ NOW FIXED: Select column     │
│  3. Calculate edge potentials       │
│  4. Load ESM2 sequence embeddings   │
│  5. Build PyTorch Geometric Data    │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  Graph Neural Network               │
│  ───────────────────────────────────│
│  • Node features → Initial process  │
│  • ESM2 embeddings → MHA layers     │
│  • Edge attributes → Message passing│
│  • GAT layers → Epitope prediction  │
│  • Output: [background, epitope,    │
│            potential epitope]        │
└─────────────────────────────────────┘
```

### Critical Data Transformations

#### 1. PDB → Node Features
```
PDB Residues (3D coordinates)
  ↓ MDAnalysis
ResID, ResName, CA positions
  ↓ Dihedral calculation
Phi, Psi, Omega, Chi angles
  ↓ AA mapping
resShort: Three-letter → One-letter codes
  ↓ Combine all
DataFrame (N × 50) → Parquet
```

#### 2. Spatial Graph → Edge List
```
CA atom positions (Nx3)
  ↓ K-d tree
Find neighbors within 10Å
  ↓ Distance calculation
source, target, distance
  ↓ Charge lookup
source, target, charge
  ↓ Save
edge_attribute_dist.parquet (M × 3)
edge_attribute_charge.parquet (M × 3)
```

#### 3. Parquet → PyTorch Tensors
```
DataFrame (M × 3) with [source, target, value]
  ↓ ❌ OLD: .to_numpy().reshape(-1)
  ↓    Corrupted array (M*3 × 1) with mixed data
  ↓
  ↓ ✅ NEW: ['value'].to_numpy().reshape(-1)
  ↓    Clean array (M × 1) with only values
  ↓
torch.Tensor for GNN
```

### Required Columns at Each Stage

| Stage | File | Required Columns |
|-------|------|------------------|
| **Processing Output** | node_feature.parquet | resId, **resShort**, resDepth, caDepth, psi, phi, omega, chi, aac, cc, H1_len, ..., VH1-14, VK1-14, ... (50 total) |
| | node_label_pi.parquet | resId, isInterface, interfaceType |
| | edge_index.parquet | source, target |
| | edge_attribute_dist.parquet | source, target, **dist** |
| | edge_attribute_charge.parquet | source, target, **charge** |
| | sequence/antigen_sequence.json | {"pdb_sequence": "DIQMTQ..."} |
| **Inference Input** | PyTorch Data object | x (node features), x_seq (ESM2), edge_index, edge_attr (dist only!), y (labels), res_short |

**Key Point:** The parquet files store `[source, target, value]` for traceability, but inference only needs the `value` column!

---

## Why the Original Code Was Broken

### Hypothesis 1: Incomplete Testing

**Evidence:**
- Multiple critical bugs (4 found)
- Basic functionality was broken
- No validation checks in place

**Likely Scenario:** Code was developed iteratively without full end-to-end testing. Each component may have been tested in isolation but not as a complete pipeline.

### Hypothesis 2: Data Format Change

**Evidence:**
- The `.reshape(-1)` suggests code was written for a 1D array or single-column DataFrame
- Current data has 3 columns (`[source, target, value]`)

**Likely Scenario:**
```python
# Original data format (single column):
dist.parquet:
   dist
0  8.52
1  5.24
2  7.89

# Works fine:
torch.tensor(dist.to_numpy().reshape(-1))  # [8.52, 5.24, 7.89]

# Later changed to 3 columns (for debugging/traceability):
edge_attribute_dist.parquet:
   source  target  dist
0      0      15  8.52
1      0      23  5.24
2      1      12  7.89

# Code not updated → BUG:
torch.tensor(edgeAttr.to_numpy().reshape(-1))  # [0, 15, 8.52, 0, 23, 5.24, ...]
```

### Hypothesis 3: Development/Research Code Shared

**Evidence:**
- Missing initialization (`residue_weights` outside `__init__`)
- Missing features (`resShort` column)
- Incomplete implementation (sequence files not saved)

**Likely Scenario:** This was research/development code that:
1. Had hardcoded paths or manual data preparation
2. Relied on external scripts to prepare data
3. Was never fully integrated into a production pipeline
4. Was shared before being production-ready

### Hypothesis 4: Version Mismatch

**Evidence:**
- Code structure suggests pandas < 1.0 behavior
- MDAnalysis usage patterns vary by version

**Possible Issue:** Code written for older library versions where:
- DataFrame `.values` behaved differently
- Default array shapes were different
- Type handling was less strict

### Most Likely Explanation

**Combination of factors:**
1. Research code not production-ready
2. Data format evolved without code updates
3. Missing integration testing
4. Shared during development phase

**Key Takeaway:** Always validate DataFrame shapes and column selections explicitly. Don't assume `.reshape(-1)` on a DataFrame will do what you expect!

---

## Impact on Graph Neural Network

### How Edge Attributes Are Used

In Graph Attention Networks (GAT), edge attributes modulate message passing:

```python
# Simplified GNN message passing:
for each node i:
    message = 0
    for each neighbor j:
        # Edge attributes weight the messages
        edge_weight = calculate_potential(distance[i,j], charge[i,j])
        message += edge_weight * feature[j]
    
    updated_feature[i] = aggregate(message)
```

### Impact of Corrupted Data

With zeros in the distance array:

```python
# Bond potential: 1/d
bond = 1.0 / distance
# If distance = 0 → bond = Inf → message = Inf → CRASH

# Lennard-Jones: (r₀/d)¹² - (r₀/d)⁶
lj = (3.5/distance)**12 - (3.5/distance)**6
# If distance = 0 → lj = Inf → CRASH

# Charge: q₁*q₂/d
charge_pot = charge / distance
# If distance = 0 → charge_pot = Inf or NaN → CRASH
```

**Even before the crash:** Wrong distances mean:
- Incorrect spatial relationships
- Wrong message passing weights
- Completely wrong predictions

### Why Assertion Was There

```python
assert torch.gt(denominator, 0).all(), \
    "Edge Attribute - Bias_distance should be greater than 0."
```

**Purpose:** Catch exactly this type of bug!
- Distances must be positive
- Prevents division by zero
- Validates data integrity

**The assertion SAVED US** from silent corruption propagating through the network!

---

## Lessons Learned

### For DataFrame Operations

1. **Always select columns explicitly**
   ```python
   # ✅ GOOD: Clear intent
   distances = df['dist'].to_numpy()
   
   # ❌ BAD: Unclear what you're getting
   data = df.to_numpy().reshape(-1)
   ```

2. **Validate shapes**
   ```python
   arr = df['dist'].to_numpy()
   assert arr.ndim == 1, f"Expected 1D, got {arr.ndim}D"
   assert arr.min() > 0, f"Distances must be positive, got min={arr.min()}"
   ```

3. **Use assertions liberally**
   ```python
   # Check data integrity
   assert not np.any(np.isnan(arr)), "Array contains NaN"
   assert not np.any(np.isinf(arr)), "Array contains Inf"
   assert arr.dtype == expected_dtype, f"Wrong dtype: {arr.dtype}"
   ```

### For Pipeline Development

1. **End-to-end testing is critical**
   - Test the FULL pipeline, not just components
   - Use real data, not synthetic/mocked data
   - Validate outputs at each stage

2. **Add validation layers**
   - Check required columns exist
   - Validate value ranges
   - Verify data types

3. **Document data formats**
   - What columns are required?
   - What are valid ranges?
   - What are the units?

4. **Version control data formats**
   - Track schema changes
   - Add compatibility checks
   - Document breaking changes

---

## Conclusion

The edge attribute bug was a perfect example of **silent data corruption** - the code ran without obvious errors until it hit an assertion. This highlights the importance of:

1. **Defensive programming** - Validate inputs, check outputs
2. **Explicit operations** - Select columns by name, not assumption
3. **Comprehensive testing** - End-to-end with real data
4. **Good assertions** - The bias_distance check caught the bug!

The fix was simple (2 characters: `['dist']`), but finding it required understanding:
- DataFrame internal structure
- NumPy reshape behavior
- GNN edge attribute usage
- Assertion failure root cause

**Always remember:** When working with DataFrames, be explicit about what columns you want!

---

**Document Version:** 1.0  
**Last Updated:** October 23, 2025  
**Author:** Debugging Session Analysis

