# Epi4Ab Pipeline Flow Diagram

**Last Updated:** October 29, 2025

## Complete Data Processing & Inference Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                    RAW INPUT                                        │
│  PDB/CIF Files (e.g., /epi4ab/pdb_files/1N8Z.pdb)                  │
└───────────────────────────┬───────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                STEP 1: DATA PROCESSING                              │
│              (epi4ab_pipeline.py)                                    │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │ 1.1 STRUCTURE LOADING & INITIALIZATION                       │  │
│  │                                                              │  │
│  │  1.1.1 Load PDB/CIF file using MDAnalysis Universe          │  │
│  │  1.1.2 Initialize ESM2 model (esm2_t30_150M_UR50D)         │  │
│  │  1.1.3 Load residue property dictionaries:                  │  │
│  │        ├─ Hydrophobicity (Kyte-Doolittle scale)             │  │
│  │        ├─ Residue weights (molecular weight in Da)          │  │
│  │        ├─ Residue volumes                                  │  │
│  │        ├─ Isoelectric points                                │  │
│  │        └─ Atom counts                                       │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │ 1.2 ANTIGEN/ANTIBODY CHAIN SELECTION                        │  │
│  │                                                              │  │
│  │  1.2.1 Attempt to select antigen chain A by default          │  │
│  │  1.2.2 If chain A is empty, auto-detect:                    │  │
│  │        ├─ Scan all chains (excluding B, C for antibody)    │  │
│  │        ├─ Select largest chain by CA atom count             │  │
│  │        └─ Update self.antigen_chain to detected chain      │  │
│  │  1.2.3 Select antibody chains (B and/or C)                 │  │
│  │  1.2.4 Store antigen_ca and antibody_ca AtomGroups           │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │ 1.3 RESIDUE DEDUPLICATION                                    │  │
│  │                                                              │  │
│  │  1.3.1 Identify duplicate residues (same resnum/insertion)    │  │
│  │  1.3.2 Remove duplicates from antigen_ca residues            │  │
│  │        └─ Example: 6B0N had duplicate residue 321            │  │
│  │  1.3.3 Update residue list to ensure uniqueness              │  │
│  │  1.3.4 Note: Ensures sequence-node count consistency         │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │ 1.4 NODE FEATURE EXTRACTION                                  │  │
│  │     (extract_node_features)                                  │  │
│  │                                                              │  │
│  │  1.4.1 PRE-COMPUTE ANTIGEN CENTER-OF-MASS                    │  │
│  │        ├─ Calculate CoM from antigen CA atoms only           │  │
│  │        ├─ Calculate raw depths for all antigen residues       │  │
│  │        └─ Compute max_raw_depth for normalization            │  │
│  │                                                              │  │
│  │  1.4.2 FOR EACH ANTIGEN RESIDUE:                             │  │
│  │                                                              │  │
│  │      A. BASIC RESIDUE PROPERTIES                             │  │
│  │         ├─ resShort: Convert 3-letter → 1-letter code        │  │
│  │         │  (e.g., ALA → A, ARG → R)                          │  │
│  │         ├─ resWeight: Molecular weight from IMGT            │  │
│  │         ├─ resVolume: Van der Waals volume                   │  │
│  │         ├─ Hydrophobicity: Kyte-Doolittle value              │  │
│  │         ├─ Isoelectric Point: pI value                       │  │
│  │         └─ Atom Count: Heavy atoms in residue                 │  │
│  │                                                              │  │
│  │      B. STRUCTURAL FEATURES                                  │  │
│  │         ├─ resDepth/caDepth:                                 │  │
│  │         │  ├─ Calculate CA distance to antigen CoM           │  │
│  │         │  ├─ Normalize: depth = 30 × (raw/max_raw)          │  │
│  │         │  └─ Clip to [0, 30] range                          │  │
│  │         ├─ RSA (Relative Solvent Accessibility):            │  │
│  │         │  └─ Simplified: distance_to_com / 20.0            │  │
│  │         └─ Partial Charge:                                   │  │
│  │            ├─ ARG, LYS → +1.0                                │  │
│  │            ├─ ASP, GLU → -1.0                                │  │
│  │            ├─ HIS → +0.5                                     │  │
│  │            └─ Others → 0.0                                   │  │
│  │                                                              │  │
│  │      C. DIHEDRAL ANGLES                                      │  │
│  │         ├─ Phi (C_{i-1}-N_{i}-CA_{i}-C_{i}):                │  │
│  │         │  └─ Backbone rotation angle                        │  │
│  │         ├─ Psi (N_{i}-CA_{i}-C_{i}-N_{i+1}):                │  │
│  │         │  └─ Backbone rotation angle                        │  │
│  │         ├─ Omega (CA_{i}-C_{i}-N_{i+1}-CA_{i+1}):           │  │
│  │         │  └─ Peptide bond angle                              │  │
│  │         └─ Chi (side chain dihedral):                        │  │
│  │            └─ Simplified (0.0 if not calculable)              │  │
│  │                                                              │  │
│  │      D. AMINO ACID COMPOSITION                               │  │
│  │         ├─ AAC (Aliphatic): ALA, VAL, LEU, ILE → 1          │  │
│  │         └─ CC (Charged): ARG, LYS, HIS → 1                   │  │
│  │                                                              │  │
│  │      E. ANTIBODY FAMILY ONE-HOT ENCODING                     │  │
│  │         ├─ VH families: VH1-VH7, VH_others, VH_unk           │  │
│  │         ├─ VK (kappa): VK1-VK6, VK_others, VK_unk            │  │
│  │         ├─ VLa (lambda): VLa1, VLa2, VLa3, VLa6,            │  │
│  │         │                VLa_others, VL_unk                 │  │
│  │         └─ Total: 50 columns (all one-hot encoded)           │  │
│  │                                                              │  │
│  │      F. ESM2 SEQUENCE EMBEDDINGS                             │  │
│  │         ├─ Extract full antigen sequence                      │  │
│  │         ├─ Tokenize with ESM2 tokenizer                       │  │
│  │         ├─ Forward pass through ESM2 model                   │  │
│  │         └─ Mean pool last hidden states → 1280-dim vector     │  │
│  │                                                              │  │
│  │  1.4.3 AGGREGATE ALL FEATURES                                │  │
│  │        └─ Create DataFrame with all 50 feature columns      │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │ 1.5 GRAPH CONNECTIVITY (Cα-Cα DISTANCE)                       │  │
│  │     (extract_graph_connectivity)                              │  │
│  │                                                              │  │
│  │  1.5.1 Calculate pairwise Cα-Cα distances                    │  │
│  │        └─ Using MDAnalysis distance_array()                   │  │
│  │  1.5.2 Apply 10Å cutoff (as per Epi4Ab paper)                │  │
│  │        ├─ Keep edges where distance <= 10.0 Å                 │  │
│  │        └─ Create edge_index: [source, target] pairs          │  │
│  │  1.5.3 Store as edge_index.parquet                            │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │ 1.6 EDGE ATTRIBUTE CALCULATION                                │  │
│  │     (extract_edge_attributes)                                 │  │
│  │                                                              │  │
│  │  1.6.1 FOR EACH EDGE (from edge_index):                      │  │
│  │                                                              │  │
│  │      A. DISTANCE ATTRIBUTES                                  │  │
│  │         ├─ dist: Raw Cα-Cα distance in Å                    │  │
│  │         ├─ Bond potential: 1/d_{i,j}                         │  │
│  │         └─ Lennard-Jones: 4ε[(σ/d)^12 - (σ/d)^6]            │  │
│  │            └─ Parameters: σ=3.4Å, ε=0.1                      │  │
│  │                                                              │  │
│  │      B. CHARGE ATTRIBUTES                                    │  │
│  │         ├─ Get partial charges for source & target residues  │  │
│  │         ├─ Calculate: q_i × q_j (charge product)             │  │
│  │         └─ Charge potential: (q_i × q_j) / d_{i,j}            │  │
│  │                                                              │  │
│  │  1.6.2 CREATE EDGE DATAFRAMES                                │  │
│  │        ├─ edge_attribute_dist.parquet:                        │  │
│  │        │  └─ Columns: [source, target, dist]                 │  │
│  │        │  └─ CRITICAL: Only 'dist' column used in           │  │
│  │        │              inference (explicit selection)          │  │
│  │        └─ edge_attribute_charge.parquet:                      │  │
│  │           └─ Columns: [source, target, charge]               │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │ 1.7 ANTIGEN SEQUENCE EXTRACTION                              │  │
│  │     (extract_antigen_sequence)                                │  │
│  │                                                              │  │
│  │  1.7.1 SEQUENCE SOURCE PRIORITY:                            │  │
│  │        ├─ Priority 1: FASTA file (if provided)                │  │
│  │        ├─ Priority 2: RCSB PDB API                          │  │
│  │        │  └─ Fetch from https://data.rcsb.org/...           │  │
│  │        └─ Priority 3: Extract from PDB structure             │  │
│  │                                                              │  │
│  │  1.7.2 IF PDB EXTRACTION USED:                               │  │
│  │        ├─ Extract sequence from antigen chain                │  │
│  │        ├─ REBUILD from deduplicated antigen_ca residues     │  │
│  │        └─ Ensures sequence length = node count               │  │
│  │           └─ Critical for 6B0N (was 610 → 609 after          │  │
│  │             removing duplicate residue 321)                  │  │
│  │                                                              │  │
│  │  1.7.3 MAP 3-LETTER → 1-LETTER CODES                          │  │
│  │        └─ Convert residue names to amino acid sequence       │  │
│  │                                                              │  │
│  │  1.7.4 SAVE SEQUENCE FILE                                    │  │
│  │        └─ sequence/antigen_sequence.json                     │  │
│  │           └─ Format: {"sequence": "ACDEFG..."}              │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │ 1.8 CDR SEQUENCE EXTRACTION                                  │  │
│  │     (extract_cdr_sequences)                                 │  │
│  │                                                              │  │
│  │  1.8.1 IDENTIFY ANTIBODY CHAINS                              │  │
│  │        ├─ Heavy chain (chain B or largest antibody chain)    │  │
│  │        └─ Light chain (chain C or kappa/lambda chain)         │  │
│  │                                                              │  │
│  │  1.8.2 EXTRACT CDR REGIONS (Chothia numbering)              │  │
│  │        ├─ Heavy chain:                                       │  │
│  │        │  ├─ H1: Residues ~26-32                             │  │
│  │        │  ├─ H2: Residues ~52-56                             │  │
│  │        │  └─ H3: Residues ~95-102 (variable length)         │  │
│  │        └─ Light chain:                                      │  │
│  │           ├─ L1: Residues ~24-34                             │  │
│  │           ├─ L2: Residues ~50-56                             │  │
│  │           └─ L3: Residues ~89-97                             │  │
│  │                                                              │  │
│  │  1.8.3 CONVERT TO 1-LETTER SEQUENCES                         │  │
│  │        └─ Map 3-letter codes → amino acid strings            │  │
│  │                                                              │  │
│  │  1.8.4 SAVE CDR SEQUENCES                                    │  │
│  │        └─ sequence/cdr_sequence.json                        │  │
│  │           └─ Format: {"H1": "...", "H2": "...", ...}         │  │
│  │           └─ Required for AntiBERTy feature extraction       │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │ 1.9 EPITOPE LABEL EXTRACTION                                 │  │
│  │     (extract_epitope_labels)                                 │  │
│  │                                                              │  │
│  │  1.9.1 ANTIBODY CHAIN DETECTION                              │  │
│  │        ├─ Robust multi-strategy detection                   │  │
│  │        ├─ Strategy 1: Standard chain IDs (B, C, H, L)        │  │
│  │        ├─ Strategy 2: Size-based (90-150 residues)           │  │
│  │        ├─ Strategy 3: Non-antigen chains fallback           │  │
│  │        └─ Validation and logging of detection results       │  │
│  │                                                              │  │
│  │  1.9.2 CALCULATE LABEL 1 (CIPS)                              │  │
│  │        ├─ CIPS method: 5Å cut-off from antibody Cα atoms    │  │
│  │        ├─ Calculate distance: antigen CA → antibody CA       │  │
│  │        └─ Label 1: Direct antibody-interacting (≤5Å)        │  │
│  │                                                              │  │
│  │  1.9.3 CALCULATE LABEL 2 (BEPIPRED 3.0 OR ELLIPRO CONSENSUS) │  │
│  │                                                              │  │
│  │        PREPROCESSING (Login Node - Optional but Recommended):│  │
│  │        ├─ Run: preprocess_epitope_predictions.py            │  │
│  │        ├─ Fetches Ellipro via web API (login node only)    │  │
│  │        ├─ Optionally pre-generates BepiPred predictions    │  │
│  │        └─ Caches both to cache directory                    │  │
│  │                                                              │  │
│  │        BEPIPRED 3.0 (Priority Order):                      │  │
│  │        ├─ Priority 1: Run bp3 locally (works on compute    │  │
│  │        │              nodes, ESM models cached)             │  │
│  │        ├─ Priority 2: Use cache (if preprocessed)          │  │
│  │        └─ Priority 3: Web API (login node only)            │  │
│  │                                                              │  │
│  │        ELLIPRO (Priority Order):                            │  │
│  │        ├─ Priority 1: Use cache (from preprocessing on      │  │
│  │        │              login node)                           │  │
│  │        ├─ Priority 2: Try local tool (unlikely, no          │  │
│  │        │              standalone tool exists)               │  │
│  │        └─ Priority 3: Web API (login node only, skipped     │  │
│  │                      on compute nodes)                     │  │
│  │                                                              │  │
│  │        LABEL 2 CONSENSUS:                                   │  │
│  │        ├─ Label 2 if BepiPred score >= 0.3 OR               │  │
│  │        │         Ellipro score >= 0.5                       │  │
│  │        ├─ Either tool predicting epitope = Label 2          │  │
│  │        ├─ Note: BepiPred 3.0 threshold = 0.3 (scores       │  │
│  │        │        typically range 0.0-0.4)                    │  │
│  │        └─ No fallback: Label 2 = 0 if predictions          │  │
│  │          unavailable (strict requirement)                   │  │
│  │                                                              │  │
│  │  1.9.4 SAVE LABELS                                           │  │
│  │        └─ node_label_pi.parquet: Three-class labels         │  │
│  │           └─ 0 = Non-epitope                                 │  │
│  │           └─ 1 = CIPS (direct antibody contact)             │  │
│  │           └─ 2 = Ellipro+BepiPred (predicted epitope)       │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │ 1.10 FILE OUTPUT & SAVING                                    │  │
│  │                                                              │  │
│  │  1.10.1 SAVE NODE FEATURES                                   │  │
│  │          └─ node_feature.parquet (50 columns)               │  │
│  │                                                              │  │
│  │  1.10.2 SAVE EDGE INDICES                                    │  │
│  │          └─ edge_index.parquet ([source, target] pairs)     │  │
│  │                                                              │  │
│  │  1.10.3 SAVE EDGE ATTRIBUTES                                 │  │
│  │          ├─ edge_attribute_dist.parquet                      │  │
│  │          └─ edge_attribute_charge.parquet                     │  │
│  │                                                              │  │
│  │  1.10.4 SAVE SEQUENCE FILES                                  │  │
│  │          ├─ sequence/antigen_sequence.json                    │  │
│  │          └─ sequence/cdr_sequence.json                       │  │
│  │                                                              │  │
│  │  1.10.5 SAVE LABELS                                          │  │
│  │          └─ node_label_pi.parquet                            │  │
│  │                                                              │  │
│  │  1.10.6 OUTPUT DIRECTORY STRUCTURE:                          │  │
│  │          processed/<PDB_ID>/                                 │  │
│  │          ├─ node_feature.parquet                             │  │
│  │          ├─ edge_index.parquet                               │  │
│  │          ├─ edge_attribute_dist.parquet                      │  │
│  │          ├─ edge_attribute_charge.parquet                     │  │
│  │          ├─ node_label_pi.parquet                            │  │
│  │          └─ sequence/                                         │  │
│  │             ├─ antigen_sequence.json                          │  │
│  │             └─ cdr_sequence.json                              │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                     │
└───────────────────────────┬───────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│              PROCESSED DATA OUTPUT                                   │
│  /epi4ab/processed/<PDB_ID>/                                       │
│    ├─ node_features.parquet      (50 columns including resShort)   │
│    ├─ edge_index.parquet          (source, target pairs)            │
│    ├─ edge_attribute_dist.parquet (dist column ONLY)                │
│    └─ sequence/                                                    │
│       ├─ antigen_sequence.json   (Antigen amino acid sequence)     │
│       └─ cdr_sequence.json       (H1/H2/H3, L1/L2/L3 sequences)   │
└───────────────────────────┬───────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                STEP 2: VALIDATION                                   │
│              (validate_all.py)                                      │
│                                                                     │
│  2.1 File Existence Checks                                          │
│      └─ Verify all required files present                          │
│                                                                     │
│  2.2 Node Feature Validation                                        │
│      ├─ Column count (50 columns required)                          │
│      ├─ resShort presence and validity                              │
│      ├─ resDepth/caDepth in [0, 30] range                          │
│      ├─ VH/VK/VLa family columns (all 6 VLa variants)              │
│      └─ Value ranges for all features                              │
│                                                                     │
│  2.3 Edge Attribute Validation                                      │
│      ├─ Distance > 0 (no zero values)                              │
│      └─ Correct column selection (dist only)                       │
│                                                                     │
│  2.4 Sequence Consistency                                          │
│      ├─ Sequence file exists                                        │
│      └─ Sequence length matches node count (after deduplication)   │
│                                                                     │
│  2.5 CDR Sequence Check                                             │
│      └─ cdr_sequence.json present (if AntiBERTy enabled)           │
│                                                                     │
└───────────────────────────┬───────────────────────────────────────┘
                            │
                            ▼
                     ✅ Validated Data
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                STEP 3: INFERENCE                                     │
│              (epi_model.py via SLURM)                                │
│                                                                     │
│  3.1 Model Loading                                                  │
│      └─ Load trained GNN from final_trained_Epi4Ab/model.pt       │
│                                                                     │
│  3.2 Data Batching (batch_list function)                            │
│      ├─ Load parquet files                                          │
│      ├─ Build torch-geometric Data objects                          │
│      ├─ Fix: Use edgeAttribute['dist'] (not all columns)          │
│      ├─ Load CDR sequences if AntiBERTy enabled                    │
│      └─ Create batch tensor                                         │
│                                                                     │
│  3.3 Forward Pass                                                   │
│      ├─ GNN ResNet forward                                          │
│      └─ Interface probability scores                                │
│                                                                     │
│  3.4 Output Generation                                              │
│      ├─ Per-residue prediction scores                               │
│      ├─ Evaluation metrics (if labels available)                    │
│      └─ Save to run_model_output/                                   │
│                                                                     │
└───────────────────────────┬───────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    FINAL OUTPUT                                     │
│  /epi4ab/run_model_output/                                          │
│    ├─ <PDB_ID>_predictions.txt/parquet                             │
│    ├─ evaluation_each_pdb.txt (metrics if labels available)        │
│    └─ Summary reports                                                │
└─────────────────────────────────────────────────────────────────────┘
```

## Key Fixes Integrated

### Bug #2: Edge Attribute Loading
- **Before:** `edgeAttribute.to_numpy().reshape(-1)` flattened ALL columns → zeros mixed in
- **After:** `edgeAttribute['dist'].to_numpy().reshape(-1)` → only distances

### Bug #4: Missing resShort
- **Fix:** Explicitly add `resShort` column to node features DataFrame

### Bug #5: Missing VLa Columns
- **Fix:** Added all 6 VLa lambda chain family one-hot columns

### Bug #6: resDepth Out of Range
- **Fix:** Calculate depth relative to antigen CoM, then scale to [0, 30]

### Bug #7: Sequence Mismatch (6B0N)
- **Fix:** Rebuild sequence from deduplicated residues after PDB extraction

### CDR Extraction
- **Fix:** Integrated CDR sequence extraction directly into pipeline

## SLURM Workflow

### Preprocessing (Login Node - Optional but Recommended for Ellipro)
```
# Pre-fetch Ellipro predictions (web API only works on login node)
python preprocess_epitope_predictions.py \
    --pdb_list pdb_list.txt \
    --output_cache /path/to/cache/
         ↓ (~5-10 min per PDB)
Cached predictions ready for compute nodes
```

### Main Processing Pipeline
```
sbatch slurm/process_data.sbatch    # Step 1: Process all PDBs
         ↓ (~60 min)
         ├─ Uses cached Ellipro (if preprocessed)
         └─ Runs BepiPred 3.0 locally on compute nodes
sbatch slurm/validate.sbatch        # Step 2: Validate processed data
         ↓ (~5 min)
sbatch slurm/infer_all.sbatch       # Step 3: Run inference
         ↓ (~60 min)
Results in run_model_output/
```

## Data Flow Key Points

1. **Antigen-Centric Depth**: `resDepth` calculated from antigen center-of-mass, not complex CoM
2. **Deduplication**: Duplicate residues removed to ensure sequence-node consistency
3. **Explicit Column Selection**: Always select DataFrame columns explicitly (especially for edge attributes)
4. **CDR Sequences**: Required for AntiBERTy feature extraction during inference
5. **Sequence Source Priority**: FASTA → RCSB API → PDB extraction (with dedup rebuild)

