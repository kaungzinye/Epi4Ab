# Epi4Ab Pipeline Documentation

**This file is deprecated.**

## Current workflow (regression / DASA)

- [PIPELINE_QUICKSTART.md](PIPELINE_QUICKSTART.md) — preprocess → Phase 1 → optional DASA
- [PHASE1_REGRESSION_PIPELINE.md](PHASE1_REGRESSION_PIPELINE.md) — regression details
- [DASA_LABEL_PIPELINE.md](DASA_LABEL_PIPELINE.md) — Delta-ASA labels
- [DASA_EVAL_AND_VIZ.md](DASA_EVAL_AND_VIZ.md) — plots and eval scripts
- [ARTIFACT_LAYOUT.md](ARTIFACT_LAYOUT.md) — where outputs live on scratch
- [STORAGE_AND_DRIVES.md](STORAGE_AND_DRIVES.md) — Leonardo drive map

## Legacy workflow (classification, isInterface 0/1/2)

The original Tran et al. paper pipeline (`node_label_pi.parquet`, cross-entropy, test3A/EpiScan) is documented in:

- [LEGACY_CLASSIFICATION_PIPELINE.md](LEGACY_CLASSIFICATION_PIPELINE.md)

Note: regression `score ∈ [0,1]` is **not** the same as legacy categorical `isInterface ∈ {0,1,2}`.
