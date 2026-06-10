# Epitope ground truth scripts

Index, inputs, outputs, and run order are documented in:

- `docs/EPITOPE_GROUND_TRUTH_TIERS.md`

Tier 1 builders live in this directory; `build_iedb_epitope_prior.py` remains at `scripts/` and consumes IQ-API CSVs produced here (`fetch_iedb_epitopes_cohort.py` = UniProt-wide; `fetch_iedb_bcell_pdb_scoped.py` = per-PDB `complex__pdb_id`). `merge_augmented_epitope_pdb_list.py` builds CoV-AbDAb ∪ IEDB-PDB augmented `pdb_list` slices.
