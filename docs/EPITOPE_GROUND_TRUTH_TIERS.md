# Epitope ground truth tiers

One residue-level contract per complex: columns align with `node_feature.resId`, same style as `node_iedb_epitope_prior.parquet` (`resId`, `epitope_prior`, `prior_status`, `prior_source`, QC counts where applicable).

| Tier | Script | Status | Output (under `nodes_edges/<pdb_id>/`) |
|------|--------|--------|----------------------------------------|
| 1a | `scripts/epitope_ground_truth/fetch_iedb_epitopes_cohort.py` | Implemented | Cohort CSV (UniProt-wide B-cell positives) |
| 1a | `scripts/build_iedb_epitope_prior.py` | Implemented | `node_iedb_epitope_prior.parquet` + manifest (`prior_source=iedb_sifts`) |
| 1a′ | `scripts/epitope_ground_truth/fetch_iedb_bcell_pdb_scoped.py` | Implemented | CSV: `bcell_export` filtered by `complex__pdb_id` per cohort PDB |
| 1a′ | `scripts/build_iedb_epitope_prior.py --pdb_scope` | Implemented | `node_iedb_pdb_epitope_prior.parquet` (recommended name) + manifest (`iedb_pdb_sifts`) |
| 1b | `scripts/epitope_ground_truth/build_covabdab_epitope_prior.py` | Implemented | `node_covabdab_epitope_prior.parquet` + manifest |
| 1c | `scripts/epitope_ground_truth/merge_augmented_epitope_pdb_list.py` | Implemented | `pdb_list` CSV: complexes whose code appears in CoV-AbDAb ∪ IEDB-PDB harvest |
| 2a | `scripts/epitope_ground_truth/build_pdbe_kb_epitope_prior.py` | Stub | `node_pdbe_kb_epitope_prior.parquet` (placeholder) |
| 2b | `scripts/epitope_ground_truth/build_uniprot_feature_epitope_prior.py` | Stub | `node_uniprot_feature_epitope_prior.parquet` (placeholder) |
| 3 | `scripts/epitope_ground_truth/build_structure_contact_epitope_prior.py` | Implemented | `node_structure_contact_epitope.parquet` (Cα contact mask) |
| Eval | `scripts/eval_dasa_vs_literature_subset.py` | Implemented | Per-complex AUROC CSV + run summaries |

## Tier 1a — IEDB multitype harvest (UniProt-wide)

1. Harvest positive B-cell export rows for each cohort antigen UniProt via IQ-API (`https://query-api.iedb.org/bcell_export`). No filter on `epitope__object_type` (includes discontinuous and other structural types when IEDB encodes positions or a parsable epitope name).
2. Pass the CSV to `build_iedb_epitope_prior.py` as `--iedb_csv`. The builder auto-detects `epitope__molecule_parent_iri` and maps positions including discontinuous names like `S60, L61, Y62`.

**Caveat:** Unioning all rows per UniProt often yields very dense `epitope_prior` masks; prefer Tier 1a′ for structure-aligned literature labels.

```bash
python scripts/epitope_ground_truth/fetch_iedb_epitopes_cohort.py \
  --metadata metadata.csv --out reports/iedb_epitope_multitype.csv

python scripts/build_iedb_epitope_prior.py \
  --metadata metadata.csv --iedb_csv reports/iedb_epitope_multitype.csv \
  --nodes_edges_dir ... --processed_dir ... \
  --sifts_cache_dir ... --infer_uniprot --manifest_out reports/iedb_prior_manifest.csv
```

## Tier 1a′ — IEDB PDB-scoped (IEDB-3D–aligned)

IQ-API `bcell_export` exposes `complex__pdb_id`. Harvest positive assays **per 4-letter PDB** in your cohort, then build with `--pdb_scope` so each complex only sees IEDB rows for **that** structure. By default rows are also intersected with the metadata/inferred antigen UniProt; use `--pdb_scope_skip_uniprot_match` to disable.

```bash
python scripts/epitope_ground_truth/fetch_iedb_bcell_pdb_scoped.py \
  --pdb_list pdb_list.csv --out reports/iedb_bcell_pdb_scoped.csv

python scripts/build_iedb_epitope_prior.py \
  --metadata metadata.csv --iedb_csv reports/iedb_bcell_pdb_scoped.csv \
  --pdb_list pdb_list.csv \
  --nodes_edges_dir ... --processed_dir ... \
  --sifts_cache_dir ... --infer_uniprot --pdb_scope \
  --output_parquet node_iedb_pdb_epitope_prior.parquet \
  --manifest_out reports/iedb_pdb_prior_manifest.csv
```

## Tier 1b — CoV-AbDab

Bulk CSV (e.g. `CoV-AbDab_*.csv` from [OPIG](https://opig.stats.ox.ac.uk/webapps/covabdab/)). Matching uses PDB IDs parsed from the `Structures` column when possible; otherwise rows are matched from `Binds to` + `Protein + Epitope` to canonical SARS-CoV-2 UniProt accessions. Domain-only epitopes (`S; RBD`, etc.) are expanded to approximate UniProt ranges (documented in the script). User-supplied residue lists in other columns are parsed when `--cov_positions_col` is set or auto-detected.

```bash
python scripts/epitope_ground_truth/build_covabdab_epitope_prior.py \
  --covabdab_csv CoV-AbDab_080224.csv --metadata metadata.csv --pdb_list pdb_list.csv \
  --nodes_edges_dir ... --processed_dir ... --sifts_cache_dir ... \
  --infer_uniprot --manifest_out reports/covabdab_prior_manifest.csv
```

## Tier 1c — Augmented `pdb_list` (CoV ∪ IEDB-PDB)

Intersects your existing complex IDs with the union of PDB codes parsed from CoV-AbDAb `Structures` and from an IEDB PDB-scoped CSV (`complex__pdb_id`). Does **not** invent new `pdbID` suffixes—only keeps rows present in `--base_pdb_list` (or `--metadata`).

```bash
python scripts/epitope_ground_truth/merge_augmented_epitope_pdb_list.py \
  --base_pdb_list pdb_list.csv \
  --covabdab_csv CoV-AbDab_080224.csv \
  --iedb_pdb_csv reports/iedb_bcell_pdb_scoped.csv \
  --out reports/pdb_list_epitope_augmented_union.csv
```

Use the output as a preprocessing / label cohort when you want train/test focused on complexes that have **either** curated CoV labels **or** PDB-linked IEDB assays.

## Evaluation — literature subset AUROC

`runs_csv` must set `epitope_prior_file` to the parquet being scored (`node_covabdab_epitope_prior.parquet`, `node_iedb_pdb_epitope_prior.parquet`, or `node_iedb_epitope_prior.parquet`). **`--manifest` must be the manifest produced by the same builder** so `pdb_id` filters line up.

Presets:

| `--truth_tier` | Manifest columns | Typical `epitope_prior_file` |
|----------------|------------------|------------------------------|
| `iedb_uniprot` (default if no tier) | `prior_source == iedb_sifts` | `node_iedb_epitope_prior.parquet` |
| `iedb_pdb` | `prior_source == iedb_pdb_sifts` | `node_iedb_pdb_epitope_prior.parquet` |
| `covabdab` | `covabdab_prior_source == covabdab_sifts` | `node_covabdab_epitope_prior.parquet` |

```bash
python scripts/eval_dasa_vs_literature_subset.py \
  --manifest reports/iedb_prior_manifest.csv \
  --runs_csv runs.csv --score_col true_score
```

CoV-AbDAb example (explicit columns equivalent to `--truth_tier covabdab`):

```bash
python scripts/eval_dasa_vs_literature_subset.py \
  --manifest reports/covabdab_prior_manifest.csv \
  --manifest_prior_source_col covabdab_prior_source \
  --manifest_prior_values covabdab_sifts \
  --runs_csv runs_covabdab.csv --score_col true_score --out_prefix reports/eval_covabdab
```

### Dual sources on the same complex

If a PDB has **both** `node_covabdab_epitope_prior.parquet` and `node_iedb_pdb_epitope_prior.parquet`, **do not OR-merge** them into one binary without a documented rule. Run **separate** eval passes (different `--manifest`, `--runs_csv` `epitope_prior_file`, and `--out_prefix`) and report metrics side by side.

Optional: intersect PDB sets by passing **both** manifests—`eval_dasa_vs_literature_subset.py` supports `--covabdab_manifest` with the primary manifest still filtered by `--manifest_prior_*` (useful when primary is IEDB and secondary is CoV QC).

See `scripts/eval_dasa_vs_literature_subset.py --help`.
