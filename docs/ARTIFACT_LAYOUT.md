# Artifact layout — plots hub

All **derived** visualizations live under:

```text
/leonardo_scratch/fast/EUHPC_D29_035/epi4ab/plots/
```

Training outputs (`model.pt`, `test_record/`) stay next to each job under `training_*` / `OUT_DIR`.

## Directory tree

```text
plots/
  index.html              # master browser index
  manifest.json           # registered runs (RUN_ID metadata)
  cohort/<OUT_BASE_TAG>/
    index.html
    html/                 # study-level Plotly dashboards
    csv/                  # runs_canonical_cohort.csv, comparisons, noise analysis
    literature_dual_by_run/   # optional 3×3 per-run dual HTML + index
  runs/<RUN_ID>/
    html/                 # dashboard.html, literature combined
    png/                  # train_loss, scatter, traces
    csv/                  # regression_summary, lit_eval_*
  legacy/
    classification/       # old run_model_output HTML
    rsa_ab_20260529/      # former Epi4Ab/reports/
```

## Canonical CSV manifests

After migration, use:

| File | Purpose |
|------|---------|
| `cohort/<tag>/csv/runs_canonical_cohort.csv` | Default 191-PDB cohort studies |
| `cohort/<tag>/csv/runs_augmented_iedb_pdb.csv` | Augmented epitope subset eval |

Build/consolidate with:

```bash
python scripts/consolidate_dasa_runs_csv.py
```

## Scripts and default outputs

| Script | Default output |
|--------|----------------|
| `visualize_dasa_backend_comparison.py` | `cohort/<tag>/html/dasa_backend_comparison.html` |
| `visualize_dasa_pred_vs_true_grid.py` | `cohort/<tag>/html/dasa_pred_vs_true_grid.html` |
| `visualize_dasa_literature_combined.py` | `runs/<run_id>/html/dasa_literature_combined.html` |
| `visualize_dasa_literature_all_runs.py` | `cohort/<tag>/literature_dual_by_run/` |
| `visualize_results.py` | `runs/<run_id>/html/dashboard.html` |
| `eval_dasa_vs_literature_subset.py` | `runs/<run_id>/csv/lit_eval_<score_col>_*.csv` |

Override with `--output` / `--out_prefix` when needed.

## Migration (one-time)

```bash
bash scripts/migrate_plots_to_hub.sh --dry-run
bash scripts/migrate_plots_to_hub.sh --apply
python scripts/consolidate_dasa_runs_csv.py
python scripts/build_plots_index.py
```

## Bulk data (not plots)

IEDB harvests, CoV-AbDab, PDBe cache:

```text
OUT_BASE/cache/iedb/
OUT_BASE/cache/covabdab/
OUT_BASE/cache/pdbe/
```
