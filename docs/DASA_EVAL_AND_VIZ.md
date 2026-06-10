# DASA evaluation and visualization

## Which script when

| Goal | Script |
|------|--------|
| One run: DASA + literature on same axis | `visualize_dasa_literature_combined.py --run_id <id>` |
| Full 3×3 backend×seed grid (9 pages) | `visualize_dasa_literature_all_runs.py` (once) |
| Compare backends (Pearson + AUROC bars) | `visualize_dasa_backend_comparison.py` |
| 3×3 pred-vs-true scatter grid | `visualize_dasa_pred_vs_true_grid.py` |
| Label distribution dashboard | `visualize_dasa_labels.py` |
| Literature-subset AUROC CSVs | `eval_dasa_vs_literature_subset.py` |
| Phase 1 / generic regression dashboard | `visualize_results.py` or `generate_dashboard.sh` |

**Prefer** `literature_combined` for a single run. Use `literature_all_runs` only when you need every seed/backend page.

## Runs CSV

Use canonical manifests under the plots hub:

```bash
RUNS=/leonardo_scratch/fast/EUHPC_D29_035/epi4ab/plots/cohort/20260305_143201/csv/runs_canonical_cohort.csv
```

Columns: `run_id`, `backend`, `split_seed`, `nodes_edges_dir`, `test_record_dir`, `label_file`, `epitope_prior_file`.

## Examples

```bash
cd /leonardo_work/EUHPC_D29_035/Epi4Ab
export RUN_ID=v103_biopy_s42

python scripts/visualize_dasa_literature_combined.py \
  --runs_csv "$RUNS" --run_id "$RUN_ID" \
  --processed_dir .../processed_data --build-index

python scripts/eval_dasa_vs_literature_subset.py \
  --manifest .../cache/iedb/iedb_pdb_prior_manifest.csv \
  --truth_tier iedb_pdb \
  --runs_csv "$RUNS" --score_col pred_score --run_id "$RUN_ID"
```

## Naming

- Do **not** date-suffix HTML filenames in `reports/`.
- Use `RUN_ID` folders under `plots/runs/`.
- Regenerate index: `python scripts/build_plots_index.py` or pass `--build-index` to viz scripts.

Open **`plots/index.html`** to browse everything.
