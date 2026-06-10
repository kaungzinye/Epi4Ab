# Storage and drives (Leonardo)

## Drive map

| Location | Path | Contents | Git |
|----------|------|----------|-----|
| Work | `/leonardo_work/EUHPC_D29_035/Epi4Ab/` | Code, SLURM, docs | Yes |
| Fast scratch | `/leonardo_scratch/fast/EUHPC_D29_035/epi4ab/` | Preprocess, training, **plots hub** | No |
| Large scratch | `/leonardo_scratch/large/userexternal/knaung00/epi4ab_archive/` | Archived superseded runs | No |

## Environment variables

See `.env.example`:

- `EPI4AB_SCRATCH` — fast scratch root (`.../epi4ab`)
- `EPI4AB_PLOTS_ROOT` — `${EPI4AB_SCRATCH}/plots`
- `EPI4AB_OUT_BASE_TAG` — canonical preprocess tag (default `20260305_143201`)
- `EPI4AB_LARGE_ARCHIVE` — cold storage for tarballs
- `RUN_ID` — short id for `plots/runs/<RUN_ID>/` when training or visualizing

## Canonical active paths

```text
epi4ab/
  plots/                                    # all HTML/PNG/eval CSV summaries
  upstream_preprocess_autodetect/20260305_143201/   # OUT_BASE (191 PDBs)
    cache/                                  # IEDB, CoV-AbDab, PDBe JSON
    reports/README.txt                      # pointer to plots hub
  training_dasa_v103/                       # active DASA training
```

## Archiving heavy data

Superseded directories (old preprocess pilots, `training_dasa_matrix*`, `run_model_output`) can be moved to large scratch:

```bash
cd /leonardo_work/EUHPC_D29_035/Epi4Ab
bash scripts/export_heavy_to_large.sh --dry-run
bash scripts/export_heavy_to_large.sh --apply --label 20260610
```

Plots are **not** deleted — they live under `epi4ab/plots/` before archiving training dirs.

## Never commit

- `reports/` under the repo
- `*.html`, plot PNGs, `dasa_runs*.csv`, parquets, scratch paths

See `.gitignore` and [ARTIFACT_LAYOUT.md](ARTIFACT_LAYOUT.md).
