# Local backup procedure (Leonardo → your hard drive)

Training data, preprocess outputs, models, and plots live on **scratch**, not in git. Use this procedure before project end or when you need an offline copy.

## What goes where

| Content | On Leonardo | In git? | Local backup? |
|---------|-------------|---------|---------------|
| Source code, docs, SLURM | `leonardo_work/.../Epi4Ab` | **Yes** (GitHub fork) | `git clone` is enough |
| Plots hub (HTML/PNG/CSV) | `epi4ab/plots/` | No | **Yes** — bundle `epi4ab_plots` |
| Preprocess + labels (OUT_BASE) | `upstream_preprocess_autodetect/20260305_143201/` | No | **Yes** — bundle `epi4ab_outbase_*` |
| Active training (checkpoints, test_record) | `training_dasa_v103/`, `training_phase1/` | No | **Yes** |
| IEDB/CoV caches | `OUT_BASE/cache/` | No | Included in OUT_BASE bundle |
| Superseded runs (cold) | `epi4ab_archive/` on large scratch | No | Optional (`--include-large-archive`) |
| `tools/conda/`, scratch junk | various | No | **No** — reinstall from docs |

**Do not commit** parquets, PDBs, `model.pt`, plots, or multi-GB CSVs to GitHub. `.gitignore` blocks most of this.

## Step 1 — Create bundles on Leonardo (login node)

```bash
cd /leonardo_work/EUHPC_D29_035/Epi4Ab

# Preview
bash scripts/export_portable_bundles.sh --dry-run

# Full backup (plots + OUT_BASE + DASA training + Phase 1)
bash scripts/export_portable_bundles.sh --apply --include-phase1

# Optional: include cold archive from large scratch
bash scripts/export_portable_bundles.sh --apply --include-phase1 --include-large-archive
```

Output directory (default):

```text
/leonardo_scratch/fast/EUHPC_D29_035/epi4ab/portable_bundles/<YYYYMMDD>/
  MANIFEST.txt
  epi4ab_plots.tar.gz
  epi4ab_outbase_20260305_143201.tar.gz
  epi4ab_training_dasa_v103.tar.gz
  epi4ab_training_phase1.tar.gz          # if --include-phase1
```

Override tag or destination:

```bash
EPI4AB_OUT_BASE_TAG=20260305_143201 \
BACKUP_DEST=/leonardo_scratch/fast/EUHPC_D29_035/epi4ab/portable_bundles/my_backup \
bash scripts/export_portable_bundles.sh --apply --label my_backup
```

## Step 2 — Download to your laptop

From your **local machine** (replace `USER`):

```bash
mkdir -p ~/Backups/Epi4Ab/leonardo_$(date +%Y%m%d)

rsync -avP --progress \
  USER@login.leonardo.cineca.it:/leonardo_scratch/fast/EUHPC_D29_035/epi4ab/portable_bundles/*/ \
  ~/Backups/Epi4Ab/leonardo_$(date +%Y%m%d)/
```

Single file:

```bash
scp USER@login.leonardo.cineca.it:/leonardo_scratch/fast/EUHPC_D29_035/epi4ab/portable_bundles/*/epi4ab_plots.tar.gz \
  ~/Backups/Epi4Ab/
```

## Step 3 — Restore locally (optional)

```bash
mkdir -p ~/Epi4Ab_restore
tar -xzf epi4ab_plots.tar.gz -C ~/Epi4Ab_restore
open ~/Epi4Ab_restore/plots/index.html   # macOS
```

Training/preprocess trees extract with the same `tar -xzf` pattern.

## Related Leonardo-only scripts

| Script | Purpose |
|--------|---------|
| `scripts/export_portable_bundles.sh` | Tarballs for **local** download (this doc) |
| `scripts/export_heavy_to_large.sh` | Move **superseded** dirs to large scratch on Leonardo |
| `scripts/migrate_plots_to_hub.sh` | One-time plots hub migration |

See also [STORAGE_AND_DRIVES.md](STORAGE_AND_DRIVES.md) and [ARTIFACT_LAYOUT.md](ARTIFACT_LAYOUT.md).

## Minimum vs full backup

| Tier | Bundles | When |
|------|---------|------|
| **Minimum** | `epi4ab_plots` only | Slides, papers, quick offline viewing |
| **Standard** | plots + outbase + `training_dasa_v103` | Reproduce DASA work locally or on another HPC |
| **Full** | standard + `training_phase1` + large archive | Complete project archive |

## Already on GitHub?

These **docs and scripts** are in the repo (`develop` on your fork). **Data is not** — only the procedure to export it.
