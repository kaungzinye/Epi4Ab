#!/usr/bin/env python3
"""Copy canonical dasa runs manifests into plots hub csv/ and archive originals."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from epi4ab_plot_paths import cohort_csv_dir, out_base, out_base_tag

CANONICAL = {
    "runs_canonical_cohort.csv": [
        "dasa_runs_20260409.csv",
        "dasa_runs_20260409_with_iedb.csv",
        "dasa_runs_loao_antigen.csv",
    ],
    "runs_augmented_iedb_pdb.csv": [
        "dasa_runs_augmented_iedb_pdb.csv",
        "dasa_runs_augmented_covabdab.csv",
        "dasa_runs_covabdab_eval.csv",
        "dasa_runs_iedb_pdb_eval.csv",
    ],
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports_dir", default=None, help="Source reports dir (default OUT_BASE/reports)")
    ap.add_argument("--out_base_tag", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    tag = args.out_base_tag or out_base_tag()
    dst_csv = cohort_csv_dir(tag)
    if args.reports_dir:
        src_reports = Path(args.reports_dir)
    else:
        cohort_csv = dst_csv
        out_reports = out_base() / "reports"
        src_reports = out_reports if any(out_reports.glob("dasa_runs*.csv")) else cohort_csv
    archive = dst_csv / "archive"
    if not args.dry_run:
        archive.mkdir(parents=True, exist_ok=True)

    for canonical_name, sources in CANONICAL.items():
        chosen = None
        for name in sources:
            p = src_reports / name
            if p.is_file():
                chosen = p
                break
        if not chosen:
            print(f"SKIP {canonical_name}: no source found in {src_reports}")
            continue
        dst = dst_csv / canonical_name
        print(f"{'DRY ' if args.dry_run else ''}WRITE {dst} <- {chosen}")
        if not args.dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(chosen, dst)
        for name in sources:
            p = src_reports / name
            if p.is_file():
                arch_dst = archive / name
                print(f"{'DRY ' if args.dry_run else ''}ARCHIVE {arch_dst} <- {p}")
                if not args.dry_run:
                    shutil.move(str(p), str(arch_dst))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
