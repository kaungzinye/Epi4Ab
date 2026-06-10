#!/usr/bin/env python3
"""
Generate dual-track literature+DASA HTML for every row in runs_csv (3 backends × 3 seeds).

Writes one HTML per run_id plus a small index page linking them all.

Example:
  python scripts/visualize_dasa_literature_all_runs.py \\
    --runs_csv reports/dasa_runs_augmented_iedb_pdb.csv \\
    --processed_dir .../processed_data \\
    --output_dir reports/literature_dual_by_run \\
    --dual_track --cols 2 --width 1600 --height_per_row 360
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

from epi4ab_plot_paths import cohort_literature_dual_dir, maybe_build_index, out_base_tag

_REPO = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser(description="Batch dual-track literature plots for all runs.")
    ap.add_argument("--runs_csv", required=True)
    ap.add_argument("--processed_dir", required=True)
    ap.add_argument(
        "--output_dir",
        default=None,
        help="Output dir (default: plots/cohort/<tag>/literature_dual_by_run/)",
    )
    ap.add_argument("--out_base_tag", default=None)
    ap.add_argument("--build-index", action="store_true")
    ap.add_argument("--dual_track", action="store_true")
    ap.add_argument("--cols", type=int, default=2)
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--height_per_row", type=int, default=360)
    ap.add_argument(
        "--epitope_assay_prior_file",
        default="node_iedb_pdb_epitope_prior_assay.parquet",
    )
    args = ap.parse_args()

    runs = pd.read_csv(args.runs_csv, comment="#")
    tag = args.out_base_tag or out_base_tag()
    out_dir = Path(args.output_dir) if args.output_dir else cohort_literature_dual_dir(tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    py = _REPO / "venv/bin/python"
    script = _REPO / "scripts/visualize_dasa_literature_combined.py"
    links: list[str] = []

    for _, row in runs.iterrows():
        run_id = str(row["run_id"])
        out_html = out_dir / f"dasa_literature_dual_{run_id}.html"
        cmd = [
            str(py),
            str(script),
            "--runs_csv",
            args.runs_csv,
            "--run_id",
            run_id,
            "--processed_dir",
            args.processed_dir,
            "--output",
            str(out_html),
            "--cols",
            str(args.cols),
            "--width",
            str(args.width),
            "--height_per_row",
            str(args.height_per_row),
        ]
        if args.dual_track:
            cmd.append("--dual_track")
        if args.epitope_assay_prior_file:
            cmd.extend(["--epitope_assay_prior_file", args.epitope_assay_prior_file])
        print("RUN", " ".join(cmd[-8:]))
        rc = subprocess.run(cmd, cwd=str(_REPO))
        if rc.returncode != 0:
            print(f"FAILED {run_id}", file=sys.stderr)
            return rc.returncode
        backend = row.get("backend", "?")
        seed = row.get("split_seed", "?")
        label = row.get("label_file", "?")
        links.append(
            f'<li><a href="{out_html.name}">{run_id}</a> '
            f"— backend <b>{backend}</b>, split seed <b>{seed}</b>, "
            f"labels <code>{label}</code></li>"
        )

    index = out_dir / "index.html"
    index.write_text(
        f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/><title>DASA literature dual — all runs</title></head>
<body>
<h1>DASA + IEDB literature (dual track) — 3×3 runs</h1>
<p>Each page: all <b>test-set</b> complexes for that training run.
Blue/orange DASA uses the backend-specific label parquet listed below.
Light red = union literature; solid red = contact-gated matched assay.</p>
<ul>
{chr(10).join(links)}
</ul>
</body></html>""",
        encoding="utf-8",
    )
    print(f"Wrote index {index.resolve()} ({len(links)} runs)")
    maybe_build_index(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
