#!/usr/bin/env python3
"""Build master and per-cohort index.html pages for the plots hub."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from epi4ab_plot_paths import cohort_dir, load_manifest, manifest_path, plots_root


def _link_list(files: list[Path], base: Path) -> str:
    if not files:
        return "<p><em>No files yet.</em></p>"
    items = []
    for f in sorted(files):
        rel = f.relative_to(base)
        items.append(f'<li><a href="{html.escape(str(rel))}">{html.escape(f.name)}</a></li>')
    return "<ul>\n" + "\n".join(items) + "\n</ul>"


def _section(title: str, body: str) -> str:
    return f"<h2>{html.escape(title)}</h2>\n{body}\n"


def build_cohort_index(tag: str, hub: Path) -> Path:
    cdir = cohort_dir(tag)
    out = cdir / "index.html"
    if not cdir.is_dir():
        return out

    parts = [f"<h1>Cohort plots — {html.escape(tag)}</h1>"]
    html_dir = cdir / "html"
    if html_dir.is_dir():
        parts.append(_section("Study dashboards", _link_list(list(html_dir.glob("*.html")), cdir)))
    csv_dir = cdir / "csv"
    if csv_dir.is_dir():
        parts.append(_section("CSV manifests & summaries", _link_list(list(csv_dir.glob("*.csv")), cdir)))
    dual = cdir / "literature_dual_by_run"
    if dual.is_dir() and (dual / "index.html").is_file():
        parts.append(
            '<p><a href="literature_dual_by_run/index.html">Literature dual-track (all runs)</a></p>'
        )
    parts.append('<p><a href="../../index.html">← Master plots index</a></p>')
    out.write_text(
        "<!DOCTYPE html>\n<html><head><meta charset=\"utf-8\"/>"
        f"<title>Cohort {tag}</title></head><body>\n"
        + "\n".join(parts)
        + "\n</body></html>",
        encoding="utf-8",
    )
    return out


def build_master_index(hub: Path) -> Path:
    out = hub / "index.html"
    hub.mkdir(parents=True, exist_ok=True)
    parts = [
        "<h1>Epi4Ab plots hub</h1>",
        f"<p>Root: <code>{html.escape(str(hub))}</code></p>",
    ]

    manifest = load_manifest()
    if manifest:
        rows = [
            "<table border='1' cellpadding='4'><tr><th>run_id</th><th>backend</th>"
            "<th>seed</th><th>out_dir</th></tr>"
        ]
        for e in sorted(manifest, key=lambda x: x.get("run_id", "")):
            rid = html.escape(str(e.get("run_id", "")))
            link = f'<a href="runs/{rid}/html/">{rid}</a>' if (hub / "runs" / rid).is_dir() else rid
            rows.append(
                f"<tr><td>{link}</td><td>{html.escape(str(e.get('backend', '')))}</td>"
                f"<td>{html.escape(str(e.get('split_seed', '')))}</td>"
                f"<td><code>{html.escape(str(e.get('out_dir', '')))}</code></td></tr>"
            )
        rows.append("</table>")
        parts.append(_section("Registered runs (manifest.json)", "\n".join(rows)))

    cohort_root = hub / "cohort"
    if cohort_root.is_dir():
        tags = sorted(p.name for p in cohort_root.iterdir() if p.is_dir())
        links = "".join(
            f'<li><a href="cohort/{html.escape(t)}/index.html">{html.escape(t)}</a></li>'
            for t in tags
        )
        parts.append(_section("Cohorts", f"<ul>{links}</ul>"))

    runs_root = hub / "runs"
    if runs_root.is_dir():
        run_ids = sorted(p.name for p in runs_root.iterdir() if p.is_dir())
        links = "".join(
            f'<li><a href="runs/{html.escape(r)}/html/">{html.escape(r)}</a></li>'
            for r in run_ids
        )
        parts.append(_section("Per-run plots", f"<ul>{links}</ul>"))

    legacy_root = hub / "legacy"
    if legacy_root.is_dir():
        names = sorted(p.name for p in legacy_root.iterdir() if p.is_dir())
        links = "".join(
            f'<li><a href="legacy/{html.escape(n)}/">{html.escape(n)}</a></li>'
            for n in names
        )
        parts.append(_section("Legacy", f"<ul>{links}</ul>"))

    if manifest_path().is_file():
        parts.append(f'<p><a href="manifest.json">manifest.json</a></p>')

    out.write_text(
        "<!DOCTYPE html>\n<html><head><meta charset=\"utf-8\"/>"
        "<title>Epi4Ab plots hub</title></head><body>\n"
        + "\n".join(parts)
        + "\n</body></html>",
        encoding="utf-8",
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Build plots hub index.html files")
    ap.add_argument("--plots_root", default=None)
    args = ap.parse_args()
    hub = Path(args.plots_root) if args.plots_root else plots_root()
    build_master_index(hub)
    cohort_root = hub / "cohort"
    if cohort_root.is_dir():
        for tag_dir in cohort_root.iterdir():
            if tag_dir.is_dir():
                build_cohort_index(tag_dir.name, hub)
    print(f"Wrote {hub / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
