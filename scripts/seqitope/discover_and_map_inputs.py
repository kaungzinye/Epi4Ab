#!/usr/bin/env python3
"""
Read-only discovery of potential Seqitope TSV/CSV sources. Does not modify files.

Outputs a JSON manifest at seqitope_discovery/report.json with entries:
  - path
  - size_bytes
  - mtime_iso
  - header_sample (first line)
  - guessed_has_resid / guessed_has_score (heuristics)

Search roots (override by --root multiple times):
  - EpiScan-Private
  - Epi4Ab
  - current working directory
"""
import argparse, os, json, time
from pathlib import Path

CAND_EXT = ('.tsv', '.csv')


def scan_root(root: Path):
    results = []
    for p in root.rglob('*'):
        if not p.is_file():
            continue
        if p.suffix.lower() not in CAND_EXT:
            continue
        try:
            size = p.stat().st_size
            mtime = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(p.stat().st_mtime))
            with p.open('r', errors='ignore') as f:
                head = f.readline().strip()[:500]
        except Exception:
            size, mtime, head = None, None, ''
        guessed_resid = any(k in head.lower() for k in ['resid', 'res_id'])
        guessed_score = any(k in head.lower() for k in ['score', 'prob', 'propensity'])
        results.append({
            'path': str(p),
            'size_bytes': size,
            'mtime_iso': mtime,
            'header_sample': head,
            'guessed_has_resid': guessed_resid,
            'guessed_has_score': guessed_score,
        })
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', action='append', help='Search root (repeatable)')
    args = ap.parse_args()
    roots = [Path(r) for r in (args.root or ['EpiScan-Private', 'Epi4Ab', '.'])]
    report = []
    for r in roots:
        if r.exists():
            report.extend(scan_root(r))
    out_dir = Path('seqitope_discovery')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / 'report.json'
    with out_file.open('w') as f:
        json.dump(report, f, indent=2)
    print(f"wrote {out_file} with {len(report)} candidates")

if __name__ == '__main__':
    main()
