#!/usr/bin/env python3
"""
Harvest IEDB IQ-API bcell_export rows filtered by experimental complex PDB id.

Uses ``complex__pdb_id`` on ``bcell_export`` (IEDB-3D–linked assays). Positive
B-cell qualitative measure only, same as ``fetch_iedb_epitopes_cohort.py``.

Typical flow:
  1. This script → ``reports/iedb_bcell_pdb_scoped_<cohort>.csv``
  2. ``build_iedb_epitope_prior.py --pdb_scope --iedb_csv that.csv ...``
     (optionally ``--output_parquet node_iedb_pdb_epitope_prior.parquet``)

Examples:
  python scripts/epitope_ground_truth/fetch_iedb_bcell_pdb_scoped.py \\
    --pdb_list pdb_list.csv --out reports/iedb_bcell_pdb_scoped.csv

  python scripts/epitope_ground_truth/fetch_iedb_bcell_pdb_scoped.py \\
    --metadata metadata.csv --out reports/iedb_bcell_pdb_scoped.csv
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import List, Set

import pandas as pd
import requests

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.build_iedb_epitope_prior import (  # noqa: E402
    _read_metadata_rows,
)

IEDB_BCELL_EXPORT = "https://query-api.iedb.org/bcell_export"


def _pdb_codes_from_pdb_list(path: str) -> Set[str]:
    df = pd.read_csv(path, comment="#")
    col = next((c for c in ("pdbID", "pdbId", "pdb_id") if c in df.columns), None)
    if col is None:
        col = df.columns[0]
    out: Set[str] = set()
    for raw in df[col].astype(str):
        code = raw.strip().split("_")[0].upper()[:4]
        if len(code) == 4 and re.match(r"^[0-9][A-Z0-9]{3}$", code):
            out.add(code)
    return out


def _pdb_codes_from_metadata(path: str) -> Set[str]:
    df = _read_metadata_rows(path)
    if "pdbID" not in df.columns:
        raise ValueError(f"metadata needs pdbID: {path}")
    out: Set[str] = set()
    for raw in df["pdbID"].astype(str):
        code = raw.strip().split("_")[0].upper()[:4]
        if len(code) == 4 and re.match(r"^[0-9][A-Z0-9]{3}$", code):
            out.add(code)
    return out


def fetch_bcell_export_for_pdb(
    session: requests.Session,
    pdb_code: str,
    page_size: int = 500,
    sleep_s: float = 0.15,
) -> List[dict]:
    """Positive B-cell assays with complex__pdb_id equal to this 4-letter PDB."""
    code = pdb_code.strip().upper()[:4]
    rows: List[dict] = []
    offset = 0
    while True:
        params = {
            "complex__pdb_id": f"eq.{code}",
            "assay__qualitative_measure": "eq.Positive",
            "order": "assay_id.asc",
            "limit": str(page_size),
            "offset": str(offset),
        }
        r = session.get(IEDB_BCELL_EXPORT, params=params, timeout=120)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        if not isinstance(batch, list):
            raise RuntimeError(f"Unexpected JSON for {code}: {type(batch)}")
        rows.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size
        time.sleep(sleep_s)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(
        description="IEDB bcell_export harvest: filter by complex PDB id (IEDB-3D–aligned)."
    )
    ap.add_argument("--metadata", help="Metadata CSV with pdbID column")
    ap.add_argument("--pdb_list", help="CSV with pdbID / first column of complex ids")
    ap.add_argument(
        "--pdb_code",
        action="append",
        help="4-letter PDB code (repeatable); if none, use --metadata / --pdb_list",
    )
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--page_size", type=int, default=500)
    ap.add_argument("--sleep", type=float, default=0.15)
    args = ap.parse_args()

    codes: Set[str] = set()
    if args.pdb_code:
        for c in args.pdb_code:
            u = c.strip().upper()[:4]
            if len(u) == 4:
                codes.add(u)
    if args.metadata:
        codes |= _pdb_codes_from_metadata(args.metadata)
    if args.pdb_list:
        codes |= _pdb_codes_from_pdb_list(args.pdb_list)

    codes = {c for c in codes if len(c) == 4}
    if not codes:
        print("No PDB codes; pass --metadata, --pdb_list, and/or --pdb_code.", file=sys.stderr)
        return 2

    session = requests.Session()
    session.headers.update({"User-Agent": "Epi4Ab-IEDB-pdb-scoped/1.0"})

    all_rows: List[dict] = []
    for code in sorted(codes):
        try:
            part = fetch_bcell_export_for_pdb(session, code, args.page_size, args.sleep)
            print(f"{code}: {len(part)} rows")
            all_rows.extend(part)
        except requests.RequestException as e:
            print(f"FAIL {code}: {e}", file=sys.stderr)
            return 1

    if not all_rows:
        pd.DataFrame().to_csv(args.out, index=False)
        print(f"Wrote empty {args.out}")
        return 0

    df = pd.DataFrame(all_rows)
    if "assay_id" in df.columns:
        df = df.drop_duplicates(subset=["assay_id"], keep="first")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}  unique_rows={len(df)}  pdb_codes={len(codes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
