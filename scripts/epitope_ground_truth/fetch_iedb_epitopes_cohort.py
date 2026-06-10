#!/usr/bin/env python3
"""
Harvest IEDB IQ-API bcell_export rows for a cohort’s antigen UniProt accessions.

Includes positive B-cell assays for all epitope structure types (linear, discontinuous,
etc.) available in bcell_export — not limited to linear peptides.

Output CSV retains IQ-API column names so scripts/build_iedb_epitope_prior.py can use
epitope__molecule_parent_iri (auto-detected) and extended position / name parsing.

Examples:
  python scripts/epitope_ground_truth/fetch_iedb_epitopes_cohort.py \\
    --metadata gates_metadata.csv --out reports/iedb_epitope_multitype.csv

  python scripts/epitope_ground_truth/fetch_iedb_epitopes_cohort.py \\
    --uniprots P0DTC2,P08669 --out reports/iedb_subset.csv
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
    UNIPROT_ACC_RE,
    _read_metadata_rows,
)

IEDB_BCELL_EXPORT = "https://query-api.iedb.org/bcell_export"


def _accessions_from_metadata(metadata_path: str) -> List[str]:
    df = _read_metadata_rows(metadata_path)
    up_candidates = (
        "antigen_uniprot",
        "antigenUniProt",
        "UniProt",
        "uniprot",
        "Antigen UniProt ID",
    )
    accs: Set[str] = set()
    for _, row in df.iterrows():
        for c in up_candidates:
            if c in df.columns and pd.notna(row.get(c)):
                s = str(row[c]).strip().upper()
                if not s:
                    continue
                m = UNIPROT_ACC_RE.search(s)
                accs.add(m.group(1) if m else s.split()[0])
                break
    return sorted(accs)


def _parse_uniprot_list_file(path: str) -> List[str]:
    accs: Set[str] = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = UNIPROT_ACC_RE.search(line.upper())
        if m:
            accs.add(m.group(1))
    return sorted(accs)


def _parse_uniprot_csv_arg(s: str) -> List[str]:
    accs: Set[str] = set()
    for part in re.split(r"[,;\s]+", s.strip()):
        if not part:
            continue
        m = UNIPROT_ACC_RE.search(part.upper())
        if m:
            accs.add(m.group(1))
    return sorted(accs)


def fetch_bcell_export_for_uniprot(
    session: requests.Session,
    accession: str,
    page_size: int = 500,
    sleep_s: float = 0.15,
) -> List[dict]:
    """Positive B-cell assays for epitopes whose molecule_parent_iri is this UniProt."""
    iri = f"http://www.uniprot.org/uniprot/{accession}"
    rows: List[dict] = []
    offset = 0
    while True:
        params = {
            "epitope__molecule_parent_iri": f"eq.{iri}",
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
            raise RuntimeError(f"Unexpected JSON for {accession}: {type(batch)}")
        rows.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size
        time.sleep(sleep_s)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="IEDB IQ-API multitype bcell_export harvest for cohort UniProts.")
    ap.add_argument("--metadata", help="gates/metadata CSV with antigen UniProt column")
    ap.add_argument("--uniprots", help="Comma-separated UniProt accessions")
    ap.add_argument("--uniprot_list_file", help="Text file: one accession per line")
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--page_size", type=int, default=500)
    ap.add_argument("--sleep", type=float, default=0.15, help="Delay between paginated requests")
    args = ap.parse_args()

    accs: List[str] = []
    if args.metadata:
        accs.extend(_accessions_from_metadata(args.metadata))
    if args.uniprots:
        accs.extend(_parse_uniprot_csv_arg(args.uniprots))
    if args.uniprot_list_file:
        accs.extend(_parse_uniprot_list_file(args.uniprot_list_file))

    accs = sorted({a for a in accs if a})
    if not accs:
        print(
            "No UniProt accessions; pass --metadata and/or --uniprots / --uniprot_list_file.",
            file=sys.stderr,
        )
        return 2

    session = requests.Session()
    session.headers.update({"User-Agent": "Epi4Ab-IEDB-multitype/1.0"})

    all_rows: List[dict] = []
    for acc in accs:
        try:
            part = fetch_bcell_export_for_uniprot(
                session, acc, page_size=args.page_size, sleep_s=args.sleep
            )
            print(f"{acc}: {len(part)} rows")
            all_rows.extend(part)
        except requests.RequestException as e:
            print(f"FAIL {acc}: {e}", file=sys.stderr)
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
    print(f"Wrote {out_path}  unique_rows={len(df)}  accessions={len(accs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
