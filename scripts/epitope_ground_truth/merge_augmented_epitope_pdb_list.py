#!/usr/bin/env python3
"""
Build an augmented pdb_list CSV: complexes whose 4-letter code appears in
CoV-AbDAb ``Structures`` and/or in an IEDB PDB-scoped harvest (``complex__pdb_id``).

Does not invent ``pdbID`` suffixes: every output row is taken from ``--base_pdb_list``
(or metadata ``pdbID``) so IDs match your preprocess / nodes_edges layout.

Examples:
  python scripts/epitope_ground_truth/merge_augmented_epitope_pdb_list.py \\
    --base_pdb_list gates/nodes_edges/pdb_list.csv \\
    --covabdab_csv CoV-AbDab_080224.csv \\
    --iedb_pdb_csv reports/iedb_bcell_pdb_scoped.csv \\
    --out reports/pdb_list_epitope_augmented.csv
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Optional, Set

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.build_iedb_epitope_prior import (  # noqa: E402
    _read_metadata_rows,
)
from scripts.epitope_ground_truth.build_covabdab_epitope_prior import (  # noqa: E402
    PDB_IN_URL,
    PDB_TOKEN,
)


def _codes_from_covabdab_csv(path: str) -> Set[str]:
    df = pd.read_csv(path, low_memory=False, encoding="utf-8-sig")
    df.columns = [str(c).lstrip("\ufeff") for c in df.columns]
    col = "Structures" if "Structures" in df.columns else None
    if not col:
        return set()
    out: Set[str] = set()
    for cell in df[col].astype(str):
        if not cell or cell.lower() in ("nan", "none"):
            continue
        for m in PDB_IN_URL.finditer(cell):
            out.add(m.group(1).upper()[:4])
        for m in PDB_TOKEN.finditer(cell):
            g = m.group(1).upper()[:4]
            if re.match(r"^[0-9][A-Z0-9]{3}$", g):
                out.add(g)
    return out


def _codes_from_iedb_pdb_csv(path: str, col: str) -> Set[str]:
    df = pd.read_csv(path, low_memory=False)
    if col not in df.columns:
        return set()
    out: Set[str] = set()
    for val in df[col].astype(str):
        v = val.strip().upper().split("_")[0][:4]
        if len(v) == 4 and re.match(r"^[0-9][A-Z0-9]{3}$", v):
            out.add(v)
    return out


def _load_base_ids(metadata: Optional[str], pdb_list: Optional[str]) -> pd.DataFrame:
    if pdb_list:
        df = pd.read_csv(pdb_list, comment="#")
        col = next((c for c in ("pdbID", "pdbId", "pdb_id") if c in df.columns), None)
        if col is None:
            col = df.columns[0]
        return pd.DataFrame({"pdbID": df[col].astype(str)})
    if metadata:
        meta = _read_metadata_rows(metadata)
        if "pdbID" not in meta.columns:
            raise ValueError("metadata must have pdbID")
        return meta[["pdbID"]].drop_duplicates()
    raise ValueError("pass --base_pdb_list or --metadata")


def main() -> int:
    ap = argparse.ArgumentParser(description="Augmented pdb_list: CoV-AbDAb + IEDB PDB codes ∩ base cohort.")
    ap.add_argument("--base_pdb_list", help="Full cohort pdb_list.csv (pdbID column)")
    ap.add_argument("--metadata", help="Alternative: metadata CSV with pdbID")
    ap.add_argument("--covabdab_csv", help="CoV-AbDAb bulk CSV from OPIG")
    ap.add_argument("--iedb_pdb_csv", help="Output of fetch_iedb_bcell_pdb_scoped.py")
    ap.add_argument(
        "--iedb_pdb_col",
        default="complex__pdb_id",
        help="PDB id column in IEDB CSV",
    )
    ap.add_argument("--out", required=True, help="Output pdb_list CSV with pdbID column")
    args = ap.parse_args()

    base = _load_base_ids(args.metadata, args.base_pdb_list)
    want: Set[str] = set()
    if args.covabdab_csv:
        want |= _codes_from_covabdab_csv(args.covabdab_csv)
    if args.iedb_pdb_csv:
        want |= _codes_from_iedb_pdb_csv(args.iedb_pdb_csv, args.iedb_pdb_col)

    if not want:
        print("No PDB codes from --covabdab_csv / --iedb_pdb_csv; nothing to merge.", file=sys.stderr)
        return 2

    def code_of(pid: str) -> str:
        return str(pid).strip().split("_")[0].upper()[:4]

    mask = base["pdbID"].map(code_of).isin(want)
    sub = base.loc[mask].drop_duplicates()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_path, index=False)
    print(
        f"Wrote {out_path}  n_output={len(sub)}  "
        f"distinct_codes_in_sources={len(want)}  base_complexes={len(base)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
