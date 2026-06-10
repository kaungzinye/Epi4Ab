#!/usr/bin/env python3
"""
Tier 2a stub: PDBe graph API epitopes for a PDB entry.

Endpoint (verify in implementation): https://www.ebi.ac.uk/pdbe/graph-api/pdbe_pages/epitopes/{pdb_id}

Current behavior:
  - Fetches JSON for one PDB, logs top-level keys to stderr.
  - Writes a schema-compatible placeholder parquet under the first complex in --pdb_list
    with prior_status=stub (unless --dry_run).

Chain matching: epitope records may name chains differently from metadata; TODO map to
antigen chain and SIFTS author numbering before marking implemented.

Examples:
  python scripts/epitope_ground_truth/build_pdbe_kb_epitope_prior.py \\
    --pdb_list pdb_list.csv --nodes_edges_dir ... --pdb_id 1abc_X --dry_run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Set

import pandas as pd
import requests

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

def _collect_pdb_ids_local(args: argparse.Namespace) -> List[str]:
    ids: List[str] = []
    if args.pdb_id:
        ids.extend(str(x) for x in args.pdb_id)
    if args.pdb_list:
        df = pd.read_csv(args.pdb_list, comment="#")
        col = next((c for c in ("pdbID", "pdbId", "pdb_id") if c in df.columns), None)
        if col is None:
            col = df.columns[0]
        ids.extend(df[col].astype(str).tolist())
    seen: Set[str] = set()
    uniq: List[str] = []
    for p in ids:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq


def main() -> int:
    ap = argparse.ArgumentParser(description="PDBe-KB epitope prior (Tier 2a stub).")
    ap.add_argument("--nodes_edges_dir", required=True)
    ap.add_argument("--pdb_id", action="append")
    ap.add_argument("--pdb_list")
    ap.add_argument(
        "--output_parquet",
        default="node_pdbe_kb_epitope_prior.parquet",
    )
    ap.add_argument("--dry_run", action="store_true", help="Fetch and log only; no parquet")
    args = ap.parse_args()

    pdb_ids = _collect_pdb_ids_local(args)
    if not pdb_ids:
        print("Provide --pdb_id and/or --pdb_list", file=sys.stderr)
        return 2

    pdb_code = pdb_ids[0].split("_")[0].lower()[:4]
    url = f"https://www.ebi.ac.uk/pdbe/graph-api/pdbe_pages/epitopes/{pdb_code}"
    session = requests.Session()
    session.headers.update({"User-Agent": "Epi4Ab-PDBe-KB-stub/1.0"})
    try:
        r = session.get(url, timeout=60)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict):
            print("PDBe-KB epitopes JSON keys:", list(data.keys())[:40], file=sys.stderr)
        else:
            print("PDBe-KB epitopes JSON type:", type(data), file=sys.stderr)
    except (requests.RequestException, ValueError) as e:
        print(f"PDBe-KB fetch failed ({e}); writing stub metadata only.", file=sys.stderr)
        data = {"_fetch_error": str(e), "_url": url}

    if args.dry_run:
        return 0

    pdb_id = pdb_ids[0]
    nf_path = Path(args.nodes_edges_dir) / pdb_id / "node_feature.parquet"
    if not nf_path.is_file():
        print(f"Missing {nf_path} (need a valid complex for stub write)", file=sys.stderr)
        return 1

    node_feature = pd.read_parquet(nf_path)
    n = len(node_feature)
    out_df = pd.DataFrame(
        {
            "resId": node_feature["resId"].astype(int),
            "epitope_prior": [0] * n,
            "prior_status": ["pdbe_kb_stub"] * n,
            "prior_source": ["none"] * n,
            "struct_fallback": [0] * n,
            "pdbe_kb_raw_keys": [json.dumps(list(data.keys())[:20]) if isinstance(data, dict) else "[]"] * n,
        }
    )
    out_dir = nf_path.parent
    out_path = out_dir / args.output_parquet
    out_df.to_parquet(out_path, index=False)
    print(f"Wrote placeholder {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
