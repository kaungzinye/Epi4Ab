#!/usr/bin/env python3
"""
Tier 3: binary Cα distance epitope mask — antigen residues within --contact_cutoff Å of any
non-antigen chain atom (same heuristic as build_iedb_epitope_prior --struct_fallback).

Writes node_structure_contact_epitope.parquet per complex with full cohort coverage.

Examples:
  python scripts/epitope_ground_truth/build_structure_contact_epitope_prior.py \\
    --metadata metadata.csv --pdb_list pdb_list.csv \\
    --nodes_edges_dir .../nodes_edges --processed_dir .../processed_data \\
    --manifest_out reports/structure_contact_prior_manifest.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.build_iedb_epitope_prior import (  # noqa: E402
    AsaFullKey,
    _metadata_dict_from_df,
    _project_masks_to_node_rows,
    _read_metadata_rows,
    _structure_contact_mask_for_chain,
)
from scripts.generate_dasa_labels import (  # noqa: E402
    _build_chain_residue_order,
    _find_complex_structure,
    _parse_pdb_residues,
    _split_chain_field,
)


def _collect_pdb_ids(args: argparse.Namespace) -> List[str]:
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


def process_one(
    pdb_id: str,
    meta_row: Dict[str, str],
    nodes_edges_dir: str,
    processed_dir: str,
    output_name: str,
    contact_cutoff: float,
) -> Tuple[Path, dict]:
    antigen_chain = _split_chain_field(meta_row.get("antigen", ""))
    pdb_code = (meta_row.get("pdb") or "").lower().strip() or pdb_id.split("_")[0].lower()

    nf_path = os.path.join(nodes_edges_dir, pdb_id, "node_feature.parquet")
    if not os.path.exists(nf_path):
        raise FileNotFoundError(nf_path)
    lig_path = os.path.join(processed_dir, pdb_id, "lig.pdb")
    if not os.path.exists(lig_path):
        raise FileNotFoundError(lig_path)

    node_feature = pd.read_parquet(nf_path)
    if not antigen_chain:
        residues = _parse_pdb_residues(lig_path)
        chains = sorted({r.chain_id for r in residues if r.chain_id})
        antigen_chain = chains[0] if chains else ""

    lig_res = _build_chain_residue_order(lig_path, antigen_chain)
    lig_keys: List[AsaFullKey] = [
        (r.chain_id, r.resseq, (r.icode or "").strip()) for r in lig_res
    ]

    complex_path = _find_complex_structure(processed_dir, pdb_id, pdb_code)
    st_bits = _structure_contact_mask_for_chain(
        complex_path, antigen_chain, lig_keys, contact_cutoff
    )
    epi_z, st_rows, _ = _project_masks_to_node_rows(
        node_feature, lig_keys, {}, set(), st_bits
    )
    del epi_z
    n_on = int(sum(st_rows))
    prior_status = "ok" if n_on > 0 else "structure_no_contact"
    out_df = pd.DataFrame(
        {
            "resId": node_feature["resId"].astype(int),
            "epitope_prior": st_rows,
            "prior_status": prior_status,
            "prior_source": "structure_contact",
            "struct_fallback": st_rows,
            "n_contact_residues": n_on,
        }
    )
    out_dir = Path(nodes_edges_dir) / pdb_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / output_name
    out_df.to_parquet(out_path, index=False)
    man = {
        "pdb_id": pdb_id,
        "antigen_chain": antigen_chain,
        "prior_status": prior_status,
        "prior_source": "structure_contact",
        "n_contact_residues": n_on,
        "out_path": str(out_path),
    }
    return out_path, man


def main() -> int:
    ap = argparse.ArgumentParser(description="Structure-only Cα contact epitope mask.")
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--nodes_edges_dir", required=True)
    ap.add_argument("--processed_dir", required=True)
    ap.add_argument("--pdb_id", action="append")
    ap.add_argument("--pdb_list")
    ap.add_argument("--contact_cutoff", type=float, default=5.0)
    ap.add_argument(
        "--output_parquet",
        default="node_structure_contact_epitope.parquet",
    )
    ap.add_argument("--manifest_out", default="structure_contact_prior_manifest.csv")
    args = ap.parse_args()

    pdb_ids = _collect_pdb_ids(args)
    if not pdb_ids:
        print("No PDB IDs.", file=sys.stderr)
        return 2

    meta_df = _read_metadata_rows(args.metadata)
    meta_map = _metadata_dict_from_df(meta_df)

    manifest_rows: List[dict] = []
    ok, fail = 0, 0
    for pdb_id in pdb_ids:
        if pdb_id not in meta_map:
            print(f"SKIP {pdb_id}: not in metadata", file=sys.stderr)
            fail += 1
            continue
        try:
            _, man = process_one(
                pdb_id=pdb_id,
                meta_row=meta_map[pdb_id],
                nodes_edges_dir=args.nodes_edges_dir,
                processed_dir=args.processed_dir,
                output_name=args.output_parquet,
                contact_cutoff=args.contact_cutoff,
            )
            manifest_rows.append(man)
            print(f"OK   {pdb_id}")
            ok += 1
        except Exception as e:
            print(f"FAIL {pdb_id}: {e}", file=sys.stderr)
            fail += 1

    pd.DataFrame(manifest_rows).to_csv(args.manifest_out, index=False)
    print(f"Wrote manifest {args.manifest_out}  ok={ok} fail={fail}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
