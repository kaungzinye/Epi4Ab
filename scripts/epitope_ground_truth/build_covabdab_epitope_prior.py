#!/usr/bin/env python3
"""
Build CoV-AbDab-derived epitope priors aligned to node_feature.resId (SIFTS), same 191
(or arbitrary) PDB list as IEDB — non-matching complexes get covabdab_no_rows, not dropped.

CoV-AbDab bulk CSV (OPIG) often encodes epitopes as protein/domain text (e.g. \"S; RBD\")
without explicit residue lists. This script:
  - Prefers rows whose Structures field references the cohort PDB code (RCSB URLs).
  - Otherwise matches rows whose inferred antigen UniProt equals the complex metadata
    UniProt (SARS-CoV-2 spike → P0DTC2, nucleocapsid → P0DTC9, etc.).
  - Expands domain tokens to approximate 1-based UniProt ranges on P0DTC2 / P0DTC9.

Pass --cov_positions_col to read explicit residue lists from a column (comma/range syntax).

Examples:
  python scripts/epitope_ground_truth/build_covabdab_epitope_prior.py \\
    --covabdab_csv CoV-AbDab_080224.csv --metadata metadata.csv --pdb_list pdb_list.csv \\
    --nodes_edges_dir .../nodes_edges --processed_dir .../processed_data \\
    --sifts_cache_dir .../pdbe_cache --infer_uniprot --manifest_out reports/covabdab_prior_manifest.csv
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import requests

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.build_iedb_epitope_prior import (  # noqa: E402
    AsaFullKey,
    _build_pdb_to_uniprot_map,
    _extract_accession,
    _fetch_pdbe_uniprot_json,
    _infer_uniprot_for_chain,
    _metadata_dict_from_df,
    _parse_int_set_from_cell,
    _project_masks_to_node_rows,
    _read_metadata_rows,
)
from scripts.generate_dasa_labels import (  # noqa: E402
    _build_chain_residue_order,
    _parse_pdb_residues,
    _split_chain_field,
)

# Approximate SARS-CoV-2 spike (UniProt P0DTC2) segments, 1-based inclusive.
P0DTC2_DOMAIN_SPANS: Dict[str, Tuple[int, int]] = {
    "NTD": (14, 306),
    "RBD": (319, 541),
    "S1": (14, 685),
    "S2": (686, 1273),
    "FP": (788, 806),
    "HR1": (912, 984),
    "HR2": (1163, 1213),
    "TM": (1214, 1273),
}

# When only \"S\" / spike is named without a subdomain, use NTD+RBD-rich span.
P0DTC2_DEFAULT_EPITOPE_SPAN = (14, 541)
P0DTC9_FULL = (1, 419)
P0DTC3_FULL = (1, 222)
P0DTC4_FULL = (1, 75)

PDB_IN_URL = re.compile(r"/structure/([0-9][A-Za-z0-9]{3})\b", re.I)
PDB_TOKEN = re.compile(r"\b([0-9][A-Za-z0-9]{3})\b")

SARS2_TOKENS = (
    "sars-cov-2",
    "sars-cov2",
    "sars cov 2",
    "covid-19",
    "covid19",
    "2019-ncov",
    "hcov-19",
    "sarscov2",
)


def _strip_bom_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).lstrip("\ufeff") for c in df.columns]
    return df


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


def _binds_sars_cov2(binds: str) -> bool:
    b = binds.lower().replace("_", " ")
    return any(t in b for t in SARS2_TOKENS)


def _row_inferred_antigen_uniprot(row: pd.Series) -> Optional[str]:
    """Map CoV-AbDab row to a canonical SARS-CoV-2 UniProt when possible."""
    binds = str(row.get("Binds to", "") or "")
    if not _binds_sars_cov2(binds):
        return None
    pe = str(row.get("Protein + Epitope", "") or "").strip().upper()
    if not pe:
        return None
    parts = [p.strip() for p in re.split(r"[|;]", pe) if p.strip()]
    if not parts:
        return None
    head = parts[0].split()[0]
    if head in ("S", "SPIKE"):
        if any(x in pe for x in ("RBD", "NTD", "S1", "S2", "FP", "HR1", "HR2", "TM")):
            return "P0DTC2"
        return "P0DTC2"
    if head in ("N", "NCAPS", "NUCLEOCAPSID"):
        return "P0DTC9"
    if head == "M":
        return "P0DTC3"
    if head == "E":
        return "P0DTC4"
    if any(x in pe for x in ("RBD", "NTD", "S1", "S2", "SPIKE")):
        return "P0DTC2"
    return None


def _extract_pdbs_from_structures_cell(val) -> Set[str]:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return set()
    s = str(val).strip()
    if not s:
        return set()
    found = {m.group(1).lower() for m in PDB_IN_URL.finditer(s)}
    for m in PDB_TOKEN.finditer(s):
        t = m.group(1).lower()
        if t.isalnum() and len(t) == 4:
            found.add(t)
    return found


def _positions_from_domain_text(pe: str, uniprot_acc: str) -> Set[int]:
    u = uniprot_acc.upper().strip()
    pe_u = pe.upper()
    out: Set[int] = set()
    if u == "P0DTC2":
        for key, (lo, hi) in P0DTC2_DOMAIN_SPANS.items():
            if key in pe_u:
                out.update(range(lo, hi + 1))
        if not out and (
            re.search(r"\bS\b", pe_u) or "SPIKE" in pe_u or "RBD" in pe_u or "NTD" in pe_u
        ):
            lo, hi = P0DTC2_DEFAULT_EPITOPE_SPAN
            out.update(range(lo, hi + 1))
    elif u == "P0DTC9":
        lo, hi = P0DTC9_FULL
        out.update(range(lo, hi + 1))
    elif u == "P0DTC3":
        lo, hi = P0DTC3_FULL
        out.update(range(lo, hi + 1))
    elif u == "P0DTC4":
        lo, hi = P0DTC4_FULL
        out.update(range(lo, hi + 1))
    return out


def _collect_covabdab_positions_for_row(
    row: pd.Series,
    cohort_uniprot: str,
    positions_col: Optional[str],
) -> Set[int]:
    pos: Set[int] = set()
    acc = _extract_accession(cohort_uniprot)
    if not acc:
        acc = _row_inferred_antigen_uniprot(row) or ""
    if positions_col and positions_col in row.index and pd.notna(row[positions_col]):
        pos |= _parse_int_set_from_cell(row[positions_col])
    if pos:
        return pos
    pe = str(row.get("Protein + Epitope", "") or "")
    pos |= _positions_from_domain_text(pe, acc)
    return pos


def _matching_covabdab_rows(
    cov_df: pd.DataFrame,
    pdb_code: str,
    cohort_uniprot: str,
) -> pd.DataFrame:
    """Rows that reference this PDB and/or the cohort antigen UniProt."""
    pdb_code = pdb_code.lower()[:4]
    acc = _extract_accession(cohort_uniprot)
    pdb_rows: List[int] = []
    uni_rows: List[int] = []
    for i, row in cov_df.iterrows():
        pdbs = _extract_pdbs_from_structures_cell(row.get("Structures"))
        if pdb_code in pdbs:
            pdb_rows.append(i)
            continue
        if acc:
            inf = _row_inferred_antigen_uniprot(row)
            if inf and inf == acc:
                uni_rows.append(i)
    idx_order = pdb_rows + [j for j in uni_rows if j not in pdb_rows]
    if not idx_order:
        return cov_df.iloc[0:0]
    return cov_df.loc[idx_order]


def _detect_positions_col(df: pd.DataFrame, override: Optional[str]) -> Optional[str]:
    if override:
        return override if override in df.columns else None
    for c in df.columns:
        cl = c.lower()
        if "residue" in cl and ("epitope" in cl or "contact" in cl or "defin" in cl):
            return c
    return None


def process_one(
    pdb_id: str,
    meta_row: Dict[str, str],
    cov_df: pd.DataFrame,
    nodes_edges_dir: str,
    processed_dir: str,
    sifts_cache: Optional[Path],
    session: requests.Session,
    output_name: str,
    infer_uniprot: bool,
    positions_col: Optional[str],
) -> Tuple[Path, dict]:
    antigen_chain = _split_chain_field(meta_row.get("antigen", ""))
    pdb_code = (meta_row.get("pdb") or "").lower().strip() or pdb_id.split("_")[0].lower()
    uniprot = (meta_row.get("antigen_uniprot") or "").strip()

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

    pdbe_data: dict = {}
    pdbe_err = ""
    try:
        pdbe_data = _fetch_pdbe_uniprot_json(pdb_code, sifts_cache, session)
    except RuntimeError as e:
        pdbe_err = str(e)

    if infer_uniprot and not uniprot and not pdbe_err:
        acc, _ = _infer_uniprot_for_chain(pdbe_data, pdb_code, antigen_chain)
        uniprot = acc

    sub = _matching_covabdab_rows(cov_df, pdb_code, uniprot)
    cov_row_count = len(sub)

    epitope_uni: Set[int] = set()
    for _, row in sub.iterrows():
        epitope_uni |= _collect_covabdab_positions_for_row(row, uniprot, positions_col)

    pdb_to_uni: Dict[AsaFullKey, int] = {}
    n_pdbe_seg = 0
    if uniprot and not pdbe_err:
        pdb_to_uni, n_pdbe_seg = _build_pdb_to_uniprot_map(
            pdbe_data,
            pdb_code,
            antigen_chain,
            _extract_accession(uniprot),
            lig_path=lig_path,
        )

    manifest_extra: dict = {
        "pdb_id": pdb_id,
        "antigen_uniprot": uniprot,
        "pdbe_error": pdbe_err,
        "n_pdbe_segments": n_pdbe_seg,
        "covabdab_row_count": cov_row_count,
    }

    lig_res = _build_chain_residue_order(lig_path, antigen_chain)
    lig_keys = [(r.chain_id, r.resseq, (r.icode or "").strip()) for r in lig_res]

    struct_fb = [0] * len(lig_keys)
    epi_rows, st_rows, n_graph_cov = _project_masks_to_node_rows(
        node_feature, lig_keys, pdb_to_uni, epitope_uni, struct_fb
    )

    if not uniprot:
        prior_status = "no_uniprot"
    elif pdbe_err:
        prior_status = "sifts_fetch_failed"
    elif cov_row_count == 0:
        prior_status = "covabdab_no_rows"
    elif not epitope_uni:
        prior_status = "covabdab_no_positions"
    elif not pdb_to_uni:
        prior_status = "sifts_empty_map"
    elif n_graph_cov == 0:
        prior_status = "covabdab_unmapped"
    else:
        prior_status = "ok"

    prior_source = "covabdab_sifts" if n_graph_cov > 0 else "none"

    n_cov_pos = len(epitope_uni)
    frac_m = float(n_graph_cov) / max(n_cov_pos, 1) if n_cov_pos else 0.0
    out_df = pd.DataFrame(
        {
            "resId": node_feature["resId"].astype(int),
            "epitope_prior": list(epi_rows),
            "prior_status": prior_status,
            "prior_source": prior_source,
            "struct_fallback": list(st_rows),
            "n_covabdab_positions": n_cov_pos,
            "n_covabdab_mapped": n_graph_cov,
            "frac_covabdab_mapped": frac_m,
        }
    )
    out_dir = Path(nodes_edges_dir) / pdb_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / output_name
    out_df.to_parquet(out_path, index=False)

    man = {
        **manifest_extra,
        "covabdab_prior_status": prior_status,
        "covabdab_prior_source": prior_source,
        "covabdab_n_positions": n_cov_pos,
        "covabdab_n_mapped": n_graph_cov,
        "frac_covabdab_mapped": frac_m,
        "out_path": str(out_path),
    }
    return out_path, man


def main() -> int:
    ap = argparse.ArgumentParser(description="Build CoV-AbDab + SIFTS epitope priors aligned to graphs.")
    ap.add_argument("--covabdab_csv", required=True, help="CoV-AbDab bulk CSV from OPIG")
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--nodes_edges_dir", required=True)
    ap.add_argument("--processed_dir", required=True)
    ap.add_argument("--pdb_id", action="append")
    ap.add_argument("--pdb_list")
    ap.add_argument("--cov_positions_col", default=None, help="Optional explicit residue list column")
    ap.add_argument("--sifts_cache_dir", default=None)
    ap.add_argument(
        "--output_parquet",
        default="node_covabdab_epitope_prior.parquet",
        help="Written under each nodes_edges/<pdb_id>/",
    )
    ap.add_argument("--manifest_out", default="covabdab_prior_manifest.csv")
    ap.add_argument("--infer_uniprot", action="store_true")
    args = ap.parse_args()

    pdb_ids = _collect_pdb_ids(args)
    if not pdb_ids:
        print("No PDB IDs.", file=sys.stderr)
        return 2

    meta_df = _read_metadata_rows(args.metadata)
    meta_map = _metadata_dict_from_df(meta_df)

    cov_df = pd.read_csv(args.covabdab_csv, low_memory=False, encoding="utf-8-sig")
    cov_df = _strip_bom_columns(cov_df)
    pos_col = _detect_positions_col(cov_df, args.cov_positions_col)

    cache = Path(args.sifts_cache_dir) if args.sifts_cache_dir else None
    session = requests.Session()
    session.headers.update({"User-Agent": "Epi4Ab-CoV-AbDab-prior/1.0"})

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
                cov_df=cov_df,
                nodes_edges_dir=args.nodes_edges_dir,
                processed_dir=args.processed_dir,
                sifts_cache=cache,
                session=session,
                output_name=args.output_parquet,
                infer_uniprot=args.infer_uniprot,
                positions_col=pos_col,
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
