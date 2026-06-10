#!/usr/bin/env python3
"""
Tier B: per-assay IEDB epitope priors (tighter than union) + assignment manifest.

Keeps existing union mask in node_iedb_pdb_epitope_prior.parquet (build separately).
Writes node_iedb_pdb_epitope_prior_assay.parquet per complex using ONE matched assay row.

Matching (same PDB code + antigen chain group):
  - Project each assay's epitope onto each complex's antigen chain.
  - Score = overlap(epitope residues, antibody-contact residues on antigen).
  - Assign assays to complexes via max-weight matching when counts are small.

Examples:
  python scripts/assign_iedb_assay_epitope_priors.py \\
    --metadata reports/metadata_ok_augmented_epitope.csv \\
    --iedb_csv reports/iedb_bcell_pdb_scoped_cohort191.csv \\
    --pdb_list reports/pdb_list_epitope_augmented_union.csv \\
    --nodes_edges_dir .../nodes_edges --processed_dir .../processed_data \\
    --sifts_cache_dir .../cache/pdbe --infer_uniprot --pdb_scope_skip_uniprot_match \\
    --manifest_out reports/iedb_assay_match_manifest.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from itertools import permutations
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import requests

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.build_iedb_epitope_prior import (  # noqa: E402
    AsaFullKey,
    _build_chain_residue_order,
    _build_pdb_to_uniprot_map,
    _collect_epitope_pdb_keys_for_antigen_chain,
    _collect_epitope_positions_from_iedb_row,
    _collect_pdb_ids,
    _detect_iedb_uniprot_column,
    _extract_accession,
    _fetch_pdbe_uniprot_json,
    _filter_iedb_for_pdb,
    _filter_iedb_for_uniprot,
    _find_complex_structure,
    _infer_uniprot_for_chain,
    _metadata_dict_from_df,
    _project_masks_to_node_rows,
    _read_metadata_rows,
    _row_has_chain_epitope_segments,
    _split_chain_field,
    _structure_contact_mask_for_chain,
)

ASSAY_PARQUET = "node_iedb_pdb_epitope_prior_assay.parquet"


def _iedb_subframe(
    iedb_df: pd.DataFrame,
    pdb_code: str,
    uniprot: str,
    ucol: str,
    pdb_scope: bool,
    skip_uniprot: bool,
) -> pd.DataFrame:
    if pdb_scope:
        sub = _filter_iedb_for_pdb(iedb_df, "complex__pdb_id", pdb_code)
        if uniprot and not skip_uniprot:
            sub = _filter_iedb_for_uniprot(sub, ucol, uniprot)
        return sub
    if uniprot:
        return _filter_iedb_for_uniprot(iedb_df, ucol, uniprot)
    return iedb_df.iloc[0:0]


def _epitope_sets_from_row(row: pd.Series, antigen_chain: str) -> Tuple[Set[int], Set[AsaFullKey]]:
    chain_keys = _collect_epitope_pdb_keys_for_antigen_chain(row, antigen_chain)
    if chain_keys:
        return set(), chain_keys
    if not _row_has_chain_epitope_segments(row):
        return _collect_epitope_positions_from_iedb_row(row), set()
    return set(), set()


def _assay_label(row: pd.Series) -> Tuple[str, str]:
    aid = str(row.get("assay_id", "") or "").strip()
    ab = str(row.get("assay_antibody__antibody_name", "") or "").strip()
    if not ab:
        ab = str(row.get("assay_antibody__antibody_source_material", "") or "").strip()
    return aid, ab


def _prepare_context(
    pdb_id: str,
    meta_row: dict,
    iedb_df: pd.DataFrame,
    ucol: str,
    nodes_edges_dir: str,
    processed_dir: str,
    sifts_cache: Optional[Path],
    session: requests.Session,
    infer_uniprot: bool,
    pdb_scope: bool,
    skip_uniprot: bool,
    contact_cutoff: float,
) -> Optional[dict]:
    antigen_chain = _split_chain_field(meta_row.get("antigen", ""))
    pdb_code = (meta_row.get("pdb") or "").lower().strip() or pdb_id.split("_")[0].lower()
    uniprot = (meta_row.get("antigen_uniprot") or "").strip()

    nf_path = os.path.join(nodes_edges_dir, pdb_id, "node_feature.parquet")
    lig_path = os.path.join(processed_dir, pdb_id, "lig.pdb")
    if not os.path.isfile(nf_path) or not os.path.isfile(lig_path):
        return None

    node_feature = pd.read_parquet(nf_path)
    pdbe_data: dict = {}
    pdbe_err = ""
    try:
        pdbe_data = _fetch_pdbe_uniprot_json(pdb_code, sifts_cache, session)
    except RuntimeError as e:
        pdbe_err = str(e)

    if infer_uniprot and not uniprot and not pdbe_err:
        acc, _ = _infer_uniprot_for_chain(pdbe_data, pdb_code, antigen_chain)
        uniprot = acc

    sub = _iedb_subframe(iedb_df, pdb_code, uniprot, ucol, pdb_scope, skip_uniprot)
    pdb_to_uni: Dict[AsaFullKey, int] = {}
    if uniprot and not pdbe_err:
        pdb_to_uni, _ = _build_pdb_to_uniprot_map(
            pdbe_data,
            pdb_code,
            antigen_chain,
            _extract_accession(uniprot),
            lig_path=lig_path,
        )

    lig_res = _build_chain_residue_order(lig_path, antigen_chain)
    lig_keys = [(r.chain_id, r.resseq, (r.icode or "").strip()) for r in lig_res]
    complex_path = _find_complex_structure(processed_dir, pdb_id, pdb_code)
    sfb = _structure_contact_mask_for_chain(
        complex_path, antigen_chain, lig_keys, contact_cutoff
    )

    return {
        "pdb_id": pdb_id,
        "pdb_code": pdb_code,
        "antigen_chain": antigen_chain,
        "uniprot": uniprot,
        "pdbe_err": pdbe_err,
        "sub": sub,
        "node_feature": node_feature,
        "lig_keys": lig_keys,
        "pdb_to_uni": pdb_to_uni,
        "sfb": sfb,
        "nodes_edges_dir": nodes_edges_dir,
    }


def _project_row(ctx: dict, row: pd.Series) -> Tuple[List[int], List[int], int]:
    """Return (final_bits, raw_bits, n_epi_positions)."""
    eu, epk = _epitope_sets_from_row(row, ctx["antigen_chain"])
    raw, _, n_epi = _project_masks_to_node_rows(
        ctx["node_feature"],
        ctx["lig_keys"],
        ctx["pdb_to_uni"],
        eu,
        ctx["sfb"],
        epk,
    )
    # Chain-less names: gate final mask to antibody-contact residues on THIS complex.
    if not _row_has_chain_epitope_segments(row):
        final = [1 if (b and c) else 0 for b, c in zip(raw, ctx["sfb"])]
    else:
        final = list(raw)
    return final, list(raw), n_epi


def _overlap_count(epi_bits: List[int], contact_bits: List[int]) -> int:
    return sum(1 for a, b in zip(epi_bits, contact_bits) if a and b)


def _overlap_score(epi_bits: List[int], contact_bits: List[int]) -> float:
    ov = _overlap_count(epi_bits, contact_bits)
    if ov == 0:
        return 0.0
    epi = sum(epi_bits)
    contact = sum(contact_bits)
    denom = epi + contact - ov
    jacc = ov / denom if denom > 0 else 0.0
    return float(ov) + 0.25 * jacc + 0.01 * epi


def _solve_assignment(
    complex_ids: List[str],
    assay_keys: List[str],
    scores: Dict[Tuple[str, str], float],
    min_contact_overlap: int,
) -> Dict[str, str]:
    if not complex_ids or not assay_keys:
        return {}

    def ok(c: str, a: str) -> bool:
        return scores.get((c, a), 0.0) >= float(min_contact_overlap)

    if len(assay_keys) == 1:
        a = assay_keys[0]
        best_c = max(complex_ids, key=lambda c: scores.get((c, a), 0.0))
        if ok(best_c, a):
            return {best_c: a}
        return {}

    if len(complex_ids) == len(assay_keys) and len(complex_ids) <= 8:
        best_map: Optional[Dict[str, str]] = None
        best_total = -1.0
        for perm in permutations(assay_keys):
            if not all(ok(c, a) for c, a in zip(complex_ids, perm)):
                continue
            total = sum(scores.get((c, a), 0.0) for c, a in zip(complex_ids, perm))
            if total > best_total:
                best_total = total
                best_map = dict(zip(complex_ids, perm))
        if best_map:
            return best_map

    assign: Dict[str, str] = {}
    used: Set[str] = set()
    for c in sorted(complex_ids):
        ranked = sorted(
            ((a, scores.get((c, a), 0.0)) for a in assay_keys if a not in used),
            key=lambda x: -x[1],
        )
        if ranked and ok(c, ranked[0][0]):
            assign[c] = ranked[0][0]
            used.add(ranked[0][0])
    return assign


def main() -> int:
    ap = argparse.ArgumentParser(description="Assign one IEDB assay per complex (tier B).")
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--iedb_csv", required=True)
    ap.add_argument("--nodes_edges_dir", required=True)
    ap.add_argument("--processed_dir", required=True)
    ap.add_argument("--pdb_list")
    ap.add_argument("--pdb_id", action="append")
    ap.add_argument("--sifts_cache_dir", default=None)
    ap.add_argument("--infer_uniprot", action="store_true")
    ap.add_argument("--no_pdb_scope", action="store_true", help="Disable PDB-scoped row filter")
    ap.add_argument("--pdb_scope_skip_uniprot_match", action="store_true")
    ap.add_argument("--contact_cutoff", type=float, default=5.0)
    ap.add_argument(
        "--min_contact_overlap",
        type=int,
        default=1,
        help="Min epitope residues (pre-gate) overlapping antibody contact to assign assay",
    )
    ap.add_argument("--manifest_out", required=True)
    args = ap.parse_args()

    pdb_ids = _collect_pdb_ids(args)
    meta_map = _metadata_dict_from_df(_read_metadata_rows(args.metadata))
    iedb_df = pd.read_csv(args.iedb_csv, low_memory=False)
    ucol = _detect_iedb_uniprot_column(iedb_df, None)

    cache = Path(args.sifts_cache_dir) if args.sifts_cache_dir else None
    session = requests.Session()
    session.headers.update({"User-Agent": "Epi4Ab-IEDB-assay-match/1.0"})

    contexts: Dict[str, dict] = {}
    for pdb_id in pdb_ids:
        if pdb_id not in meta_map:
            continue
        ctx = _prepare_context(
            pdb_id,
            meta_map[pdb_id],
            iedb_df,
            ucol,
            args.nodes_edges_dir,
            args.processed_dir,
            cache,
            session,
            args.infer_uniprot,
            not args.no_pdb_scope,
            args.pdb_scope_skip_uniprot_match,
            args.contact_cutoff,
        )
        if ctx is not None:
            contexts[pdb_id] = ctx

    # Score every (complex, assay row) — projection is per complex (chain / graph differ).
    row_bits: Dict[Tuple[str, str], List[int]] = {}
    row_bits_raw: Dict[Tuple[str, str], List[int]] = {}
    row_meta: Dict[str, Tuple[str, str, int]] = {}  # assay_key -> (assay_id, ab_name, n_epi)
    scores: Dict[Tuple[str, str], float] = {}

    for pdb_id, ctx in contexts.items():
        sub = ctx["sub"]
        if sub.empty:
            continue
        for idx, row in sub.iterrows():
            aid, ab = _assay_label(row)
            akey = f"{aid}:{idx}"
            final, raw, n_epi = _project_row(ctx, row)
            row_bits[(pdb_id, akey)] = final
            row_bits_raw[(pdb_id, akey)] = raw
            row_meta[akey] = (aid, ab, n_epi)
            # Rank by overlap of raw epitope with contact; threshold uses same count.
            scores[(pdb_id, akey)] = float(_overlap_count(raw, ctx["sfb"]))

    # Assign within (pdb_code, antigen_chain) groups
    groups: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for pdb_id, ctx in contexts.items():
        groups[(ctx["pdb_code"], ctx["antigen_chain"])].append(pdb_id)

    assignment: Dict[str, str] = {}
    for (_pdb_code, _ag), members in groups.items():
        assay_keys: List[str] = []
        for pid in members:
            for idx, row in contexts[pid]["sub"].iterrows():
                aid, _ = _assay_label(row)
                akey = f"{aid}:{idx}"
                if akey not in assay_keys:
                    assay_keys.append(akey)
        if not assay_keys:
            continue
        assign = _solve_assignment(
            members, assay_keys, scores, args.min_contact_overlap
        )
        assignment.update(assign)

    manifest_rows: List[dict] = []
    for pdb_id in pdb_ids:
        ctx = contexts.get(pdb_id)
        if ctx is None:
            manifest_rows.append(
                {
                    "pdb_id": pdb_id,
                    "prior_status_assay": "missing_graph",
                    "matched_assay_id": "",
                    "matched_antibody_name": "",
                    "n_assay_mapped": 0,
                }
            )
            continue

        akey = assignment.get(pdb_id)
        sub = ctx["sub"]
        union_rows = len(sub)

        if not akey:
            bits = [0] * len(ctx["node_feature"])
            aid, ab, n_epi = "", "", 0
            if union_rows == 0:
                st = "iedb_pdb_no_rows"
            else:
                st = "assay_unassigned"
        else:
            bits = row_bits.get((pdb_id, akey), [0] * len(ctx["node_feature"]))
            aid, ab, n_epi = row_meta.get(akey, ("", "", 0))
            st = "ok" if sum(bits) > 0 else "assay_unmapped"

        out_df = pd.DataFrame(
            {
                "resId": ctx["node_feature"]["resId"].astype(int),
                "epitope_prior": bits,
                "prior_status": st,
                "prior_source": "iedb_pdb_assay_matched",
                "struct_fallback": [0] * len(bits),
                "n_iedb_positions": n_epi,
                "n_iedb_mapped": int(sum(bits)),
                "frac_iedb_mapped": float(sum(bits)) / max(n_epi, 1) if n_epi else 0.0,
            }
        )
        out_path = Path(args.nodes_edges_dir) / pdb_id / ASSAY_PARQUET
        out_df.to_parquet(out_path, index=False)

        manifest_rows.append(
            {
                "pdb_id": pdb_id,
                "pdb_code": ctx["pdb_code"],
                "antigen_chain": ctx["antigen_chain"],
                "iedb_row_count_union_scope": union_rows,
                "matched_assay_id": aid,
                "matched_antibody_name": ab,
                "n_contact_overlap": int(scores.get((pdb_id, akey), 0.0)) if akey else 0,
                "min_contact_overlap": args.min_contact_overlap,
                "prior_status_assay": st,
                "n_assay_mapped": int(sum(bits)),
                "out_path_assay": str(out_path),
            }
        )
        print(f"OK   {pdb_id} assay={ab or aid or '-'} mapped={sum(bits)} status={st}")

    man_path = Path(args.manifest_out)
    man_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(manifest_rows).to_csv(man_path, index=False)
    n_ok = sum(1 for r in manifest_rows if r["prior_status_assay"] == "ok")
    print(f"Wrote {man_path} ok={n_ok}/{len(manifest_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
