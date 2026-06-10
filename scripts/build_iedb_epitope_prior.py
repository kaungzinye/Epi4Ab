#!/usr/bin/env python3
"""
Build per-residue binary IEDB-derived epitope priors aligned to node_feature.parquet.

Uses IEDB bulk CSV (B-cell / epitope positions) keyed by antigen UniProt, then PDBe
REST API SIFTS-style mappings (mappings/uniprot/{pdb}) to map UniProt positions to
PDB author numbering on the antigen chain. Independent of DASA labels.

With ``--pdb_scope``, only rows whose ``complex__pdb_id`` (IQ-API ``bcell_export``) match
the structure's 4-letter code are used (IEDB-3D–aligned, sparse masks). Successful maps
set ``prior_source=iedb_pdb_sifts``. Harvest with
``scripts/epitope_ground_truth/fetch_iedb_bcell_pdb_scoped.py``.

Outputs per PDB under nodes_edges/<pdb_id>/:
  <output_parquet> (default node_iedb_epitope_prior.parquet) with columns:
    resId, epitope_prior, prior_status, prior_source, struct_fallback,
    n_iedb_positions, n_iedb_mapped, frac_iedb_mapped

Also writes a cohort manifest CSV (--manifest_out).

Caveats (see plan): uneven IEDB coverage; incomplete discontinuous epitopes;
linear PDBe segment mapping can miss insertions; use prior_status for QA.
When PDBe returns null ``author_residue_number`` but ``residue_number`` span
matches the UniProt span, we map through ``lig.pdb`` chain order (same as DASA).

For IQ-API harvests (linear + discontinuous B-cell positives), use
`scripts/epitope_ground_truth/fetch_iedb_epitopes_cohort.py` and pass the
resulting CSV as --iedb_csv (column epitope__molecule_parent_iri is auto-detected).

Examples:
  python scripts/build_iedb_epitope_prior.py \\
    --metadata metadata.csv --iedb_csv iedb_export.csv \\
    --nodes_edges_dir .../nodes_edges --processed_dir .../processed_data \\
    --sifts_cache_dir .../cache/pdbe --manifest_out .../iedb_prior_manifest.csv
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import requests

# Reuse DASA parsing / chain order invariants
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.generate_dasa_labels import (  # noqa: E402
    _build_chain_residue_order,
    _find_complex_structure,
    _parse_pdb_residues,
    _split_chain_field,
)

AsaFullKey = Tuple[str, int, str]

UNIPROT_ACC_RE = re.compile(
    r"\b([OPQ][0-9][A-Z0-9]{3}|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})\b"
)


def _norm_icode(icode) -> str:
    if icode is None:
        return ""
    s = str(icode).strip()
    if s in (".", "None", "?"):
        return ""
    return s


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


def _read_metadata_rows(metadata_path: str) -> pd.DataFrame:
    df = pd.read_csv(metadata_path, comment="#")
    if "pdbID" not in df.columns:
        raise ValueError(f"metadata must have pdbID column: {metadata_path}")
    return df


def _metadata_dict_from_df(df: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    up_candidates = (
        "antigen_uniprot",
        "antigenUniProt",
        "UniProt",
        "uniprot",
        "Antigen UniProt ID",
    )
    for _, row in df.iterrows():
        pid = str(row["pdbID"])
        uni = ""
        for c in up_candidates:
            if c in df.columns and pd.notna(row.get(c)):
                uni = str(row[c]).strip()
                if uni:
                    break
        out[pid] = {
            "pdb": str(row["pdb"]).strip().lower() if "pdb" in df.columns else "",
            "antigen": _split_chain_field(str(row["antigen"])) if "antigen" in df.columns else "",
            "antigen_uniprot": uni,
        }
    return out


def _detect_iedb_uniprot_column(df: pd.DataFrame, override: Optional[str]) -> str:
    if override and override in df.columns:
        return override
    for c in df.columns:
        cl = c.lower().replace(" ", "_")
        if "uniprot" in cl and "antigen" in cl:
            return c
    for c in df.columns:
        cl = c.lower()
        if cl in ("antigen_uniprot_id", "uniprot_id", "antigen_accession"):
            return c
    # IQ-API / bcell_export exports use epitope__molecule_parent_iri (contains UniProt URL)
    for c in df.columns:
        cl = c.lower()
        if "molecule_parent_iri" in cl and "epitope" in cl:
            return c
    for c in df.columns:
        if c.lower() == "antigen_accession" or "accession" in c.lower():
            return c
    raise ValueError(
        "Could not detect IEDB UniProt column; pass --iedb_uniprot_col explicitly. "
        f"Columns: {list(df.columns)[:40]}"
    )


def _extract_accession(text: str) -> str:
    if not text or not str(text).strip() or str(text).lower() in ("nan", "none"):
        return ""
    s_raw = str(text).strip()
    # IQ-API / OBO IRIs: http://www.uniprot.org/uniprot/P0CL66 — tail may not match
    # UNIPROT_ACC_RE alone (e.g. P0CL66 is 6 chars; OPQ branch expects 5 body chars).
    if "uniprot.org" in s_raw.lower():
        tail = s_raw.rstrip("/").split("/")[-1]
        tail = tail.split("?")[0].strip().upper()
        if tail:
            m2 = UNIPROT_ACC_RE.fullmatch(tail)
            if m2:
                return m2.group(1)
            # Relaxed IRI tail: 6–10 chars, letter + digit + alnum (covers P0CL66, A0A…, etc.).
            # Do not use [A-NR-Z] here — that class wrongly excludes P.
            if 6 <= len(tail) <= 10 and re.fullmatch(r"[A-Z][0-9][A-Z0-9]{3,9}", tail):
                return tail
    s = s_raw.upper().strip()
    m = UNIPROT_ACC_RE.search(s)
    if m:
        return m.group(1)
    return s.split()[0].strip()


def _positions_from_discontinuous_epitope_name(name: str) -> Set[int]:
    """Parse residue indices from IEDB discontinuous epitope names like 'S60, L61, Y62'."""
    out: Set[int] = set()
    if not name or not str(name).strip():
        return out
    s = str(name).strip()
    for m in re.finditer(r"\b([A-Z])\s*(\d+)\b", s):
        try:
            out.add(int(m.group(2)))
        except ValueError:
            continue
    return out


def _parse_pdb_residue_numbers_from_text(text: str) -> Set[int]:
    """Residue numbers from IEDB text: 'E48, N49' or 'P34' (optional 1–3 letter amino-acid prefix)."""
    out: Set[int] = set()
    if not text or not str(text).strip():
        return out
    for m in re.finditer(r"(?:\b[A-Z]{1,3})?(\d+)\b", str(text)):
        try:
            out.add(int(m.group(1)))
        except ValueError:
            continue
    return out


def _epitope_name_chain_segments(name: str) -> Dict[str, Set[int]]:
    """Parse 'C: E48, N49; P: P34, D35' → {chain_id: {residue numbers on that chain}}."""
    out: Dict[str, Set[int]] = {}
    if not name or not str(name).strip():
        return out
    s = str(name).strip()
    if ":" not in s:
        return out
    for segment in re.split(r"[;]+", s):
        segment = segment.strip()
        if not segment or ":" not in segment:
            continue
        chain_part, rest = segment.split(":", 1)
        chain_id = chain_part.strip()
        if not chain_id or len(chain_id) > 4:
            continue
        nums = _parse_pdb_residue_numbers_from_text(rest)
        if nums:
            out.setdefault(chain_id, set()).update(nums)
    return out


def _row_has_chain_epitope_segments(row: pd.Series) -> bool:
    for key in ("epitope__name", "epitope_name", "Epitope Name"):
        if key in row.index and pd.notna(row[key]):
            if _epitope_name_chain_segments(str(row[key])):
                return True
    return False


def _collect_epitope_pdb_keys_for_antigen_chain(
    row: pd.Series, antigen_chain: str
) -> Set[AsaFullKey]:
    """When epitope__name lists PDB chain:residue, keep only the segment for this antigen chain."""
    if not antigen_chain:
        return set()
    for key in ("epitope__name", "epitope_name", "Epitope Name"):
        if key not in row.index or pd.isna(row[key]):
            continue
        segments = _epitope_name_chain_segments(str(row[key]))
        if not segments:
            continue
        nums = segments.get(antigen_chain.strip())
        if not nums:
            return set()
        return {(antigen_chain.strip(), int(n), "") for n in nums}
    return set()


def _parse_int_set_from_cell(val) -> Set[int]:
    out: Set[int] = set()
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return out
    if isinstance(val, (int, np.integer)):
        out.add(int(val))
        return out
    s = str(val).strip()
    if not s or s.lower() in ("nan", "none"):
        return out
    for part in re.split(r"[,;/\s]+", s):
        part = part.strip()
        if not part:
            continue
        m = re.match(r"^(\d+)\s*-\s*(\d+)$", part)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            lo, hi = (a, b) if a <= b else (b, a)
            out.update(range(lo, hi + 1))
            continue
        try:
            out.add(int(part))
        except ValueError:
            continue
    return out


def _collect_epitope_positions_from_iedb_row(row: pd.Series) -> Set[int]:
    pos: Set[int] = set()
    low_map = {c.lower().replace(" ", "_"): c for c in row.index}
    # Common IEDB / export column name patterns
    # Epitope-specific spans only. Do NOT use assay_antigen__* start/end: those are the
    # full antigen construct (e.g. Spike 333–528) and mark almost every residue as epitope.
    pairs = [
        ("starting_position", "ending_position"),
        ("start_position", "end_position"),
        ("epitope_starting_position", "epitope_ending_position"),
        ("epitope__starting_position", "epitope__ending_position"),
        ("起始位置", "结束位置"),
    ]
    for a, b in pairs:
        if a in low_map and b in low_map:
            try:
                rs = row[low_map[a]]
                re_ = row[low_map[b]]
                if pd.notna(rs) and pd.notna(re_):
                    start = int(float(rs))
                    end = int(float(re_))
                    lo, hi = (start, end) if start <= end else (end, start)
                    pos.update(range(lo, hi + 1))
            except (ValueError, TypeError):
                pass
    for c in row.index:
        cl = c.lower()
        if "position" in cl and cl not in (
            "starting_position",
            "ending_position",
            "start_position",
            "end_position",
        ):
            pos |= _parse_int_set_from_cell(row[c])
    if not pos:
        for key in ("epitope__name", "epitope_name", "Epitope Name"):
            if key in row.index and pd.notna(row[key]):
                pos |= _positions_from_discontinuous_epitope_name(str(row[key]))
    return pos


def _filter_iedb_for_uniprot(iedb_df: pd.DataFrame, uniprot_col: str, target_acc: str) -> pd.DataFrame:
    target = _extract_accession(target_acc)
    if not target:
        return iedb_df.iloc[0:0]
    acc_norm = iedb_df[uniprot_col].map(_extract_accession)
    return iedb_df[acc_norm == target]


def _normalize_pdb_code_cell(val) -> str:
    """Return 4-letter PDB code uppercase, or empty if missing / invalid."""
    if val is None or pd.isna(val):
        return ""
    s = str(val).strip().upper()
    if not s:
        return ""
    # IEDB IQ-API uses e.g. "7SOE"; strip extension if ever "7soe_A"
    tok = s.split("_")[0].split()[0]
    if len(tok) >= 4:
        return tok[:4]
    return ""


def _filter_iedb_for_pdb(
    iedb_df: pd.DataFrame, pdb_col: str, pdb_code_four: str
) -> pd.DataFrame:
    """Keep rows whose complex PDB id matches the 4-letter code (case-insensitive)."""
    want = (pdb_code_four or "").strip().upper()[:4]
    if not want or pdb_col not in iedb_df.columns:
        return iedb_df.iloc[0:0]
    norm = iedb_df[pdb_col].map(_normalize_pdb_code_cell)
    return iedb_df[norm == want]


def _fetch_pdbe_uniprot_json(pdb_code: str, cache_dir: Optional[Path], session: requests.Session) -> dict:
    pdb_code = pdb_code.lower()[:4]
    cache_path = cache_dir / f"{pdb_code}_uniprot.json" if cache_dir else None
    if cache_path and cache_path.is_file():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    url = f"https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb_code}"
    for attempt in range(3):
        try:
            r = session.get(url, timeout=60)
            r.raise_for_status()
            data = r.json()
            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps(data), encoding="utf-8")
            time.sleep(0.15)
            return data
        except (requests.RequestException, json.JSONDecodeError) as e:
            if attempt == 2:
                raise RuntimeError(f"PDBe uniprot mapping fetch failed for {pdb_code}: {e}") from e
            time.sleep(1.0 + attempt)
    return {}


def _infer_uniprot_for_chain(
    pdbe_data: dict, pdb_code: str, chain_id: str
) -> Tuple[str, str]:
    pdb_code = pdb_code.lower()[:4]
    ent = pdbe_data.get(pdb_code, {}).get("UniProt", {})
    for acc, block in ent.items():
        for m in block.get("mappings", []):
            if str(m.get("chain_id", "")).strip() == str(chain_id).strip():
                name = str(block.get("identifier", acc))
                return acc, name
    if ent:
        acc = next(iter(ent.keys()))
        block = ent[acc]
        name = str(block.get("identifier", acc))
        return acc, name
    return "", ""


def _pdbe_int_or_none(val) -> Optional[int]:
    if val is None:
        return None
    try:
        if isinstance(val, float) and np.isnan(val):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _build_pdb_to_uniprot_map(
    pdbe_data: dict,
    pdb_code: str,
    chain_id: str,
    uniprot_acc: str,
    lig_path: Optional[str] = None,
) -> Tuple[Dict[AsaFullKey, int], int]:
    """Author (PDB) residue key -> UniProt sequence index (1-based).

    PDBe often returns ``author_residue_number``; when it is null (common on some
    mmCIF-derived entries), we fall back to ``residue_number`` (contiguous segment
    index in the mapped polymer) and align to :func:`_build_chain_residue_order`
    on ``lig_path`` for the antigen chain.
    """
    pdb_code = pdb_code.lower()[:4]
    ch = str(chain_id).strip() or "A"
    ent = (pdbe_data.get(pdb_code, {}) or {}).get("UniProt", {})
    block = ent.get(uniprot_acc) or {}
    mappings = block.get("mappings", [])
    out: Dict[AsaFullKey, int] = {}
    n_seg = 0
    for m in mappings:
        if str(m.get("chain_id", "")).strip() != ch:
            continue
        n_seg += 1
        try:
            unp_start = int(m["unp_start"])
            unp_end = int(m.get("unp_end", unp_start))
        except (KeyError, ValueError, TypeError):
            continue
        if unp_end >= unp_start:
            span_unp = unp_end - unp_start + 1
            unp_step = 1
        else:
            span_unp = unp_start - unp_end + 1
            unp_step = -1

        st = m.get("start") or {}
        en = m.get("end") or {}
        rs = _pdbe_int_or_none(st.get("author_residue_number"))
        re = _pdbe_int_or_none(en.get("author_residue_number"))

        filled = False
        if rs is not None and re is not None:
            span_auth = abs(re - rs) + 1
            n_take = min(span_auth, span_unp)
            step = 1 if re >= rs else -1
            icode0 = _norm_icode(st.get("author_insertion_code", ""))
            for k in range(n_take):
                auth_num = rs + k * step
                unp = unp_start + k * unp_step
                icode_v = icode0 if k == 0 else ""
                key = (ch, int(auth_num), icode_v)
                out[key] = int(unp)
            filled = n_take > 0

        if not filled and lig_path and os.path.isfile(lig_path):
            lab_s = _pdbe_int_or_none(st.get("residue_number"))
            lab_e = _pdbe_int_or_none(en.get("residue_number"))
            if lab_s is None:
                continue
            if lab_e is None:
                lab_e = lab_s + span_unp - 1 if unp_step > 0 else lab_s - span_unp + 1
            span_lab = abs(lab_e - lab_s) + 1
            n_take = min(span_lab, span_unp)
            try:
                lig_ord = _build_chain_residue_order(lig_path, ch)
            except (OSError, ValueError):
                lig_ord = []
            if not lig_ord:
                continue
            idx0 = min(lab_s, lab_e) - 1
            for k in range(n_take):
                ii = idx0 + k
                if ii < 0 or ii >= len(lig_ord):
                    break
                res = lig_ord[ii]
                key = (ch, int(res.resseq), _norm_icode(res.icode))
                out[key] = int(unp_start + k * unp_step)
    return out, n_seg


def _ca_coord_map(pdb_path: str) -> Dict[Tuple[str, int, str], np.ndarray]:
    coords: Dict[Tuple[str, int, str], np.ndarray] = {}
    lower = pdb_path.lower()
    if lower.endswith(".cif"):
        try:
            from Bio.PDB import MMCIFParser
        except ImportError:
            return coords
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure("c", pdb_path)
        for model in structure:
            for chain in model:
                ch = (chain.id or "").strip() or "A"
                for residue in chain:
                    if residue.id[0] != " ":
                        continue
                    _, resseq, icode = residue.id
                    ic = _norm_icode(icode)
                    ca = None
                    for atom in residue:
                        if atom.name == "CA":
                            ca = atom.coord
                            break
                    if ca is not None:
                        coords[(ch, int(resseq), ic)] = np.asarray(ca, dtype=float)
            break
        return coords
    with open(pdb_path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            if len(line) < 54:
                continue
            aname = line[12:16].strip()
            if aname != "CA":
                continue
            chain = line[21].strip() or "A"
            try:
                resseq = int(line[22:26].strip())
            except ValueError:
                continue
            icode = _norm_icode(line[26].strip())
            try:
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            except ValueError:
                continue
            coords[(chain, resseq, icode)] = np.array([x, y, z], dtype=float)
    return coords


def _structure_contact_mask_for_chain(
    complex_path: str,
    antigen_chain: str,
    lig_keys_in_order: Sequence[AsaFullKey],
    cutoff_angstrom: float,
) -> List[int]:
    cmap = _ca_coord_map(complex_path)
    ag_coords = []
    ab_coords = []
    for key, xyz in cmap.items():
        ch, _, _ = key
        if ch == antigen_chain:
            ag_coords.append(xyz)
        else:
            ab_coords.append(xyz)
    if not ag_coords or not ab_coords:
        return [0] * len(lig_keys_in_order)
    ab_stack = np.stack(ab_coords, axis=0)
    mask_bits: List[int] = []
    for key in lig_keys_in_order:
        xyz = cmap.get(key)
        if xyz is None:
            mask_bits.append(0)
            continue
        d = np.linalg.norm(ab_stack - xyz, axis=1)
        mask_bits.append(1 if float(np.min(d)) < cutoff_angstrom else 0)
    return mask_bits


def _project_masks_to_node_rows(
    node_feature: pd.DataFrame,
    ligand_keys_in_occurrence_order: List[AsaFullKey],
    pdb_to_uni: Dict[AsaFullKey, int],
    epitope_uni_positions: Set[int],
    struct_fb: List[int],
    epitope_pdb_keys: Optional[Set[AsaFullKey]] = None,
) -> Tuple[List[int], List[int], int]:
    """Return (epitope_bits, struct_bits_per_node_row, n_graph_epitope_from_iedb)."""
    epitope_pdb_keys = epitope_pdb_keys or set()
    by_resseq: Dict[int, List[AsaFullKey]] = {}
    for chain, resseq, icode in ligand_keys_in_occurrence_order:
        by_resseq.setdefault(resseq, []).append((chain, resseq, icode))

    use_counter: Dict[int, int] = {}
    resseq_occurrence_to_i: Dict[Tuple[int, int], int] = {}
    seen: Dict[int, int] = {}
    for i, (chain, resseq, icode) in enumerate(ligand_keys_in_occurrence_order):
        occ = seen.get(resseq, 0)
        resseq_occurrence_to_i[(resseq, occ)] = i
        seen[resseq] = occ + 1

    epitope_out: List[int] = []
    struct_out: List[int] = []
    n_graph_iedb = 0
    for _, row in node_feature.iterrows():
        res_id = int(row["resId"])
        idx = use_counter.get(res_id, 0)
        candidates = by_resseq.get(res_id, [])
        if idx >= len(candidates):
            epitope_out.append(0)
            struct_out.append(0)
            use_counter[res_id] = idx + 1
            continue
        key = candidates[idx]
        use_counter[res_id] = idx + 1
        lig_i = resseq_occurrence_to_i.get((res_id, idx))
        if lig_i is None:
            epitope_out.append(0)
            struct_out.append(0)
            continue
        up = pdb_to_uni.get(key)
        bit = 0
        if key in epitope_pdb_keys:
            bit = 1
            n_graph_iedb += 1
        elif up is not None and up in epitope_uni_positions:
            bit = 1
            n_graph_iedb += 1
        epitope_out.append(bit)
        struct_out.append(int(struct_fb[lig_i]) if lig_i < len(struct_fb) else 0)

    return epitope_out, struct_out, n_graph_iedb


def process_one(
    pdb_id: str,
    meta_row: Dict[str, str],
    iedb_df: pd.DataFrame,
    iedb_uniprot_col: str,
    nodes_edges_dir: str,
    processed_dir: str,
    sifts_cache: Optional[Path],
    session: requests.Session,
    output_name: str,
    infer_uniprot: bool,
    struct_fallback: bool,
    contact_cutoff: float,
    pdb_scope: bool = False,
    iedb_pdb_col: str = "complex__pdb_id",
    pdb_scope_require_uniprot_match: bool = True,
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

    epitope_uni: Set[int] = set()
    epitope_pdb_keys: Set[AsaFullKey] = set()
    iedb_row_count = 0
    pdb_to_uni: Dict[AsaFullKey, int] = {}
    n_pdbe_seg = 0

    if pdb_scope:
        sub = _filter_iedb_for_pdb(iedb_df, iedb_pdb_col, pdb_code)
        if uniprot and pdb_scope_require_uniprot_match:
            sub = _filter_iedb_for_uniprot(sub, iedb_uniprot_col, uniprot)
        iedb_row_count = len(sub)
        for _, row in sub.iterrows():
            chain_keys = _collect_epitope_pdb_keys_for_antigen_chain(row, antigen_chain)
            if chain_keys:
                epitope_pdb_keys |= chain_keys
            elif not _row_has_chain_epitope_segments(row):
                epitope_uni |= _collect_epitope_positions_from_iedb_row(row)
    elif uniprot:
        sub = _filter_iedb_for_uniprot(iedb_df, iedb_uniprot_col, uniprot)
        iedb_row_count = len(sub)
        for _, row in sub.iterrows():
            chain_keys = _collect_epitope_pdb_keys_for_antigen_chain(row, antigen_chain)
            if chain_keys:
                epitope_pdb_keys |= chain_keys
            elif not _row_has_chain_epitope_segments(row):
                epitope_uni |= _collect_epitope_positions_from_iedb_row(row)

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
        "iedb_row_count": iedb_row_count,
    }

    lig_res = _build_chain_residue_order(lig_path, antigen_chain)
    lig_keys = [(r.chain_id, r.resseq, (r.icode or "").strip()) for r in lig_res]

    complex_path = _find_complex_structure(processed_dir, pdb_id, pdb_code)
    if struct_fallback:
        sfb = _structure_contact_mask_for_chain(
            complex_path, antigen_chain, lig_keys, contact_cutoff
        )
    else:
        sfb = [0] * len(lig_keys)

    epi_rows, st_rows, n_graph_iedb = _project_masks_to_node_rows(
        node_feature, lig_keys, pdb_to_uni, epitope_uni, sfb, epitope_pdb_keys
    )

    if not uniprot:
        prior_status = "no_uniprot"
    elif pdbe_err:
        prior_status = "sifts_fetch_failed"
    elif iedb_row_count == 0:
        prior_status = "iedb_pdb_no_rows" if pdb_scope else "iedb_no_rows"
    elif not epitope_uni and not epitope_pdb_keys:
        prior_status = "iedb_empty_positions"
    elif not pdb_to_uni and not epitope_pdb_keys:
        prior_status = "sifts_empty_map"
    elif n_graph_iedb == 0:
        prior_status = "iedb_unmapped"
    else:
        prior_status = "ok"

    if n_graph_iedb > 0:
        prior_source = "iedb_pdb_sifts" if pdb_scope else "iedb_sifts"
    else:
        prior_source = "none"
    if struct_fallback and sum(st_rows) > 0 and prior_source == "none":
        prior_source = "structure_contact"

    epitope_final = list(epi_rows)
    if struct_fallback and sum(epitope_final) == 0 and prior_status in (
        "iedb_empty_positions",
        "iedb_no_rows",
        "no_uniprot",
        "iedb_unmapped",
        "sifts_fetch_failed",
        "sifts_empty_map",
    ):
        epitope_final = [max(a, b) for a, b in zip(epi_rows, st_rows)]
        prior_status = f"{prior_status}_struct_fallback"
        if sum(epitope_final) > 0:
            prior_source = (
                prior_source + "+struct"
                if prior_source not in ("none", "structure_contact")
                else "structure_contact"
            )

    n_iedb_pos = len(epitope_uni)
    frac_m = float(n_graph_iedb) / max(n_iedb_pos, 1) if n_iedb_pos else 0.0
    out_df = pd.DataFrame(
        {
            "resId": node_feature["resId"].astype(int),
            "epitope_prior": epitope_final,
            "prior_status": prior_status,
            "prior_source": prior_source,
            "struct_fallback": st_rows,
            "n_iedb_positions": n_iedb_pos,
            "n_iedb_mapped": n_graph_iedb,
            "frac_iedb_mapped": frac_m,
        }
    )
    out_dir = Path(nodes_edges_dir) / pdb_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / output_name
    out_df.to_parquet(out_path, index=False)

    man = {
        **manifest_extra,
        "prior_status": prior_status,
        "prior_source": prior_source,
        "n_iedb_positions": n_iedb_pos,
        "n_iedb_mapped": n_graph_iedb,
        "frac_iedb_mapped": frac_m,
        "out_path": str(out_path),
    }
    return out_path, man


def main() -> int:
    ap = argparse.ArgumentParser(description="Build IEDB+SIFTS epitope priors aligned to graphs.")
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--iedb_csv", required=True, help="IEDB database_export CSV (gz or plain)")
    ap.add_argument("--nodes_edges_dir", required=True)
    ap.add_argument("--processed_dir", required=True)
    ap.add_argument("--pdb_id", action="append")
    ap.add_argument("--pdb_list")
    ap.add_argument("--iedb_uniprot_col", default=None)
    ap.add_argument("--sifts_cache_dir", default=None, help="Cache dir for PDBe JSON (recommend on HPC)")
    ap.add_argument(
        "--output_parquet",
        default="node_iedb_epitope_prior.parquet",
        help="Written under each nodes_edges/<pdb_id>/",
    )
    ap.add_argument(
        "--manifest_out",
        default="iedb_prior_manifest.csv",
        help="Cohort QC manifest CSV path",
    )
    ap.add_argument(
        "--infer_uniprot",
        action="store_true",
        help="If metadata lacks UniProt, infer from PDBe for antigen chain",
    )
    ap.add_argument(
        "--struct_fallback",
        action="store_true",
        help="Add distance-based epitope mask and merge when IEDB/map fails",
    )
    ap.add_argument("--contact_cutoff", type=float, default=5.0)
    ap.add_argument(
        "--pdb_scope",
        action="store_true",
        help="Restrict IEDB rows to complex__pdb_id (or --iedb_pdb_col) matching each structure's 4-letter code (IEDB-3D–aligned harvest). prior_source becomes iedb_pdb_sifts when mapped.",
    )
    ap.add_argument(
        "--iedb_pdb_col",
        default="complex__pdb_id",
        help="Column with PDB id for --pdb_scope (IEDB bcell_export: complex__pdb_id)",
    )
    ap.add_argument(
        "--pdb_scope_skip_uniprot_match",
        action="store_true",
        help="With --pdb_scope, do not intersect IEDB rows with metadata/inferred antigen UniProt (use all positive assays for that PDB).",
    )
    args = ap.parse_args()

    pdb_ids = _collect_pdb_ids(args)
    if not pdb_ids:
        print("No PDB IDs.", file=sys.stderr)
        return 2

    meta_df = _read_metadata_rows(args.metadata)
    meta_map = _metadata_dict_from_df(meta_df)

    iedb_path = args.iedb_csv
    if iedb_path.endswith(".gz"):
        iedb_df = pd.read_csv(gzip.open(iedb_path, "rt"), low_memory=False)
    else:
        iedb_df = pd.read_csv(iedb_path, low_memory=False)
    ucol = _detect_iedb_uniprot_column(iedb_df, args.iedb_uniprot_col)
    if args.pdb_scope and args.iedb_pdb_col not in iedb_df.columns:
        raise SystemExit(
            f"--pdb_scope requires column {args.iedb_pdb_col!r} in IEDB CSV (missing). "
            "Use IQ-API bcell_export or fetch_iedb_bcell_pdb_scoped.py."
        )

    cache = Path(args.sifts_cache_dir) if args.sifts_cache_dir else None
    session = requests.Session()
    session.headers.update({"User-Agent": "Epi4Ab-IEDB-prior/1.0"})

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
                iedb_df=iedb_df,
                iedb_uniprot_col=ucol,
                nodes_edges_dir=args.nodes_edges_dir,
                processed_dir=args.processed_dir,
                sifts_cache=cache,
                session=session,
                output_name=args.output_parquet,
                infer_uniprot=args.infer_uniprot,
                struct_fallback=args.struct_fallback,
                contact_cutoff=args.contact_cutoff,
                pdb_scope=args.pdb_scope,
                iedb_pdb_col=args.iedb_pdb_col,
                pdb_scope_require_uniprot_match=not args.pdb_scope_skip_uniprot_match,
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
