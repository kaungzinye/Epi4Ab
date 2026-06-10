#!/usr/bin/env python3
"""
Generate Delta ASA labels per antigen residue using a swappable ASA backend.

Supported backends (see --asa_backend):
  - dssp: absolute residue ASA from DSSP (mkdssp)
  - freesasa: absolute per-residue SASA from the Python freesasa package
  - biopython_sr: Shrake–Rupley residue SASA via Bio.PDB.SASA (same biopython as MMCIF parsing)

Output per PDB (under --nodes_edges_dir/<pdb_id>):
  node_label_dasa.parquet with columns:
    - resId:int
    - score:float in [0,1]
    - asa_alone:float
    - asa_complex:float
    - delta_asa:float
    - delta_asa_clipped:float

Label definition:
  delta_asa = asa_alone - asa_complex
  delta_asa_clipped = max(delta_asa, 0)
  score = minmax(delta_asa_clipped) per complex          (--normalization minmax, default)
  score = clip(delta_asa_clipped / maxASA(residue), 0, 1) (--normalization rsa)

RSA normalization (Tien et al. 2013 max ASA) is per-residue and complex-independent,
so it does not manufacture a 1.0 residue or stretch noise the way per-complex min-max does.

The script enforces strict alignment to node_feature.parquet:
  - one output row per node_feature row
  - output resId order must match node_feature.resId order exactly
  - failures per PDB are reported and skipped (batch continues)
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from Bio.PDB import MMCIFParser, PDBIO, PDBParser, Select
    from Bio.PDB.Polypeptide import is_aa
except ImportError:
    MMCIFParser = None  # type: ignore
    PDBIO = None  # type: ignore
    PDBParser = None  # type: ignore
    Select = object  # type: ignore
    is_aa = None  # type: ignore


ResidueKey = Tuple[int, str]  # (resseq, icode)
AsaFullKey = Tuple[str, int, str]  # (chain_id, resseq, icode)
AsaOrderedEntry = Tuple[str, int, str, str, float]  # chain, resseq, icode, aa_one, asa


@dataclass
class AsaBackendResult:
    """Contract between ASA engines and the shared alignment / scoring layer."""

    asa_map: Dict[AsaFullKey, float]
    ordered_entries: List[AsaOrderedEntry]


@dataclass
class ResidueRecord:
    chain_id: str
    resseq: int
    icode: str
    resname: str

    @property
    def key(self) -> ResidueKey:
        return (self.resseq, self.icode)


AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


def _safe_minmax(values: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype=float)
    if vals.size == 0:
        return vals
    vmin = float(np.nanmin(vals))
    vmax = float(np.nanmax(vals))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return np.zeros_like(vals, dtype=float)
    out = (vals - vmin) / (vmax - vmin)
    out[~np.isfinite(out)] = 0.0
    return np.clip(out, 0.0, 1.0)


# Tien et al. 2013 theoretical maximum solvent accessibility (Gly-X-Gly), in Ų.
# Used for RSA-style normalization: score = delta_asa_clipped / maxASA(residue).
# Per-residue and complex-independent, unlike per-complex min-max.
_MAX_ASA_TIEN = {
    "A": 129.0, "R": 274.0, "N": 195.0, "D": 193.0, "C": 167.0,
    "E": 223.0, "Q": 225.0, "G": 104.0, "H": 224.0, "I": 197.0,
    "L": 201.0, "K": 236.0, "M": 224.0, "F": 240.0, "P": 159.0,
    "S": 155.0, "T": 172.0, "W": 285.0, "Y": 263.0, "V": 174.0,
}
# Fallback for unknown / non-standard residues: median of the table.
_MAX_ASA_DEFAULT = float(np.median(list(_MAX_ASA_TIEN.values())))


def _rsa_normalize(delta_clipped: np.ndarray, res_short: Sequence[str]) -> np.ndarray:
    """Relative-ΔASA: fraction of each residue's max surface buried on binding.

    score_i = clip(delta_asa_clipped_i / maxASA(res_i), 0, 1)

    This is per-residue and independent of the rest of the complex, so it does
    not manufacture a 1.0 residue or stretch noise the way per-complex min-max does.
    """
    vals = np.asarray(delta_clipped, dtype=float)
    if vals.size == 0:
        return vals
    denom = np.array(
        [_MAX_ASA_TIEN.get(str(r).strip().upper()[:1], _MAX_ASA_DEFAULT) for r in res_short],
        dtype=float,
    )
    out = np.divide(vals, denom, out=np.zeros_like(vals), where=denom > 0)
    out[~np.isfinite(out)] = 0.0
    return np.clip(out, 0.0, 1.0)


def _split_chain_field(value: str) -> str:
    raw = (value or "").strip()
    if not raw:
        return raw
    token = re.split(r"[|,; ]+", raw)[0].strip()
    return token[:1] if token else raw[:1]


def _read_metadata_map(metadata_path: str) -> Dict[str, Dict[str, str]]:
    df = pd.read_csv(metadata_path, comment="#")
    if "pdbID" not in df.columns:
        raise ValueError(f"metadata file lacks pdbID column: {metadata_path}")
    out: Dict[str, Dict[str, str]] = {}
    for _, row in df.iterrows():
        pdb_id = str(row["pdbID"])
        out[pdb_id] = {
            "pdb": str(row["pdb"]).strip().lower() if "pdb" in df.columns else "",
            "antigen": _split_chain_field(str(row["antigen"])) if "antigen" in df.columns else "",
        }
    return out


def _collect_pdb_ids(args: argparse.Namespace) -> List[str]:
    ids: List[str] = []
    if args.pdb_id:
        ids.extend([str(x) for x in args.pdb_id])
    if args.pdb_list:
        df = pd.read_csv(args.pdb_list, comment="#")
        col = next((c for c in ("pdbID", "pdbId", "pdb_id") if c in df.columns), None)
        if col is None:
            col = df.columns[0]
        ids.extend(df[col].astype(str).tolist())
    # preserve order, unique
    seen = set()
    uniq: List[str] = []
    for pid in ids:
        if pid not in seen:
            uniq.append(pid)
            seen.add(pid)
    return uniq


def _find_complex_structure(processed_dir: str, pdb_id: str, pdb_code: str) -> str:
    base = os.path.join(processed_dir, pdb_id)
    if not os.path.isdir(base):
        raise FileNotFoundError(f"missing processed dir for {pdb_id}: {base}")
    candidates: List[str] = []
    if pdb_code:
        candidates.extend(
            [
                os.path.join(base, f"{pdb_code}.pdb"),
                os.path.join(base, f"{pdb_code}.cif"),
            ]
        )
    candidates.extend(
        [
            os.path.join(base, f"{pdb_id}.pdb"),
            os.path.join(base, f"{pdb_id}.cif"),
        ]
    )
    for path in candidates:
        if os.path.exists(path):
            return path
    # fallback: any pdb/cif besides lig.pdb
    files = sorted(os.listdir(base))
    for name in files:
        lname = name.lower()
        if lname == "lig.pdb":
            continue
        if lname.endswith(".pdb") or lname.endswith(".cif"):
            return os.path.join(base, name)
    raise FileNotFoundError(f"no full-complex structure found under {base}")


def _parse_pdb_residues(pdb_path: str) -> List[ResidueRecord]:
    residues: List[ResidueRecord] = []
    seen = set()
    with open(pdb_path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            if len(line) < 27:
                continue
            chain_id = line[21].strip()
            resseq_raw = line[22:26].strip()
            icode = line[26].strip()
            resname = line[17:20].strip().upper()
            if not resseq_raw:
                continue
            try:
                resseq = int(resseq_raw)
            except ValueError:
                continue
            key = (chain_id, resseq, icode)
            if key in seen:
                continue
            seen.add(key)
            residues.append(ResidueRecord(chain_id, resseq, icode, resname))
    if not residues:
        raise ValueError(f"no residues parsed from {pdb_path}")
    return residues


def _run_dssp_once(input_path: str, dssp_bin: str, out_path: str) -> Tuple[bool, str]:
    candidate_cmds = [
        [dssp_bin, input_path, out_path],
        [dssp_bin, "-i", input_path, "-o", out_path],
    ]
    last_error = ""
    try:
        for cmd in candidate_cmds:
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if proc.returncode == 0:
                return True, ""
            stderr = (proc.stderr or "").strip()
            stdout = (proc.stdout or "").strip()
            last_error = stderr or stdout or f"exit_code={proc.returncode}"
    except FileNotFoundError as exc:
        return False, f"DSSP binary not found: {dssp_bin} ({exc})"
    return False, last_error or "unknown error"


class _PolymerOnlyPdbSelect(Select):
    """Drop waters/ligands when exporting temporary PDBs from mmCIF."""

    def accept_residue(self, residue) -> int:
        hetflag = str(residue.id[0] or "").strip()
        if hetflag == "W":
            return 0
        if hetflag and is_aa is not None and not is_aa(residue, standard=False):
            return 0
        return 1


def _cif_to_temp_pdb(cif_path: str) -> str:
    if MMCIFParser is None or PDBIO is None:
        raise RuntimeError("Bio.PDB is required for CIF→PDB conversion; install biopython")
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("complex", cif_path)
    fd, tmp_pdb = tempfile.mkstemp(suffix=".pdb", text=False)
    os.close(fd)
    io = PDBIO()
    io.set_structure(structure)
    io.save(tmp_pdb, select=_PolymerOnlyPdbSelect())
    return tmp_pdb


def _run_dssp(input_path: str, dssp_bin: str) -> str:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".dssp", delete=False) as tmp:
        out_path = tmp.name
    ok, err = _run_dssp_once(input_path, dssp_bin, out_path)
    if ok:
        return out_path

    tmp_pdb: Optional[str] = None
    try:
        if input_path.lower().endswith(".cif"):
            tmp_pdb = _cif_to_temp_pdb(input_path)
            ok2, err2 = _run_dssp_once(tmp_pdb, dssp_bin, out_path)
            if ok2:
                return out_path
            raise RuntimeError(
                f"DSSP failed for {input_path} (and temp PDB fallback): {err2 or err}"
            )
    finally:
        if tmp_pdb and os.path.exists(tmp_pdb):
            try:
                os.unlink(tmp_pdb)
            except OSError:
                pass

    raise RuntimeError(f"DSSP failed for {input_path}: {err}")


def _parse_dssp_file(path: str) -> Dict[Tuple[str, int, str], float]:
    # map key: (chain_id, resseq, icode) -> ACC (absolute ASA)
    out: Dict[Tuple[str, int, str], float] = {}
    entries = _parse_dssp_entries(path)
    for chain_id, resseq, icode, _aa, acc in entries:
        out[(chain_id, resseq, icode)] = acc
    if not out:
        raise ValueError(f"no DSSP residue entries parsed from {path}")
    return out


def _parse_dssp_entries(path: str) -> List[Tuple[str, int, str, str, float]]:
    # ordered entries as they appear in DSSP
    # tuple: (chain_id, resseq, icode, aa_one_letter, acc)
    out: List[Tuple[str, int, str, str, float]] = []
    in_body = False
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if not in_body:
                if line.startswith("  #  RESIDUE AA STRUCTURE"):
                    in_body = True
                continue
            if len(line) < 39:
                continue
            aa = line[13:14]
            if aa == "!":
                continue  # missing residue marker in DSSP
            resseq_raw = line[5:10].strip()
            icode = line[10:11].strip()
            chain_id = line[11:12].strip()
            acc_raw = line[34:38].strip()
            if not resseq_raw or not acc_raw:
                continue
            try:
                resseq = int(resseq_raw)
                acc = float(int(acc_raw))
            except ValueError:
                continue
            out.append((chain_id, resseq, icode, aa.strip() or "X", acc))
    if not out:
        raise ValueError(f"no DSSP residue entries parsed from {path}")
    return out


def _ensure_pdb_path_for_freesasa(struct_path: str) -> Tuple[str, Optional[str]]:
    """Return (path_for_freesasa, temp_pdb_to_delete). FreeSASA reads PDB; mmCIF is converted."""
    lower = struct_path.lower()
    if lower.endswith(".cif"):
        tmp = _cif_to_temp_pdb(struct_path)
        return tmp, tmp
    return struct_path, None


def _parse_freesasa_residue_label(resnum_label: str) -> Tuple[int, str]:
    """
    Map FreeSASA residueNumber strings to (resseq, insertion_code).
    Handles forms like '123', '123A' (insertion), and signed integers.
    """
    s = (resnum_label or "").strip().upper()
    if not s:
        return 0, ""
    if len(s) > 1 and s[-1].isalpha() and s[-2].isdigit():
        icode = s[-1]
        num_part = s[:-1]
    else:
        icode = ""
        num_part = s
    digits = "".join(c for c in num_part if c.isdigit() or c in "+-")
    if not digits or not any(ch.isdigit() for ch in digits):
        return 0, ""
    try:
        return int(digits), icode
    except ValueError:
        return 0, ""


def _freesasa_aa_one(area: object) -> str:
    try:
        raw = (getattr(area, "residueType", "") or "").strip().upper()
    except Exception:
        raw = ""
    if not raw:
        return "X"
    if len(raw) == 1:
        return raw if raw.isalpha() else "X"
    return AA3_TO_1.get(raw[:3], "X")


def _compute_asa_backend_dssp(struct_path: str, dssp_bin: str) -> AsaBackendResult:
    dssp_path = _run_dssp(struct_path, dssp_bin)
    try:
        asa_map = _parse_dssp_file(dssp_path)
        ordered = _parse_dssp_entries(dssp_path)
        return AsaBackendResult(asa_map=asa_map, ordered_entries=ordered)
    finally:
        try:
            os.unlink(dssp_path)
        except OSError:
            pass


def _compute_asa_backend_freesasa(struct_path: str) -> AsaBackendResult:
    try:
        import freesasa
    except ImportError as exc:
        raise RuntimeError(
            "FreeSASA backend requires the Python 'freesasa' package. "
            "Install with: pip install freesasa"
        ) from exc

    pdb_path, tmp_pdb = _ensure_pdb_path_for_freesasa(struct_path)
    try:
        try:
            parameters = freesasa.Parameters({"algorithm": "lee-richards"})
        except Exception:
            parameters = freesasa.Parameters()
        structure = freesasa.Structure(pdb_path)
        result = freesasa.calc(structure, parameters)
        residue_areas = result.residueAreas()
    except Exception as exc:
        raise RuntimeError(f"FreeSASA failed for {struct_path}: {exc}") from exc
    finally:
        if tmp_pdb and os.path.exists(tmp_pdb):
            try:
                os.unlink(tmp_pdb)
            except OSError:
                pass

    asa_map: Dict[AsaFullKey, float] = {}
    row_by_key: Dict[AsaFullKey, AsaOrderedEntry] = {}

    for chain_id, chain_residues in residue_areas.items():
        chain = (chain_id or "").strip() or "A"
        for _rk, area in chain_residues.items():
            try:
                resnum_label = str(getattr(area, "residueNumber", "") or "").strip()
                resseq, icode = _parse_freesasa_residue_label(resnum_label)
                total = float(getattr(area, "total", 0.0) or 0.0)
            except Exception:
                continue
            if resseq == 0 and not icode:
                continue
            aa_one = _freesasa_aa_one(area)
            key: AsaFullKey = (chain, resseq, icode)
            asa_map[key] = total
            row_by_key[key] = (chain, resseq, icode, aa_one, total)

    if not asa_map:
        raise ValueError(f"no FreeSASA residue areas parsed from {struct_path}")

    ordered_entries = sorted(row_by_key.values(), key=lambda t: (t[0], t[1], t[2] or ""))
    return AsaBackendResult(asa_map=asa_map, ordered_entries=ordered_entries)


def _load_biopython_structure(struct_path: str):
    if PDBParser is None or MMCIFParser is None:
        raise RuntimeError("Bio.PDB is required for biopython_sr; install biopython")
    lower = struct_path.lower()
    if lower.endswith(".cif"):
        parser = MMCIFParser(QUIET=True)
        return parser.get_structure("asa", struct_path)
    parser = PDBParser(QUIET=True)
    return parser.get_structure("asa", struct_path)


def _icode_normalize(icode) -> str:
    if icode is None:
        return ""
    s = str(icode).strip()
    if s in (".", "None"):
        return ""
    return s


def _compute_asa_backend_biopython_sr(struct_path: str) -> AsaBackendResult:
    try:
        from Bio.PDB.SASA import ShrakeRupley
    except ImportError as exc:
        raise RuntimeError(
            "biopython_sr backend requires Bio.PDB.SASA (Shrake–Rupley); install biopython"
        ) from exc

    structure = _load_biopython_structure(struct_path)
    sr = ShrakeRupley(probe_radius=1.4, n_points=100)
    model = structure[0]
    sr.compute(model, level="R")

    asa_map: Dict[AsaFullKey, float] = {}
    row_by_key: Dict[AsaFullKey, AsaOrderedEntry] = {}

    for chain in model:
        chain_id = (chain.id or "").strip() or "A"
        for residue in chain:
            if residue.id[0] != " ":
                continue
            _, resseq, icode = residue.id
            icode_s = _icode_normalize(icode)
            if not hasattr(residue, "sasa") or residue.sasa is None:
                continue
            try:
                sasa = float(residue.sasa)
            except (TypeError, ValueError):
                continue
            resname = residue.resname.upper()
            aa_one = AA3_TO_1.get(resname, "X")
            key: AsaFullKey = (chain_id, int(resseq), icode_s)
            asa_map[key] = sasa
            row_by_key[key] = (chain_id, int(resseq), icode_s, aa_one, sasa)

    if not asa_map:
        raise ValueError(f"no Biopython SASA residue data parsed from {struct_path}")

    ordered_entries = sorted(row_by_key.values(), key=lambda t: (t[0], t[1], t[2] or ""))
    return AsaBackendResult(asa_map=asa_map, ordered_entries=ordered_entries)


def compute_asa_backend(struct_path: str, backend: str, dssp_bin: str) -> AsaBackendResult:
    b = (backend or "dssp").lower().strip()
    if b in ("opensasa", "biopython", "biopython_sr", "shrake_rupley"):
        b = "biopython_sr"
    if b == "dssp":
        return _compute_asa_backend_dssp(struct_path, dssp_bin)
    if b == "freesasa":
        return _compute_asa_backend_freesasa(struct_path)
    if b == "biopython_sr":
        return _compute_asa_backend_biopython_sr(struct_path)
    raise ValueError(
        f"unknown ASA backend {backend!r}; expected 'dssp', 'freesasa', or 'biopython_sr'"
    )


def _fill_missing_positional_keys(
    keys: List[Optional[Tuple[str, int, str]]],
) -> Optional[List[Tuple[str, int, str]]]:
    """Fill None slots by carrying the nearest assigned ASA key (terminal gaps)."""
    out: List[Optional[Tuple[str, int, str]]] = list(keys)
    last: Optional[Tuple[str, int, str]] = None
    for i in range(len(out)):
        if out[i] is not None:
            last = out[i]
        elif last is not None:
            out[i] = last
    nxt: Optional[Tuple[str, int, str]] = None
    for i in range(len(out) - 1, -1, -1):
        if out[i] is not None:
            nxt = out[i]
        elif nxt is not None:
            out[i] = nxt
    if any(k is None for k in out):
        return None
    return [(t[0], t[1], t[2]) for t in out]


def _build_complex_positional_keys(
    ligand_residues: Sequence[ResidueRecord],
    complex_entries: Sequence[Tuple[str, int, str, str, float]],
    antigen_chain_hint: str,
    min_aa_fraction: float,
) -> Optional[List[Tuple[str, int, str]]]:
    """
    Fallback: map each ligand residue (in order) to complex ordered ASA rows (chain, resseq, icode).

    The `complex_entries` rows come from the active backend (DSSP via `_parse_dssp_entries`, or
    FreeSASA); the sliding-window alignment logic is the same for both.

    Handles:
    - Longer complex segment than ligand: slide a window of length n along the chain.
    - Shorter complex segment (e.g. one fewer ASA rows): slide ligand start so a length-m
      substring aligns to the full chain; any remaining positions use the nearest assigned key.
    """
    if not ligand_residues or not complex_entries:
        return None

    lig_aa = [AA3_TO_1.get(r.resname.upper(), "X") for r in ligand_residues]
    n = len(lig_aa)
    denom_all = sum(1 for a in lig_aa if a != "X")
    if denom_all == 0:
        return None

    by_chain: Dict[str, List[Tuple[int, str, str, float]]] = {}
    for chain, resseq, icode, aa, acc in complex_entries:
        by_chain.setdefault(chain, []).append((resseq, icode, aa or "X", acc))

    # best: (frac, chain, kind, chain_start, lig_start, rows_or_window)
    # kind 1 = window on chain length n; kind 2 = full chain length m, lig offset lig_start
    best: Optional[
        Tuple[float, str, int, int, int, List[Tuple[int, str, str, float]]]
    ] = None

    def consider(
        frac: float,
        chain: str,
        kind: int,
        c_start: int,
        l_start: int,
        rows_slice: List[Tuple[int, str, str, float]],
    ) -> None:
        nonlocal best
        if frac < min_aa_fraction:
            return
        if best is None:
            best = (frac, chain, kind, c_start, l_start, rows_slice)
            return
        b_frac, b_chain, b_kind, b_c0, b_l0, _ = best
        if frac > b_frac + 1e-12:
            best = (frac, chain, kind, c_start, l_start, rows_slice)
        elif abs(frac - b_frac) <= 1e-12:
            if chain == antigen_chain_hint and b_chain != antigen_chain_hint:
                best = (frac, chain, kind, c_start, l_start, rows_slice)
            elif chain != antigen_chain_hint and b_chain == antigen_chain_hint:
                return
            elif (c_start, l_start, chain) < (b_c0, b_l0, b_chain):
                best = (frac, chain, kind, c_start, l_start, rows_slice)

    for chain, rows in by_chain.items():
        m = len(rows)
        if m == 0:
            continue
        if m >= n:
            for start in range(0, m - n + 1):
                window = rows[start : start + n]
                aa_win = [r[2] for r in window]
                matches = sum(1 for a, b in zip(lig_aa, aa_win) if a == b and a != "X")
                frac = matches / denom_all
                consider(frac, chain, 1, start, 0, window)
        else:
            for lig_start in range(0, n - m + 1):
                seg = lig_aa[lig_start : lig_start + m]
                denom_seg = sum(1 for a in seg if a != "X")
                if denom_seg == 0:
                    continue
                aa_chain = [r[2] for r in rows]
                matches = sum(1 for a, b in zip(seg, aa_chain) if a == b and a != "X")
                frac = matches / denom_seg
                consider(frac, chain, 2, 0, lig_start, rows)

    if best is None:
        return None
    _frac, chain, kind, _c0, lig_start, rows_slice = best
    if kind == 1:
        return [(chain, r[0], r[1]) for r in rows_slice]

    keys: List[Optional[Tuple[str, int, str]]] = [None] * n
    for j, r in enumerate(rows_slice):
        keys[lig_start + j] = (chain, r[0], r[1])
    filled = _fill_missing_positional_keys(keys)
    return filled


def _extract_chain_from_lig(lig_path: str) -> str:
    residues = _parse_pdb_residues(lig_path)
    chain_ids = sorted({r.chain_id for r in residues if r.chain_id})
    if not chain_ids:
        return ""
    return chain_ids[0]


def _build_chain_residue_order(lig_path: str, antigen_chain: str) -> List[ResidueRecord]:
    all_res = _parse_pdb_residues(lig_path)
    if antigen_chain:
        chain_res = [r for r in all_res if r.chain_id == antigen_chain]
        if chain_res:
            return chain_res
    # fallback if chain metadata mismatches lig.pdb chain label
    return all_res


def _align_scores_to_node_feature(
    node_feature: pd.DataFrame,
    ligand_residues: Sequence[ResidueRecord],
    asa_alone_map: Dict[Tuple[str, int, str], float],
    asa_complex_map: Dict[Tuple[str, int, str], float],
    antigen_chain: str,
    complex_entries: Optional[Sequence[Tuple[str, int, str, str, float]]] = None,
    alone_entries: Optional[Sequence[Tuple[str, int, str, str, float]]] = None,
    allow_fallback_mapping: bool = True,
    fallback_min_aa_fraction: float = 0.95,
    normalization: str = "minmax",
) -> pd.DataFrame:
    if "resId" not in node_feature.columns:
        raise ValueError("node_feature.parquet lacks resId")
    if "resShort" not in node_feature.columns:
        raise ValueError("node_feature.parquet lacks resShort")

    # Build occurrence-aware mapping for duplicate residue numbers.
    by_resseq: Dict[int, List[ResidueRecord]] = {}
    for rec in ligand_residues:
        by_resseq.setdefault(rec.resseq, []).append(rec)

    use_counter: Dict[int, int] = {}
    rows = []
    missing = []
    positional_complex_keys: Optional[List[Tuple[str, int, str]]] = None
    positional_alone_keys: Optional[List[Tuple[str, int, str]]] = None
    if allow_fallback_mapping and complex_entries:
        positional_complex_keys = _build_complex_positional_keys(
            ligand_residues=ligand_residues,
            complex_entries=complex_entries,
            antigen_chain_hint=antigen_chain,
            min_aa_fraction=fallback_min_aa_fraction,
        )
    if allow_fallback_mapping and alone_entries:
        positional_alone_keys = _build_complex_positional_keys(
            ligand_residues=ligand_residues,
            complex_entries=alone_entries,
            antigen_chain_hint=antigen_chain,
            min_aa_fraction=fallback_min_aa_fraction,
        )

    # Map (resseq, k-th occurrence of that resseq in ligand order) -> index in ligand_residues.
    # Matches node_feature walking logic so positional_complex_keys[lig_i] lines up with `rec`.
    resseq_occurrence_to_lig_idx: Dict[Tuple[int, int], int] = {}
    resseq_seen: Dict[int, int] = {}
    for lig_i, rec in enumerate(ligand_residues):
        occ = resseq_seen.get(rec.resseq, 0)
        resseq_occurrence_to_lig_idx[(rec.resseq, occ)] = lig_i
        resseq_seen[rec.resseq] = occ + 1

    for _, row in node_feature.iterrows():
        res_id = int(row["resId"])
        res_short = str(row["resShort"]).strip().upper()
        candidates = by_resseq.get(res_id, [])
        idx = use_counter.get(res_id, 0)
        if idx >= len(candidates):
            missing.append(f"resId={res_id} missing in lig.pdb residue order")
            continue
        rec = candidates[idx]
        use_counter[res_id] = idx + 1

        lig_i = resseq_occurrence_to_lig_idx.get((res_id, idx))
        if lig_i is None:
            missing.append(f"resId={res_id} internal ligand index error")
            continue

        # Prefer explicit antigen_chain key in complex map, but fall back to lig chain.
        dssp_chain = antigen_chain or rec.chain_id
        key_complex = (dssp_chain, rec.resseq, rec.icode)
        if key_complex not in asa_complex_map and rec.chain_id:
            key_complex = (rec.chain_id, rec.resseq, rec.icode)
        if key_complex not in asa_complex_map and positional_complex_keys is not None:
            if lig_i < len(positional_complex_keys):
                key_complex = positional_complex_keys[lig_i]
        key_alone = (rec.chain_id, rec.resseq, rec.icode)
        if key_alone not in asa_alone_map and antigen_chain:
            key_alone = (antigen_chain, rec.resseq, rec.icode)
        if key_alone not in asa_alone_map and positional_alone_keys is not None:
            if lig_i < len(positional_alone_keys):
                key_alone = positional_alone_keys[lig_i]

        if key_alone not in asa_alone_map:
            missing.append(f"resId={res_id} missing in antigen-only ASA map")
            continue
        if key_complex not in asa_complex_map:
            missing.append(
                f"resId={res_id} missing in complex ASA map chain={dssp_chain or rec.chain_id}"
            )
            continue

        asa_alone = float(asa_alone_map[key_alone])
        asa_complex = float(asa_complex_map[key_complex])
        delta = asa_alone - asa_complex
        delta_clip = max(delta, 0.0)
        rows.append(
            {
                "resId": res_id,
                "resShort": res_short,
                "asa_alone": asa_alone,
                "asa_complex": asa_complex,
                "delta_asa": delta,
                "delta_asa_clipped": delta_clip,
            }
        )

    if missing:
        preview = "; ".join(missing[:8])
        more = f" (+{len(missing) - 8} more)" if len(missing) > 8 else ""
        raise ValueError(f"residue alignment failed: {preview}{more}")
    if len(rows) != len(node_feature):
        raise ValueError(
            f"alignment size mismatch: rows={len(rows)} node_feature={len(node_feature)}"
        )

    out = pd.DataFrame(rows)
    if normalization == "rsa":
        out["score"] = _rsa_normalize(
            out["delta_asa_clipped"].to_numpy(dtype=float),
            out["resShort"].astype(str).tolist(),
        )
    else:
        out["score"] = _safe_minmax(out["delta_asa_clipped"].to_numpy(dtype=float))
    # enforce column order
    out = out[
        [
            "resId",
            "score",
            "asa_alone",
            "asa_complex",
            "delta_asa",
            "delta_asa_clipped",
        ]
    ]
    # strict order check
    if not out["resId"].equals(node_feature["resId"].astype(int)):
        raise ValueError("resId order mismatch against node_feature")
    return out


def _write_error_log(processed_dir: str, pdb_id: str, message: str) -> None:
    err_dir = os.path.join(processed_dir, pdb_id, "errors")
    os.makedirs(err_dir, exist_ok=True)
    path = os.path.join(err_dir, "generate_dasa_labels.log")
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(message.strip() + "\n")


def process_one(
    pdb_id: str,
    nodes_edges_dir: str,
    processed_dir: str,
    dssp_bin: str,
    output_file: str,
    metadata_map: Dict[str, Dict[str, str]],
    asa_backend: str = "dssp",
    allow_fallback_mapping: bool = True,
    fallback_min_aa_fraction: float = 0.95,
    normalization: str = "minmax",
) -> str:
    if pdb_id not in metadata_map:
        raise KeyError(f"{pdb_id} missing from metadata; required for antigen chain + pdb code")
    row = metadata_map[pdb_id]
    antigen_chain = _split_chain_field(row.get("antigen", ""))
    pdb_code = (row.get("pdb", "") or "").lower().strip()
    if not pdb_code:
        pdb_code = pdb_id.split("_")[0].lower()

    node_feature_path = os.path.join(nodes_edges_dir, pdb_id, "node_feature.parquet")
    if not os.path.exists(node_feature_path):
        raise FileNotFoundError(f"missing node_feature.parquet: {node_feature_path}")
    node_feature = pd.read_parquet(node_feature_path)

    lig_path = os.path.join(processed_dir, pdb_id, "lig.pdb")
    if not os.path.exists(lig_path):
        raise FileNotFoundError(f"missing antigen-only lig.pdb: {lig_path}")
    if not antigen_chain:
        antigen_chain = _extract_chain_from_lig(lig_path)

    complex_path = _find_complex_structure(processed_dir, pdb_id, pdb_code)

    alone_backend = compute_asa_backend(lig_path, asa_backend, dssp_bin)
    complex_backend = compute_asa_backend(complex_path, asa_backend, dssp_bin)

    ligand_residues = _build_chain_residue_order(lig_path, antigen_chain)
    out_df = _align_scores_to_node_feature(
        node_feature=node_feature,
        ligand_residues=ligand_residues,
        asa_alone_map=alone_backend.asa_map,
        asa_complex_map=complex_backend.asa_map,
        antigen_chain=antigen_chain,
        complex_entries=complex_backend.ordered_entries,
        alone_entries=alone_backend.ordered_entries,
        allow_fallback_mapping=allow_fallback_mapping,
        fallback_min_aa_fraction=fallback_min_aa_fraction,
        normalization=normalization,
    )
    out_path = os.path.join(nodes_edges_dir, pdb_id, output_file)
    out_df.to_parquet(out_path, index=False)
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Generate Delta-ASA labels per PDB (DSSP, FreeSASA, or Biopython Shrake–Rupley)."
    )
    ap.add_argument("--nodes_edges_dir", required=True, help="Path to nodes_edges root")
    ap.add_argument("--processed_dir", required=True, help="Path to processed_data root")
    ap.add_argument("--metadata", required=True, help="CSV with pdbID,pdb,antigen columns")
    ap.add_argument("--pdb_id", action="append", help="PDB ID(s) to process, repeatable")
    ap.add_argument("--pdb_list", help="CSV with pdbID column")
    ap.add_argument(
        "--asa_backend",
        choices=("dssp", "freesasa", "biopython_sr"),
        default="dssp",
        help="ASA engine: dssp | freesasa | biopython_sr (Shrake–Rupley). Aliases opensasa/biopython accepted internally.",
    )
    ap.add_argument(
        "--dssp_bin",
        default="mkdssp",
        help="DSSP executable (only used when --asa_backend dssp)",
    )
    ap.add_argument("--output_file", default="node_label_dasa.parquet", help="Output parquet filename")
    ap.add_argument(
        "--normalization",
        choices=("minmax", "rsa"),
        default="minmax",
        help="score normalization: minmax (per-complex, legacy) | rsa (delta_asa/maxASA per residue, complex-independent)",
    )
    ap.add_argument(
        "--strict_residue_mapping",
        action="store_true",
        help="Disable sequence/window ASA fallback (exact residue keys only for alone and complex)",
    )
    ap.add_argument(
        "--fallback_min_aa_fraction",
        type=float,
        default=0.95,
        help="Min non-X AA match fraction for ASA sequence fallback (antigen-only and complex; default 0.95)",
    )
    ap.add_argument("--fail_fast", action="store_true", help="Abort on first failed PDB")
    args = ap.parse_args()

    pdb_ids = _collect_pdb_ids(args)
    if not pdb_ids:
        print("No PDB IDs provided. Use --pdb_id and/or --pdb_list.", file=sys.stderr)
        return 2

    metadata_map = _read_metadata_map(args.metadata)
    ok = 0
    fail = 0
    failures: List[Tuple[str, str]] = []

    for pdb_id in pdb_ids:
        try:
            out_path = process_one(
                pdb_id=pdb_id,
                nodes_edges_dir=args.nodes_edges_dir,
                processed_dir=args.processed_dir,
                dssp_bin=args.dssp_bin,
                output_file=args.output_file,
                metadata_map=metadata_map,
                asa_backend=args.asa_backend,
                allow_fallback_mapping=not args.strict_residue_mapping,
                fallback_min_aa_fraction=args.fallback_min_aa_fraction,
                normalization=args.normalization,
            )
            print(f"OK   {pdb_id} -> {out_path}")
            ok += 1
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            print(f"FAIL {pdb_id}: {msg}", file=sys.stderr)
            _write_error_log(args.processed_dir, pdb_id, msg)
            failures.append((pdb_id, msg))
            fail += 1
            if args.fail_fast:
                break

    print(f"Done. ok={ok} fail={fail}")
    if failures:
        print("Failures:", file=sys.stderr)
        w = csv.writer(sys.stderr)
        for pid, msg in failures:
            w.writerow([pid, msg])
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

