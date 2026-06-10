"""
Per-residue CDR proximity features for antigen residues using ANARCI.

For each antigen residue CA (from lig.pdb), computes the minimum CA-CA distance
to each CDR loop (H1-L3) identified by IMGT numbering via ANARCI.

ANARCI directly numbers antibody chain sequences using germline HMM profiles,
so CDR residues are found structurally — no metadata CDR sequences required.

IMGT CDR ranges (same for H and L chains):
  CDR1: 27-38   CDR2: 56-65   CDR3: 105-117

Output: {pdb_id}/cdr_distances/cdr_dist_result.parquet
Columns: pdbId, resId, min_dist_H1, min_dist_H2, min_dist_H3,
                        min_dist_L1, min_dist_L2, min_dist_L3
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from Bio.PDB.MMCIFParser import FastMMCIFParser
from Bio.PDB import PDBParser

# HMMER is needed by ANARCI — ensure conda bin is on PATH
import sys
_CONDA_BIN = '/leonardo_work/AIFAC_F01_302/Epi4Ab/tools/conda/bin'
if _CONDA_BIN not in os.environ.get('PATH', ''):
    os.environ['PATH'] = _CONDA_BIN + ':' + os.environ.get('PATH', '')

from anarci import anarci as _anarci_run

# IMGT CDR position ranges (inclusive)
_IMGT_CDR = {
    'H': {'H1': (27, 38), 'H2': (56, 65), 'H3': (105, 117)},
    'K': {'L1': (27, 38), 'L2': (56, 65), 'L3': (105, 117)},
    'L': {'L1': (27, 38), 'L2': (56, 65), 'L3': (105, 117)},
}

MISSING_DIST = 100.0
CDR_NAMES = ['H1', 'H2', 'H3', 'L1', 'L2', 'L3']

AA_3_TO_1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
}


def _std_residues(chain):
    return [r for r in chain if r.get_id()[0] == ' ']


def _chain_sequence(chain) -> str:
    return ''.join(AA_3_TO_1.get(r.get_resname(), 'X') for r in _std_residues(chain))


def _run_anarci(chain_id: str, seq: str):
    """
    Run ANARCI on a single chain sequence.
    Returns (chain_type, numbered_list, start) or None if no match.
      chain_type: 'H', 'K', or 'L'
      numbered_list: [((imgt_pos, ins_code), aa), ...]  with '-' for gaps
      start: 0-based index in seq where the V-domain begins
    """
    if len(seq) < 70:
        return None
    try:
        result = _anarci_run(
            [(chain_id, seq)],
            scheme='imgt',
            output=False,
            allow=set(['H', 'K', 'L']),
        )
        numbered_seqs = result[0]   # list of per-sequence results
        details       = result[1]
        if not numbered_seqs or not numbered_seqs[0]:
            return None
        aln = numbered_seqs[0][0]   # first alignment: (numbered_list, start, end)
        numbered_list, start, end = aln[0], aln[1], aln[2]
        chain_type = details[0][0]['chain_type']
        return chain_type, numbered_list, start
    except Exception:
        return None


def _cdr_residue_indices(numbered_list, start: int, cdr_range: tuple) -> list:
    """
    Map IMGT CDR positions back to 0-based indices into the chain's residue list.
    Gaps ('-') don't consume a residue; everything else does.
    """
    lo, hi = cdr_range
    indices = []
    res_idx = start
    for (imgt_pos, ins_code), aa in numbered_list:
        if aa == '-':
            continue
        if lo <= imgt_pos <= hi:
            indices.append(res_idx)
        res_idx += 1
    return indices


def _ca_coords(chain, indices: list) -> np.ndarray:
    residues = _std_residues(chain)
    coords = []
    for i in indices:
        if i < len(residues) and 'CA' in residues[i]:
            coords.append(residues[i]['CA'].get_vector().get_array())
    return np.array(coords, dtype=float) if coords else np.empty((0, 3))


def _min_dist(ag_ca: np.ndarray, cdr_ca: np.ndarray) -> float:
    if len(cdr_ca) == 0:
        return MISSING_DIST
    return float(np.min(np.linalg.norm(cdr_ca - ag_ca, axis=1)))


def _get_cdr_ca(structure, antigen_chain_id: str) -> dict:
    """
    Run ANARCI on every non-antigen chain, assign CDRs by IMGT numbering.
    Returns {cdr_name: ndarray(N, 3)}.
    """
    model = structure[0]
    cdr_coords = {name: np.empty((0, 3)) for name in CDR_NAMES}

    for chain in model:
        if chain.get_id() == antigen_chain_id:
            continue
        seq = _chain_sequence(chain)
        anarci_result = _run_anarci(chain.get_id(), seq)
        if anarci_result is None:
            continue
        chain_type, numbered_list, start = anarci_result
        cdr_map = _IMGT_CDR.get(chain_type, {})
        for cdr_name, cdr_range in cdr_map.items():
            if len(cdr_coords[cdr_name]) > 0:
                continue  # already found from another chain
            indices = _cdr_residue_indices(numbered_list, start, cdr_range)
            coords = _ca_coords(chain, indices)
            if len(coords) > 0:
                cdr_coords[cdr_name] = coords

    return cdr_coords


def extract_cdr_distances(pdb_df: pd.DataFrame, logging) -> None:
    """
    Compute CDR distance features for every PDB in pdb_df.

    Reads:
      {directory_data}/{pdbId}/{pdb}.cif  — full structure (antibody + antigen)
      {directory_data}/{pdbId}/lig.pdb    — filtered antigen chain

    Writes:
      {directory_data}/{pdbId}/cdr_distances/cdr_dist_result.parquet
    """
    cif_parser = FastMMCIFParser(auth_residues=False, QUIET=True)
    pdb_parser = PDBParser(QUIET=True)

    for _, row in tqdm(pdb_df.iterrows(), total=len(pdb_df),
                       desc='CDR distances', unit='pdb'):
        pdb_id = str(row['pdbID'])
        pdb_stem = str(row['pdb'])
        antigen_chain = (str(row['antigen']).strip()
                         if pd.notna(row.get('antigen')) else None)

        data_path = os.path.join(logging.directory_data, pdb_id)
        cif_path = os.path.join(data_path, f'{pdb_stem}.cif')
        lig_path = os.path.join(data_path, 'lig.pdb')
        out_dir = os.path.join(data_path, 'cdr_distances')
        out_path = os.path.join(out_dir, 'cdr_dist_result.parquet')

        if not os.path.exists(lig_path) or not os.path.exists(cif_path):
            print(f'[{pdb_id}] lig.pdb or CIF missing — skip')
            continue

        try:
            full_structure = cif_parser.get_structure(pdb_id, cif_path)
        except Exception as e:
            print(f'[{pdb_id}] CIF parse error: {e}')
            continue

        try:
            ag_structure = pdb_parser.get_structure(pdb_id, lig_path)
        except Exception as e:
            print(f'[{pdb_id}] lig.pdb parse error: {e}')
            continue

        cdr_ca = _get_cdr_ca(full_structure, antigen_chain)

        # Diagnostic: which CDRs were found
        found = [n for n in CDR_NAMES if len(cdr_ca[n]) > 0]
        missing = [n for n in CDR_NAMES if len(cdr_ca[n]) == 0]
        if missing:
            print(f'[{pdb_id}] CDRs found={found}  missing(→{MISSING_DIST}Å)={missing}')

        rows = []
        for chain in ag_structure[0]:
            for res in _std_residues(chain):
                if 'CA' not in res:
                    continue
                ag_ca = res['CA'].get_vector().get_array()
                entry = {'pdbId': pdb_id, 'resId': int(res.get_id()[1])}
                for name in CDR_NAMES:
                    entry[f'min_dist_{name}'] = _min_dist(ag_ca, cdr_ca[name])
                rows.append(entry)

        if not rows:
            print(f'[{pdb_id}] No CA atoms in lig.pdb — skip')
            continue

        os.makedirs(out_dir, exist_ok=True)
        pd.DataFrame(rows).to_parquet(out_path, index=False)
