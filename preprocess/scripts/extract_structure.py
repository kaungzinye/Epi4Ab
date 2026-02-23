import os
from tqdm import tqdm
from Bio.PDB.PDBIO import Select, PDBIO
from Bio.PDB.MMCIFParser import FastMMCIFParser
import json
from pathlib import Path


AA_3_TO_1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}


def _chain_sequence(chain) -> str:
    seq = []
    for res in chain:
        if res.get_id()[0] != ' ':
            continue
        seq.append(AA_3_TO_1.get(res.get_resname(), 'X'))
    return ''.join(seq)


def _looks_like_antibody_variable(seq: str) -> bool:
    """Conservative heuristic: Ig variable domains often start with these motifs."""
    s = (seq or '').upper()
    if len(s) < 80 or len(s) > 300:
        return False
    prefixes = (
        'QVQL', 'EVQL', 'DVQL', 'DIQMT', 'DIVMT', 'DVVMT', 'EIVLT', 'QSVLT',
        'IVLTQ', 'IQLVQ', 'VQLVQ', 'QKQVL',
    )
    return s.startswith(prefixes)


def _autodetect_antigen_chain(structure, requested_chain: str | None = None):
    """Pick antigen chain by excluding antibody-like chains and choosing largest remaining."""
    model = structure[0]
    chains = [c for c in model]
    chain_info = []
    for c in chains:
        seq = _chain_sequence(c)
        chain_info.append({
            'chain_id': c.get_id(),
            'len': len(seq),
            'seq': seq,
            'antibody_like': _looks_like_antibody_variable(seq)
        })

    # If requested chain exists and does not look antibody-like, keep it.
    if requested_chain:
        req = next((x for x in chain_info if x['chain_id'] == requested_chain), None)
        if req and not req['antibody_like'] and req['len'] > 0:
            return requested_chain, chain_info, 'metadata'

    # Choose largest non-antibody-like chain
    candidates = [x for x in chain_info if (not x['antibody_like']) and x['len'] > 0]
    if candidates:
        best = max(candidates, key=lambda x: x['len'])
        return best['chain_id'], chain_info, 'auto_non_antibody_largest'

    # Fallback: choose largest chain overall
    non_empty = [x for x in chain_info if x['len'] > 0]
    if non_empty:
        best = max(non_empty, key=lambda x: x['len'])
        return best['chain_id'], chain_info, 'auto_largest_fallback'

    return requested_chain, chain_info, 'auto_failed'

class RecSelect(Select):
    def __init__(self, cond_lst, rare_aa):
        self.cond = cond_lst
        self.rare_aa = rare_aa

    def accept_chain(self, chain):
        if chain.get_id() in self.cond:
            return True
        else:
            return False
            
    def accept_residue(self,residue):
        if (residue.get_id()[0] == ' ') & (residue.get_resname() != 'UNK'):
            return True
        elif (residue.get_id()[0][2:] in self.rare_aa):
            residue.resname = self.rare_aa[residue.resname]
            residue.id = tuple([' ',residue.id[1],residue.id[2]])
            return True
        else:
            return False

    def accept_atom(self, atom):
        if (atom.get_altloc() == ' '):
            return True
        elif (atom.get_altloc() == 'A'):
            atom.set_altloc(' ')
            return True
        else:
            return False

def filter_pdb(pdb_df, logging):
    parser = FastMMCIFParser(auth_residues = False, QUIET = True)
    io = PDBIO()
    with open(Path(__file__).parent / 'amino_acid_rare.json', 'r') as f:
        rare_aa = json.load(f)
    for pdbId in tqdm(pdb_df.pdbID, desc = 'Filter structure', unit='pdb'):
        # Filter PDB structure
        pdbSeries = pdb_df[pdb_df.pdbID == pdbId]
        pdb_id_path = os.path.join(logging.directory_data, pdbId)
        pdbId = pdbSeries.pdbID.item()
        pdb = pdbSeries.pdb.item()
        AgChain = pdbSeries.antigen.item()
        # Normalize to single-character chain IDs (allow comma-separated or multi-char strings)
        if isinstance(AgChain, str):
            AgChain = AgChain.strip()
        try:
            structure = parser.get_structure(pdbId, os.path.join(pdb_id_path, f'{pdb}.cif'))

            # Optional auto-detection if metadata chain looks wrong
            chosen_chain = AgChain
            if getattr(logging, 'autodetect_antigen_chain', False):
                chosen_chain, chain_info, mode = _autodetect_antigen_chain(structure, requested_chain=AgChain)
                if chosen_chain != AgChain:
                    print(f"[{pdbId}] Antigen chain override: {AgChain} -> {chosen_chain} (mode={mode})")
                    # Useful debug: show top 3 chains
                    top = sorted(chain_info, key=lambda x: x['len'], reverse=True)[:3]
                    for t in top:
                        print(f"  chain {t['chain_id']}: len={t['len']} antibody_like={t['antibody_like']} seq_head={t['seq'][:12]}")
            else:
                # Even without autodetection, warn if metadata chain looks antibody-like
                try:
                    model = structure[0]
                    if chosen_chain in model:
                        seq = _chain_sequence(model[chosen_chain])
                        if _looks_like_antibody_variable(seq):
                            print(f"[{pdbId}] WARNING: metadata antigen chain {chosen_chain} looks antibody-like (seq_head={seq[:12]})")
                except Exception:
                    pass
                 
            io.set_structure(structure)

            io.save(os.path.join(pdb_id_path, 'lig.pdb'), RecSelect(chosen_chain,rare_aa))
        except Exception as e:
            print(pdbId)
            logging.error_extract_structure.append(pdbId)
            if hasattr(logging, 'log_step_error'):
                logging.log_step_error(pdbId, 'extract_structure', e)
        
    if not logging.error_extract_structure:
        logging.message += '''
All pdb structures have been extracted successfully.'''
    else:
        logging.message += f'''
Not extract structure pdb(s): {logging.error_extract_structure}'''
