import os
from Bio.PDB import PDBParser
from tqdm import tqdm
import pandas as pd

def extract_aac_from_chain(chain):
    aa_dict = {
        "ALA":0,
        "VAL":0,
        "LEU":0,
        "ILE":0,
        "PRO":0,
        "MET":0,
        "PHE":0,
        "TRP":0,
        "GLY":0,
        "SER":0,
        "THR":0,
        "CYS":0,
        "ASN":0,
        "GLN":0,
        "TYR":0,
        "LYS":0,
        "ARG":0,
        "HIS":0,
        "ASP":0,
        "GLU":0
    }
    seq_len = 0
    for res in chain:
        resname = res.get_resname()
        if resname in aa_dict:
            aa_dict[resname] += 1
            seq_len += 1
    if seq_len == 0:
        return {k: 0.0 for k in aa_dict}
    for key, value in aa_dict.items():
        aa_dict[key] = value / seq_len
    return aa_dict

def extract_aac(pdb_df, logging):
    parser = PDBParser(QUIET = True)
    for pdb_id in tqdm(pdb_df.pdbID, desc='Extract AAC', unit='pdb'):
        try:
            data_path = os.path.join(logging.directory_data, pdb_id)
            pdb_file = os.path.join(data_path, 'lig.pdb')
            structure = parser.get_structure(pdb_id, pdb_file)

            aac_profile = {}
            for chain in structure[0]:
                aac_profile[chain.id] = extract_aac_from_chain(chain)

            if not aac_profile:
                raise RuntimeError('No chains found in lig.pdb')

            df = pd.DataFrame(aac_profile).reset_index(names='resName').melt(
                id_vars=['resName'],
                value_vars=aac_profile.keys(),
                var_name='chainId',
                value_name='aac'
            )

            output_path = os.path.join(data_path, 'aac')
            if not os.path.exists(output_path):
                os.mkdir(output_path)
            df[['resName', 'chainId']] = df[['resName', 'chainId']].astype('string')
            df.to_parquet(os.path.join(output_path, 'aac_result.parquet'))
        except Exception as e:
            if hasattr(logging, 'error_aac'):
                logging.error_aac.append(pdb_id)
            if hasattr(logging, 'log_step_error'):
                logging.log_step_error(pdb_id, 'extract_aac', e)
