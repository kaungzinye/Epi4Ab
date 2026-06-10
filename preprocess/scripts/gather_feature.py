import pandas as pd
import os
from tqdm import tqdm
from Bio.PDB import PDBParser
import json
from pathlib import Path

def extract_ab_feature(pdb_fam, fam_dict, fam_columns):
    """
    One-hot encode VH/VL family with basic edge-case handling.

    If the family name is missing or not found in fam_dict, we return an
    all-zero vector instead of raising, so a single unexpected family
    name does not break feature gathering for the whole PDB.
    """
    fam_list = [0] * len(fam_columns)
    if pdb_fam is None or len(pdb_fam) == 0:
        return fam_list

    fam = pdb_fam[0]
    # Handle missing/NaN or non-string values gracefully
    try:
        if pd.isna(fam):
            return fam_list
    except Exception:
        pass
    if not isinstance(fam, str):
        return fam_list
    fam = fam.strip()
    if len(fam) == 0:
        return fam_list
    # Direct match
    if fam in fam_dict:
        fam_key = fam_dict[fam]
    else:
        # Try a few tolerant variants (common formatting differences)
        fam_norm = fam.replace("IGHV0", "IGHV").replace("IGKV0", "IGKV").replace("IGLV0", "IGLV")
        if fam_norm in fam_dict:
            fam_key = fam_dict[fam_norm]
        else:
            # Unknown family: leave all zeros
            return fam_list

    if fam_key in fam_columns:
        fam_list[fam_columns.index(fam_key)] = 1
    return fam_list

def gather_feature(pdb_df, logging):
    parser = PDBParser(QUIET = True)
    # Read aa profile
    with open(Path(__file__).parent / 'amino_acid_profile.json', 'r') as f:
        aa_profile = json.load(f)
    with open(Path(__file__).parent / 'vhvl_profile.json', 'r') as f:
        vhvl_profile = json.load(f)
    with open(Path(__file__).parent / 'vhvl_columns.json', 'r') as f:
        vhvl_columns = json.load(f)
    # fw_df = pd.read_csv(Path(__file__).parent / 'framework_template.csv')
    for pdb_id in tqdm(pdb_df.pdbID, desc='Gather all feature', unit='pdb'):
        filter_pdb = pdb_df[pdb_df.pdbID == pdb_id]
        data_path = os.path.join(logging.directory_data, pdb_id)
        pdb_file = os.path.join(data_path, 'lig.pdb')
        missing_any = False

        try:
            structure = parser.get_structure(pdb_id, pdb_file)
        except Exception:
            if pdb_id not in logging.error_gather:
                logging.error_gather.append(pdb_id)
            continue
        profile_lst = []
        for chain in structure[0]:
            for res in chain:
                res_name = res.get_resname()
                # Some structures may contain non-standard residue names.
                # Fall back to 'X' for unknown residues instead of failing.
                res_short = aa_profile['resShort'].get(res_name, 'X')
                profile_lst.append({'pdbId':pdb_id,
                                    'chainId':chain.id,
                                    'resName':res_name,
                                    'resShort':res_short,
                                    'resId':res.id[1]})
        profile_dat = pd.DataFrame(profile_lst)
        profile_dat['resId'] = profile_dat['resId'].astype(int)

        # merge depth (default to 0 if missing)
        depth_path = os.path.join(data_path, 'depth', 'depth_result.parquet')
        if os.path.exists(depth_path):
            try:
                depth_dat = pd.read_parquet(depth_path)
                depth_dat['resId'] = depth_dat['resId'].astype(int)
                profile_dat = profile_dat.merge(depth_dat, how='left')
            except Exception:
                profile_dat['resDepth'] = 0.0
                profile_dat['caDepth'] = 0.0
                missing_any = True
        else:
            profile_dat['resDepth'] = 0.0
            profile_dat['caDepth'] = 0.0
            missing_any = True

        # merge charge (default to 0 if missing)
        charge_path = os.path.join(data_path, 'charge', 'charge_result.parquet')
        if os.path.exists(charge_path):
            try:
                charge_dat = pd.read_parquet(charge_path)
                charge_dat['resId'] = charge_dat['resId'].astype(int)
                profile_dat = profile_dat.merge(charge_dat, how='left')
            except Exception:
                profile_dat['charge'] = 0.0
                missing_any = True
        else:
            profile_dat['charge'] = 0.0
            missing_any = True

        # merge angle (default to 0 if missing)
        angle_path = os.path.join(data_path, 'angle', 'angle_result.parquet')
        angle_defaults = {
            'psi': 0.0,
            'phi': 0.0,
            'omega': 0.0,
            'chi': 0.0,
            'angleNan': 0,
            'chiNan': 0,
        }
        if os.path.exists(angle_path):
            try:
                angle_dat = pd.read_parquet(angle_path)
                angle_dat['resId'] = angle_dat['resId'].astype(int)
                profile_dat = profile_dat.merge(angle_dat, how='left')
            except Exception:
                for k, v in angle_defaults.items():
                    profile_dat[k] = v
                missing_any = True
        else:
            for k, v in angle_defaults.items():
                profile_dat[k] = v
            missing_any = True

        # merge aa profile
        aa_dat = pd.DataFrame(aa_profile).reset_index(names='resName')
        profile_dat = profile_dat.merge(aa_dat, how='left')

        # merge aac (default to 0 if missing)
        aac_path = os.path.join(data_path, 'aac', 'aac_result.parquet')
        if os.path.exists(aac_path):
            try:
                aac_dat = pd.read_parquet(aac_path)
                profile_dat = profile_dat.merge(aac_dat, how='left')
            except Exception:
                profile_dat['aac'] = 0.0
                missing_any = True
        else:
            profile_dat['aac'] = 0.0
            missing_any = True

        # merge charge composition (default to 0 if missing)
        cc_path = os.path.join(data_path, 'charge_composition', 'cc_result.parquet')
        if os.path.exists(cc_path):
            try:
                cc_dat = pd.read_parquet(cc_path)
                profile_dat = profile_dat.merge(cc_dat, how='left')
            except Exception:
                profile_dat['cc'] = 0.0
                missing_any = True
        else:
            profile_dat['cc'] = 0.0
            missing_any = True

        if missing_any and pdb_id not in logging.error_gather:
            logging.error_gather.append(pdb_id)

        # merge CDR distances (per-residue CDR proximity, optional feature)
        cdr_dist_path = os.path.join(data_path, 'cdr_distances', 'cdr_dist_result.parquet')
        cdr_dist_columns = ['min_dist_H1', 'min_dist_H2', 'min_dist_H3',
                            'min_dist_L1', 'min_dist_L2', 'min_dist_L3']
        if os.path.exists(cdr_dist_path):
            try:
                cdr_dist_dat = pd.read_parquet(cdr_dist_path)
                cdr_dist_dat['resId'] = cdr_dist_dat['resId'].astype(int)
                profile_dat = profile_dat.merge(
                    cdr_dist_dat[['resId'] + cdr_dist_columns], on='resId', how='left')
                for col in cdr_dist_columns:
                    profile_dat[col] = profile_dat[col].fillna(100.0)
            except Exception:
                for col in cdr_dist_columns:
                    profile_dat[col] = 100.0
        else:
            for col in cdr_dist_columns:
                profile_dat[col] = 100.0

        # merge antibody feature
        vh_fam = filter_pdb['VH_fam'].values
        vl_fam = filter_pdb['VL_fam'].values
        
        vh_list = extract_ab_feature(vh_fam, vhvl_profile['vh_fam'], vhvl_columns['vh_fam'])
        vl_list = extract_ab_feature(vl_fam, vhvl_profile['vl_fam'], vhvl_columns['vl_fam'])
        profile_dat[vhvl_columns['vh_fam']] = vh_list
        profile_dat[vhvl_columns['vl_fam']] = vl_list

        # If any of the upstream feature merges produced NaNs, prefer a stable
        # numeric default (0) over propagating NaNs into node features.
        numeric_defaults = [
            'resDepth', 'caDepth', 'charge',
            'psi', 'phi', 'omega', 'chi', 'angleNan', 'chiNan',
            'aac', 'cc',
        ]
        for col in numeric_defaults:
            if col in profile_dat.columns:
                profile_dat[col] = profile_dat[col].fillna(0)

        cdr_len_columns = ['H1_len', 'H2_len', 'H3_len', 'L1_len', 'L2_len', 'L3_len']
        # Some metadata files (e.g. input/pdb_info_test3A.csv) do not include *_len columns.
        # In that case, derive lengths from the corresponding CDR sequence columns.
        if not set(cdr_len_columns).issubset(set(filter_pdb.columns)):
            seq_map = {
                'H1_len': 'H1_seq',
                'H2_len': 'H2_seq',
                'H3_len': 'H3_seq',
                'L1_len': 'L1_seq',
                'L2_len': 'L2_seq',
                'L3_len': 'L3_seq',
            }
            derived = {}
            for len_col, seq_col in seq_map.items():
                if len_col in filter_pdb.columns:
                    derived[len_col] = filter_pdb[len_col].values[0]
                    continue
                if seq_col in filter_pdb.columns:
                    raw = filter_pdb[seq_col].values[0]
                    seq = '' if pd.isna(raw) else str(raw).strip()
                    # Treat short all-'A' placeholders (e.g. 'AAAA') as missing.
                    if seq and set(seq.upper()) == {'A'} and len(seq) <= 6:
                        derived[len_col] = 0
                    else:
                        derived[len_col] = len(seq)
                else:
                    derived[len_col] = 0
            cdr_len_df = pd.DataFrame([derived], columns=cdr_len_columns)
        else:
            cdr_len_df = filter_pdb[cdr_len_columns]

        # Be defensive: coerce any non-numeric lengths to integers via pandas
        cdr_len_df = cdr_len_df.apply(pd.to_numeric, errors='coerce')
        cdr_len_df = cdr_len_df.fillna(0).astype(int)
        cdr_len_list = cdr_len_df.values.flatten().tolist()
        profile_dat[cdr_len_columns] = cdr_len_list
        
        cdr_score_columns = ['H3_score','L1_score']
        # Some metadata files do not include score columns. Default to 0.0 in that case.
        if not set(cdr_score_columns).issubset(set(filter_pdb.columns)):
            cdr_score_df = pd.DataFrame([{c: 0.0 for c in cdr_score_columns}], columns=cdr_score_columns)
        else:
            # Handle edge cases like 'nil', 'NA', empty strings etc. by coercing to 0.0
            cdr_score_df = filter_pdb[cdr_score_columns].replace(
                ['nil', 'NA', 'NaN', 'nan', ''], 0.0
            )
        cdr_score_df = cdr_score_df.apply(pd.to_numeric, errors='coerce').fillna(0.0)
        cdr_scores = cdr_score_df.values.flatten().tolist()
        profile_dat[cdr_score_columns] = cdr_scores
        # # merge interface
        # if logging.process_relaxed:
        #     interface_dat = pd.read_parquet(os.path.join(data_raw_path, 'interface', 'interface_result.parquet'))
        #     # interface_linear_dat = pd.read_parquet(os.path.join(data_raw_path, 'interface', 'interface_linear_result.parquet'))
        #     interface_dat.loc[:,'pdbId'] = pdb_id
        #     # interface_linear_dat.loc[:,'pdbId'] = pdb_id
        # elif logging.process_alphafold:
        #     interface_dat = pd.read_parquet(os.path.join(data_path, 'interface', 'interface_result.parquet'))
        #     # interface_linear_dat = pd.read_parquet(os.path.join(data_raw_path, 'interface', 'interface_linear_result.parquet'))
        #     interface_dat.loc[:,'pdbId'] = pdb_id
        # else:
        #     interface_dat = pd.read_parquet(os.path.join(data_path, 'interface', 'interface_result.parquet'))
        #     # interface_linear_dat = pd.read_parquet(os.path.join(data_path, 'interface', 'interface_linear_result.parquet'))
        # ellipro_pi_dat = merge_interface(interface_dat, profile_dat)
        # # ellipro_linear_dat = merge_interface(interface_linear_dat, profile_dat)
        # chainType
        chain_type_dat = filter_pdb[['pdbID', 'antigen']].melt(id_vars=['pdbID'], value_vars = ['antigen'],
                                                                var_name='chainType', value_name = 'chainId')

        chain_type_dat = chain_type_dat.rename(columns = {'pdbID':'pdbId'})
        profile_dat = profile_dat.merge(chain_type_dat, on=['pdbId','chainId'], how='left')
        # profile_dat.to_parquet(os.path.join(data_path, 'pdb_profile.parquet'))
        # ellipro_pi_dat = ellipro_pi_dat.merge(chain_type_dat, how='left')
        # ellipro_linear_dat = ellipro_linear_dat.merge(chain_type_dat, how='left')
        # ellipro_pi_dat.to_parquet(os.path.join(data_path, 'pdb_profile.parquet'))
        # ellipro_linear_dat.to_parquet(os.path.join(data_path, 'pdb_linear_profile.parquet'))
        
        # if (profile_dat.isnull().values.any()) | (ellipro_linear_dat.isnull().values.any()) | (ellipro_pi_dat.isnull().values.any()):
        # if (profile_dat.isnull().values.any()) | (ellipro_pi_dat.isnull().values.any()):
        #     logging.error_gather.append(pdb_id)
        if profile_dat.isnull().values.any():
            logging.error_gather.append(pdb_id)
        profile_dat.to_parquet(os.path.join(data_path, 'pdb_profile.parquet'))

    if not logging.error_gather:
        logging.message += '''
All the pdbs feature have been gathered successfully.'''
    else:
        logging.message += f'''
Not gathered pdb(s): {logging.error_gather}'''
        
def merge_interface(interface_dat, profile_dat):
    interface_dat['resId'] = interface_dat['resId'].astype(int)
    merge_dat = profile_dat.merge(interface_dat, how='left')
    merge_dat['isInterface'] = merge_dat['isInterface'].fillna(0).astype(int)
    return merge_dat
