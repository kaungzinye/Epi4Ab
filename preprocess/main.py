import pandas as pd
from datetime import datetime
from pathlib import Path
import pandas as pd
from scripts.run_setup.arguments import initiate_argument
from scripts.run_setup.logging import DataLogging
from scripts.download_pdb import download_pdb, create_folder
from scripts.extract_structure import filter_pdb
from scripts.run_pdb2pqr import run_pdb2pqr
from scripts.run_md_analysis import extract_angle
from scripts.extract_depth import extract_depth
from scripts.extract_charge import extract_charge
from scripts.extract_aac import extract_aac
from scripts.extract_charge_composition import extract_cc
from scripts.gather_feature import gather_feature
from scripts.extract_sequence import extract_sequence
from scripts.extract_cdr_distances import extract_cdr_distances

data_start_time = datetime.now()
args = initiate_argument()
logging = DataLogging(args)


meta_df = pd.read_csv(logging.directory_metadata)
create_folder(meta_df, logging)
if logging.download_pdb:
    download_pdb(meta_df, logging)
    filter_pdb(meta_df, logging)

# Ensure lig.pdb exists (chain extraction). This is required even when PDBs are pre-downloaded.
# If --autodetect_antigen_chain is enabled, we may need to re-extract lig.pdb from CIF.
if getattr(logging, 'autodetect_antigen_chain', False):
    filter_pdb(meta_df, logging)
else:
    # Run extraction only when lig.pdb is missing for any PDB in metadata
    needs = False
    for pdb_id in meta_df.pdbID.values:
        lig = Path(logging.directory_data) / pdb_id / 'lig.pdb'
        if not lig.exists():
            needs = True
            break
    if needs:
        filter_pdb(meta_df, logging)
run_pdb2pqr(meta_df, logging)
extract_angle(meta_df, logging)
extract_depth(meta_df, logging)
extract_charge(meta_df, logging)
extract_aac(meta_df, logging)
extract_cc(meta_df, logging)
extract_cdr_distances(meta_df, logging)
gather_feature(meta_df, logging)
extract_sequence(meta_df, logging)

# logging.total_time = datetime.now() - data_start_time
# logging.save_log()
