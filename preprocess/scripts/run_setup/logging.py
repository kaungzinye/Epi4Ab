import os
import os
from datetime import date


class DataLogging():
    def __init__(self, args):
        self.directory_metadata = args.directory_metadata
        self.directory_data = args.directory_output

        self.download_pdb = args.download_pdb
        self.autodetect_antigen_chain = getattr(args, 'autodetect_antigen_chain', False)

        if not os.path.exists(self.directory_data):
            os.mkdir(self.directory_data)

        self.run_date = date.today()

        # error
        self.error_download = []
        self.error_extract_structure = []
        self.error_extract_cdrs = []
        self.error_pdb2pqr = []
        self.error_sasa = []
        self.error_depth = []
        self.error_angle = []
        self.error_charge = []
        self.error_charge_compostion = []
        self.error_interface = []
        self.error_ellipro = []
        self.error_gather = []
        self.error_empty_interface = []
        self.error_sphere = []

        self.total_time = None
        self.message = f'''
Run date: {self.run_date}'''

    def save_log(self):
        self.message += f'''
Total runtime: {self.total_time}'''
        print(self.message)
        # with open(os.path.join(self.directory_output, 'error.txt'), 'a') as f:
        #     f.write(self.message)

    def log_step_error(self, pdb_id: str, step: str, exc: Exception):
        """Persist a per-PDB error log for the given step.

        This is intentionally lightweight so batch jobs can continue and
        the pipeline can quarantine failed PDBs later via gating.
        """
        try:
            err_dir = os.path.join(self.directory_data, pdb_id, 'errors')
            os.makedirs(err_dir, exist_ok=True)
            path = os.path.join(err_dir, f'{step}.log')
            with open(path, 'a', encoding='utf-8') as f:
                f.write(f'[{date.today()}] {type(exc).__name__}: {exc}\n')
        except Exception:
            # Never fail the pipeline due to logging.
            pass
