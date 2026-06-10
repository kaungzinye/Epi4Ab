#!/usr/bin/env python3
"""
Refresh node_feature.parquet with CDR distance features.

Steps:
  1. extract_cdr_distances — reads full CIF, writes cdr_distances/cdr_dist_result.parquet
  2. gather_feature        — re-merges all features into pdb_profile.parquet
  3. Refresh node_feature.parquet in nodes_edges dir (no PyMOL re-run)

Usage:
  python scripts/refresh_cdr_features.py \\
      --metadata  <gated_metadata.csv> \\
      --processed <processed_data_dir> \\
      --nodes_edges <nodes_edges_dir>
"""

import argparse
import os
import sys
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Make preprocess modules importable
PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ / 'preprocess'))

from scripts.extract_cdr_distances import extract_cdr_distances
from scripts.gather_feature import gather_feature


class _SimpleLogging:
    """Minimal logging stub compatible with extract_cdr_distances and gather_feature."""
    def __init__(self, directory_data: str):
        self.directory_data = directory_data
        self.autodetect_antigen_chain = False
        self.error_extract_structure = []
        self.error_extract_cdrs = []
        self.error_gather = []
        self.message = ''

    def log_step_error(self, pdb_id, step, exc):
        err_dir = os.path.join(self.directory_data, pdb_id, 'errors')
        os.makedirs(err_dir, exist_ok=True)
        with open(os.path.join(err_dir, f'{step}.log'), 'a') as f:
            f.write(f'{type(exc).__name__}: {exc}\n')


def refresh_node_features(pdb_ids, processed_dir: str, nodes_edges_dir: str):
    """Regenerate node_feature.parquet from updated pdb_profile.parquet (no PyMOL)."""
    DROP_COLS = ['pdbId', 'resName', 'chainId', 'chainType']
    errors = []
    for pdb_id in tqdm(pdb_ids, desc='Refresh node_feature.parquet', unit='pdb'):
        profile_path = os.path.join(processed_dir, pdb_id, 'pdb_profile.parquet')
        node_feat_path = os.path.join(nodes_edges_dir, pdb_id, 'node_feature.parquet')

        if not os.path.exists(profile_path):
            print(f'[{pdb_id}] pdb_profile.parquet missing — skip node_feature refresh')
            errors.append(pdb_id)
            continue
        if not os.path.exists(os.path.dirname(node_feat_path)):
            print(f'[{pdb_id}] nodes_edges dir missing — skip node_feature refresh')
            errors.append(pdb_id)
            continue

        try:
            df = pd.read_parquet(profile_path)
            df_ag = df[df['pdbId'] == pdb_id]
            drop = [c for c in DROP_COLS if c in df_ag.columns]
            df_feat = df_ag.drop(columns=drop)
            df_feat.to_parquet(node_feat_path, engine='fastparquet', index=False)
        except Exception as e:
            print(f'[{pdb_id}] node_feature refresh error: {e}')
            errors.append(pdb_id)

    return errors


def main():
    parser = argparse.ArgumentParser(description='Refresh CDR distance features in node_feature.parquet')
    parser.add_argument('--metadata',    required=True, help='Gated metadata CSV (metadata.ok.fill_edge.csv)')
    parser.add_argument('--processed',   required=True, help='processed_data directory')
    parser.add_argument('--nodes_edges', required=True, help='nodes_edges directory')
    args = parser.parse_args()

    meta_df = pd.read_csv(args.metadata)
    pdb_ids = meta_df['pdbID'].tolist()
    print(f'Processing {len(pdb_ids)} PDBs')

    logging = _SimpleLogging(args.processed)

    print('\n=== Step 1: extract_cdr_distances ===')
    extract_cdr_distances(meta_df, logging)
    if logging.error_extract_cdrs:
        print(f'CDR distance errors: {logging.error_extract_cdrs}')

    print('\n=== Step 2: gather_feature (rebuild pdb_profile.parquet) ===')
    gather_feature(meta_df, logging)
    if logging.error_gather:
        print(f'gather_feature errors: {logging.error_gather}')

    print('\n=== Step 3: refresh node_feature.parquet ===')
    errors = refresh_node_features(pdb_ids, args.processed, args.nodes_edges)
    if errors:
        print(f'node_feature refresh errors: {errors}')

    print('\nDone.')


if __name__ == '__main__':
    main()
