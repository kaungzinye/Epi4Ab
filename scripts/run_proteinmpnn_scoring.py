#!/usr/bin/env python3
"""
Run ProteinMPNN scoring from a manifest and write per-PDB TSV outputs.

Manifest CSV columns (required):
  - pdb_id
  - pdb_path
Optional:
  - chain_id

Command template supports placeholders:
  {pdb_path} {chain_id} {out_tsv}

Example:
  --cmd_template "python /path/to/proteinmpnn_run.py --pdb_path {pdb_path} --chain_id {chain_id} --out_path {out_tsv}"
"""

import argparse
import os
import subprocess
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Run ProteinMPNN scoring from manifest")
    parser.add_argument('--manifest', required=True, help='CSV manifest with pdb_id, pdb_path, chain_id')
    parser.add_argument('--output_dir', required=True, help='Directory to write per-PDB TSV outputs')
    parser.add_argument('--cmd_template', required=True, help='Command template with placeholders')
    args = parser.parse_args()

    if not os.path.exists(args.manifest):
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")
    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.manifest)
    for col in ['pdb_id', 'pdb_path']:
        if col not in df.columns:
            raise KeyError(f"Manifest missing required column: {col}")

    for _, row in df.iterrows():
        pdb_id = str(row['pdb_id'])
        pdb_path = str(row['pdb_path'])
        chain_id = str(row['chain_id']) if 'chain_id' in row and pd.notna(row['chain_id']) else ''
        out_tsv = os.path.join(args.output_dir, f"{pdb_id}_proteinmpnn.tsv")

        if not os.path.exists(pdb_path):
            raise FileNotFoundError(f"PDB not found: {pdb_path}")

        chain_id_arg = f"--chain_id {chain_id}" if chain_id else ""
        cmd = args.cmd_template.format(
            pdb_path=pdb_path,
            chain_id=chain_id,
            chain_id_arg=chain_id_arg,
            out_tsv=out_tsv
        )
        print(f"Running: {cmd}")
        subprocess.run(cmd, shell=True, check=True)


if __name__ == '__main__':
    main()
