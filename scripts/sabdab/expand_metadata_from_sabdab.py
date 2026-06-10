#!/usr/bin/env python3
"""
Build metadata CSV for SAbDab expansion (Phase 1.5 Track 1).

Inputs:
  --sabdab_summary  path to sabdab_summary.tsv (must include PDB code, chain annotations, resolution)
  --out_csv         output CSV with columns: pdbID, antigen (comma-separated chain IDs)
  --max_resolution  max Å (default 3.5)

Logic (offline-safe, heuristic):
  - Keep entries with resolution <= max_resolution (if available)
  - Identify antibody chains (H/L) by SAbDab annotations; antigen = non-H/L protein chains
  - Drop rows without antigen chains
  - Deduplicate by pdbID+antigen set

This file is a wrapper; adapt parsing to your sabdab_summary.tsv column names if they differ.
"""
import argparse, csv
import pandas as pd

DEFAULT_PDB_COLS = ['pdb','pdb_id','pdbID']
DEFAULT_RES_COLS = ['resolution','res','resol']
DEFAULT_CHAIN_ANN = ['chain_type','chainClass','chain_class']
DEFAULT_CHAIN_ID  = ['chain','chain_id','auth_chain_id']

AB_CLASSES = {'H','L','HEAVY','LIGHT','heavy','light'}


def _find_col(df, cands):
    for c in cands:
        if c in df.columns:
            return c
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sabdab_summary', required=True)
    ap.add_argument('--out_csv', required=True)
    ap.add_argument('--max_resolution', type=float, default=3.5)
    args = ap.parse_args()

    df = pd.read_csv(args.sabdab_summary, sep='\t', dtype=str)
    # Try to coerce resolution
    res_col = _find_col(df, DEFAULT_RES_COLS)
    if res_col and df[res_col].notna().any():
        try:
            df['_res_'] = pd.to_numeric(df[res_col], errors='coerce')
            df = df[df['_res_'].isna() | (df['_res_'] <= args.max_resolution)]
        except Exception:
            pass

    pdb_col = _find_col(df, DEFAULT_PDB_COLS) or df.columns[0]
    class_col = _find_col(df, DEFAULT_CHAIN_ANN)
    chain_col = _find_col(df, DEFAULT_CHAIN_ID)

    if class_col is None or chain_col is None:
        raise SystemExit('Cannot identify chain class/id columns in sabdab_summary.tsv')

    g = df.groupby(pdb_col)
    rows = []
    for pdb_id, sub in g:
        ab_chains = set(sub.loc[sub[class_col].isin(AB_CLASSES), chain_col].dropna().astype(str))
        all_chains = set(sub[chain_col].dropna().astype(str))
        ag_chains = sorted(list(all_chains - ab_chains))
        if not ag_chains:
            continue
        rows.append({'pdbID': pdb_id, 'antigen': ','.join(ag_chains)})

    out = pd.DataFrame(rows).drop_duplicates(subset=['pdbID','antigen'])
    out.to_csv(args.out_csv, index=False)
    print(f'Wrote {args.out_csv} with {len(out)} rows')

if __name__ == '__main__':
    main()
