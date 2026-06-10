#!/usr/bin/env python3
"""Build a Phase 1 metadata CSV from sabdab_summary.tsv.

Creates an Epi4Ab-style metadata CSV suitable for preprocessing.

This script does NOT provide CDR sequences; it fills them as empty strings.
If you want AntiBERTy conditioning, you must source/derive CDR sequences.

Default filtering is conservative to reduce preprocessing failures:
- antigen_type == protein
- single-chain antigen (skip antigen_chain like "A | B")
- Hchain and Lchain present
- X-ray structures with resolution <= 3.0 A

Output columns match preprocessing expectations:
- pdbID, pdb, antigen, VH_fam, VL_fam, H1_seq..L3_seq

Usage:
  python scripts/build_phase1_metadata_from_sabdab.py \
    --sabdab_summary sabdab_summary.tsv \
    --out_csv input/metadata_sabdab_phase1.csv \
    --max_rows 5000
"""

import argparse
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sabdab_summary', required=True)
    ap.add_argument('--out_csv', required=True)
    ap.add_argument('--max_rows', type=int, default=0, help='0 means no limit')
    ap.add_argument('--method', default='X-RAY DIFFRACTION', help='Method filter (exact match)')
    ap.add_argument('--max_resolution', type=float, default=3.0)
    args = ap.parse_args()

    df = pd.read_csv(args.sabdab_summary, sep='\t')

    df = df[df['antigen_type'].astype(str).str.fullmatch('protein', na=False)]
    df = df[df['antigen_chain'].astype(str).str.contains('\\|', regex=True) == False]
    df = df[df['Hchain'].notna() & df['Lchain'].notna()]
    if args.method:
        df = df[df['method'].astype(str) == args.method]

    df['resolution_num'] = pd.to_numeric(df['resolution'], errors='coerce')
    df = df[df['resolution_num'].notna() & (df['resolution_num'] <= float(args.max_resolution))]

    out = pd.DataFrame({
        'pdb': df['pdb'].astype(str).str.lower(),
        'antigen': df['antigen_chain'].astype(str).str.strip(),
    })
    out['pdbID'] = (
        df['pdb'].astype(str).str.lower()
        + '_'
        + df['Hchain'].astype(str).str.strip()
        + df['Lchain'].astype(str).str.strip()
        + df['antigen_chain'].astype(str).str.strip()
    )
    out['VH_fam'] = ''
    out['VL_fam'] = ''
    for c in ['H1_seq','H2_seq','H3_seq','L1_seq','L2_seq','L3_seq']:
        out[c] = ''

    out = out[['pdbID','pdb','antigen','VH_fam','VL_fam','H1_seq','H2_seq','H3_seq','L1_seq','L2_seq','L3_seq']]
    if args.max_rows and args.max_rows > 0:
        out = out.head(int(args.max_rows))

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv} with {len(out)} rows")


if __name__ == '__main__':
    main()
