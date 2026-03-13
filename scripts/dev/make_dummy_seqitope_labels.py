#!/usr/bin/env python3
"""
Dev-only helper to generate example Seqitope label files per PDB.
Writes node_label_seqitope.parquet with columns: resId:int, score:float[0,1].
If pandas/pyarrow are unavailable, falls back to CSV next to the parquet path.
Usage:
  python scripts/dev/make_dummy_seqitope_labels.py \
    --nodes_edges_dir <OUT_BASE>/nodes_edges \
    --pdb_id <pdb_id> [--seed 42]
"""
import argparse, os, random
try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nodes_edges_dir', required=True)
    ap.add_argument('--pdb_id', required=True)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    rnd = random.Random(args.seed)
    pdb_dir = os.path.join(args.nodes_edges_dir, args.pdb_id)
    nf = os.path.join(pdb_dir, 'node_feature.parquet')
    if pd is None:
        raise SystemExit('pandas not available; run inside project venv to generate example parquet')
    try:
        df = pd.read_parquet(nf)
    except Exception as e:
        raise SystemExit(f'cannot read {nf}: {e}')
    if 'resId' not in df.columns:
        raise SystemExit('node_feature.parquet missing resId column')
    out_path = os.path.join(pdb_dir, 'node_label_seqitope.parquet')
    scores = [min(1.0, max(0.0, rnd.random()*0.2 + (0.6 if (i%50==0) else 0.2))) for i in range(len(df))]
    out = pd.DataFrame({'resId': df['resId'].astype(int), 'score': scores})
    try:
        out.to_parquet(out_path, index=False)
        print('wrote', out_path)
    except Exception as e:
        # fallback CSV
        csv_path = out_path.replace('.parquet', '.csv')
        out.to_csv(csv_path, index=False)
        print('parquet failed; wrote CSV', csv_path, 'error:', e)

if __name__ == '__main__':
    main()
