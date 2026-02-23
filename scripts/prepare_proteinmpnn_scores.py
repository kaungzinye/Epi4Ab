#!/usr/bin/env python3
"""
Prepare ProteinMPNN per-residue scores aligned to node_feature.parquet.

Expected input: a CSV/TSV with at least resId and score columns.
Optional: chain column for multi-chain inputs.

Output: proteinmpnn_scores.parquet with columns [resId, score]
written into nodes_edges/<pdb_id>/
"""

import argparse
import os
import pandas as pd


def load_scores(path: str, res_id_col: str, score_col: str, chain_col: str = None) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Scores file not found: {path}")
    sep = '\t' if path.endswith('.tsv') or path.endswith('.txt') else ','
    df = pd.read_csv(path, sep=sep)
    missing = [col for col in [res_id_col, score_col] if col not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in scores file: {missing}")
    df = df[[res_id_col, score_col] + ([chain_col] if chain_col else [])].copy()
    df.rename(columns={res_id_col: 'resId', score_col: 'score'}, inplace=True)
    df['resId'] = pd.to_numeric(df['resId'], errors='raise').astype(int)
    if chain_col:
        df.rename(columns={chain_col: 'chainId'}, inplace=True)
    return df


def main():
    parser = argparse.ArgumentParser(description="Align ProteinMPNN scores to node_feature.parquet")
    parser.add_argument('--pdb_id', required=True, help='PDB ID (folder name in nodes_edges)')
    parser.add_argument('--nodes_edges_dir', required=True, help='Path to nodes_edges directory')
    parser.add_argument('--scores_file', required=True, help='CSV/TSV with per-residue scores')
    parser.add_argument('--res_id_col', default='resId', help='Column name for residue id')
    parser.add_argument('--score_col', default='score', help='Column name for score')
    parser.add_argument('--chain_col', default=None, help='Optional column name for chain id')
    parser.add_argument('--chain_id', default=None, help='Optional chain id to filter')
    parser.add_argument('--output_file', default='proteinmpnn_scores.parquet', help='Output parquet filename')
    args = parser.parse_args()

    pdb_dir = os.path.join(args.nodes_edges_dir, args.pdb_id)
    node_feature_path = os.path.join(pdb_dir, 'node_feature.parquet')
    if not os.path.exists(node_feature_path):
        raise FileNotFoundError(f"node_feature.parquet not found: {node_feature_path}")

    df_features = pd.read_parquet(node_feature_path)
    if 'resId' not in df_features.columns:
        raise KeyError("node_feature.parquet missing resId column")
    if 'chainId' not in df_features.columns:
        raise KeyError("node_feature.parquet missing chainId column")

    if args.chain_id:
        df_features = df_features[df_features['chainId'] == args.chain_id]
    else:
        unique_chains = df_features['chainId'].dropna().unique().tolist()
        if len(unique_chains) > 1:
            raise ValueError(f"Multiple chains found in node_feature: {unique_chains}. Provide --chain_id.")

    df_scores = load_scores(args.scores_file, args.res_id_col, args.score_col, args.chain_col)
    if 'chainId' in df_scores.columns:
        if args.chain_id:
            df_scores = df_scores[df_scores['chainId'] == args.chain_id]
        elif 'chainId' in df_features.columns and df_features['chainId'].nunique() == 1:
            df_scores = df_scores[df_scores['chainId'] == df_features['chainId'].unique()[0]]

    merge_cols = ['resId'] + (['chainId'] if 'chainId' in df_scores.columns else [])
    df_merged = df_features[['resId', 'chainId']].merge(df_scores, on=merge_cols, how='left')
    if df_merged['score'].isna().any():
        missing = df_merged[df_merged['score'].isna()]['resId'].tolist()[:10]
        raise ValueError(f"Missing scores for {df_merged['score'].isna().sum()} residues. Examples: {missing}")

    output_path = os.path.join(pdb_dir, args.output_file)
    df_merged[['resId', 'score']].to_parquet(output_path, index=False)
    print(f"Wrote {output_path} ({len(df_merged)} rows)")


if __name__ == '__main__':
    main()
