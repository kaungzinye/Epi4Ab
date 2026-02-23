#!/usr/bin/env python3
""" 
End-to-end Phase 1 target generation.

For each row in a manifest CSV:
1) Run ProteinMPNN in conditional_probs_only mode (per-position log-probs)
2) Compute per-residue negative log-probability for the native sequence at designed positions
3) Map those designed positions to PDB residue IDs for the specified chain and write TSV (resId, score)
4) Align to node_feature.parquet and write proteinmpnn_scores.parquet

Manifest CSV columns:
  - pdb_id (nodes_edges folder name)
  - pdb_path (path to PDB file)
  - chain_id (antigen chain ID; required)

Notes:
- resId alignment uses PDB residue numbering (Bio.PDB res.id[1]).
- ProteinMPNN score_only produces an .npz; we locate it automatically.
"""

import argparse
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.PDB.PDBParser import PDBParser
from Bio import pairwise2


AA_3_TO_1 = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}


def extract_chain_resid_seq(pdb_path: str, chain_id: str):
    """Return (resId_list, resShort_list) for a chain from PDB."""
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure('p', pdb_path)
    model = next(struct.get_models())
    if chain_id not in model:
        raise ValueError(f"Chain {chain_id} not found in {pdb_path}")
    chain = model[chain_id]
    resids = []
    seq = []
    for res in chain:
        hetflag, resid, icode = res.id
        if hetflag.strip():
            continue
        aa = AA_3_TO_1.get(res.get_resname(), 'X')
        resids.append(int(resid))
        seq.append(aa)
    return resids, seq


def get_all_chain_sequences(pdb_path: str):
    """Return dict chain_id -> (resIds, seq) for all chains."""
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure('p', pdb_path)
    model = next(struct.get_models())
    out = {}
    for chain in model:
        cid = chain.id
        try:
            resids, seq = extract_chain_resid_seq(pdb_path, cid)
        except Exception:
            continue
        if len(seq) > 0:
            out[cid] = (resids, seq)
    return out


def best_chain_match(node_seq, chains: dict):
    """Pick chain with best alignment identity+coverage."""
    best = None
    node = ''.join(node_seq)
    for cid, (_resids, seq) in chains.items():
        aln = pairwise2.align.globalms(node, ''.join(seq), 2, -1, -2, -0.5, one_alignment_only=True)
        if not aln:
            continue
        a, b, score, start, end = aln[0]
        matches = sum((x == y) and (x != '-') for x, y in zip(a, b))
        aligned = sum((x != '-') and (y != '-') for x, y in zip(a, b))
        coverage = aligned / max(1, len(node_seq))
        identity = matches / max(1, aligned)
        key = (coverage, identity, score)
        if best is None or key > best['key']:
            best = {'chain_id': cid, 'coverage': coverage, 'identity': identity, 'score': score, 'key': key}
    return best


def map_scores_by_alignment(node_seq, pdb_seq, pdb_scores):
    """Map per-residue scores from pdb_seq onto node_seq by alignment indices."""
    aln = pairwise2.align.globalms(''.join(node_seq), ''.join(pdb_seq), 2, -1, -2, -0.5, one_alignment_only=True)
    if not aln:
        raise RuntimeError('Alignment failed')
    a, b, score, start, end = aln[0]
    pdb_i = 0
    mapped = []
    matches = 0
    aligned = 0
    for x, y in zip(a, b):
        if x != '-':
            if y != '-':
                aligned += 1
                if x == y:
                    matches += 1
                mapped.append(float(pdb_scores[pdb_i]))
                pdb_i += 1
            else:
                mapped.append(None)
        else:
            if y != '-':
                pdb_i += 1
    coverage = sum(v is not None for v in mapped) / max(1, len(node_seq))
    identity = matches / max(1, aligned)
    return mapped, coverage, identity


def find_single_npz(out_folder: Path, subdir: str) -> Path:
    d = out_folder / subdir
    if not d.exists():
        raise FileNotFoundError(f"Missing ProteinMPNN output dir: {d}")
    npzs = sorted(d.glob('*.npz'))
    if len(npzs) != 1:
        raise RuntimeError(f"Expected 1 npz in {d}, found {len(npzs)}")
    return npzs[0]


def conditional_npz_to_per_res_scores(npz_path: Path) -> np.ndarray:
    """Return per-residue NLL scores for designed positions."""
    data = np.load(npz_path, allow_pickle=True)
    log_p = data['log_p']  # [B, L, 21]
    S = data['S']          # [L]
    design_mask = data['design_mask']  # [L]

    if log_p.ndim != 3:
        raise RuntimeError(f"Unexpected log_p shape: {log_p.shape}")
    if S.ndim != 1:
        raise RuntimeError(f"Unexpected S shape: {S.shape}")

    log_p0 = log_p[0]
    idx = np.arange(log_p0.shape[0])
    native_logp = log_p0[idx, S]
    nll = (-native_logp).astype(np.float32)
    # Only designed positions correspond to chain(s) specified by pdb_path_chains
    mask = design_mask.astype(bool)
    return nll[mask]


def main():
    parser = argparse.ArgumentParser(description='Run ProteinMPNN score_only and prepare targets')
    parser.add_argument('--manifest', required=True, help='CSV with pdb_id,pdb_path,chain_id')
    parser.add_argument('--nodes_edges_dir', required=True, help='nodes_edges directory containing <pdb_id>/node_feature.parquet')
    parser.add_argument('--proteinmpnn_run', required=True, help='Path to protein_mpnn_run.py')
    parser.add_argument('--out_root', required=True, help='Output root (will create per-pdb folders here)')
    parser.add_argument('--target_file', default='proteinmpnn_scores.parquet', help='Output parquet name under nodes_edges/<pdb_id>/')
    parser.add_argument('--fail_fast', action='store_true', help='Stop on first per-complex error (default: continue and report at end)')
    parser.add_argument('--skip_existing', action='store_true', help='Skip rows where target parquet already exists in nodes_edges/<pdb_id>/')
    args = parser.parse_args()

    df = pd.read_csv(args.manifest, comment='#')
    # Deduplicate by pdb_id (keep last)
    if 'pdb_id' in df.columns:
        df = df.drop_duplicates(subset=['pdb_id'], keep='last')
    for col in ['pdb_id', 'pdb_path']:
        if col not in df.columns:
            raise KeyError(f"Manifest missing column: {col}")

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    nodes_edges_dir = Path(args.nodes_edges_dir)

    failed = []  # (pdb_id, error_message)

    for _, row in df.iterrows():
        pdb_id = str(row['pdb_id']).strip()
        pdb_path = str(row['pdb_path']).strip()
        chain_id = str(row['chain_id']).strip() if 'chain_id' in row and pd.notna(row['chain_id']) else ''

        if not pdb_id or pdb_id.startswith('#'):
            continue
        if not pdb_path or pdb_path == 'nan':
            print(f"[SKIP] {pdb_id}: pdb_path empty")
            continue
        # chain_id is optional; if empty or incorrect we'll auto-detect from node_feature sequence
        if not os.path.exists(pdb_path):
            print(f"[SKIP] {pdb_id}: PDB not found: {pdb_path}")
            continue

        out_parquet = nodes_edges_dir / pdb_id / args.target_file
        if args.skip_existing and out_parquet.exists():
            print(f"[SKIP] {pdb_id}: already exists {out_parquet}")
            continue

        try:
            out_folder = out_root / pdb_id
            out_folder.mkdir(parents=True, exist_ok=True)

            # Load node_feature (required for chain detection and final alignment)
            pdb_nodes_dir = nodes_edges_dir / pdb_id
            node_feature_path = pdb_nodes_dir / 'node_feature.parquet'
            if not node_feature_path.exists():
                raise FileNotFoundError(f"Missing node_feature: {node_feature_path}")
            df_feat_full = pd.read_parquet(node_feature_path)
            if 'resId' not in df_feat_full.columns or 'resShort' not in df_feat_full.columns:
                raise KeyError(f"node_feature must include resId and resShort for {pdb_id}")
            node_resids = df_feat_full['resId'].astype(int).tolist()
            node_seq = df_feat_full['resShort'].astype(str).tolist()

            # Choose best chain by alignment if chain_id missing or not matching
            chains = get_all_chain_sequences(pdb_path)
            chosen = None
            if chain_id and chain_id in chains:
                chosen = best_chain_match(node_seq, {chain_id: chains[chain_id]})
                if chosen and chosen['coverage'] < 0.5:
                    chosen = None
            if chosen is None:
                chosen = best_chain_match(node_seq, chains)
                if chosen is None:
                    raise RuntimeError(f"Could not auto-detect chain for {pdb_id}")
                chain_id = chosen['chain_id']
            print(f"[CHAIN] {pdb_id}: using chain {chain_id} (coverage={chosen['coverage']:.3f}, identity={chosen['identity']:.3f})")

            # Run ProteinMPNN on selected chain
            cmd = [
                'python', args.proteinmpnn_run,
                '--conditional_probs_only', '1',
                '--num_seq_per_target', '1',
                '--batch_size', '1',
                '--suppress_print', '1',
                '--out_folder', str(out_folder) + '/',
                '--pdb_path', pdb_path,
                '--pdb_path_chains', chain_id,
            ]
            print('[RUN]', ' '.join(cmd))
            subprocess.run(cmd, check=True)

            npz_path = find_single_npz(out_folder, 'conditional_probs_only')
            pdb_scores = conditional_npz_to_per_res_scores(npz_path)

            pdb_resids, pdb_seq = extract_chain_resid_seq(pdb_path, chain_id)
            if len(pdb_scores) != len(pdb_resids):
                raise RuntimeError(f"ProteinMPNN output length mismatch: scores={len(pdb_scores)} residues={len(pdb_resids)}")

            # Tier 1: merge by resId if possible
            df_scores = pd.DataFrame({'resId': pdb_resids, 'score': pdb_scores})
            merged_direct = pd.DataFrame({'resId': node_resids}).merge(df_scores, on='resId', how='left')
            if not merged_direct['score'].isna().any() and len(merged_direct) == len(node_resids):
                mapped_scores = merged_direct['score'].tolist()
                map_mode = 'resId'
                cov = 1.0
                ident = 1.0
            else:
                # Tier 2: sequence-alignment mapping
                mapped_scores, cov, ident = map_scores_by_alignment(node_seq, pdb_seq, pdb_scores)
                if cov < 0.95:
                    missing = sum(v is None for v in mapped_scores)
                    raise RuntimeError(f"Alignment mapping low coverage: coverage={cov:.3f} missing={missing}")
                fill = float(np.nanmean(pdb_scores))
                mapped_scores = [float(v) if v is not None else fill for v in mapped_scores]
                map_mode = 'alignment'

            # Write TSV aligned to node_feature (graph) residues
            tsv_path = out_folder / f"{pdb_id}_proteinmpnn.tsv"
            pd.DataFrame({'resId': node_resids, 'score': mapped_scores}).to_csv(tsv_path, sep='\t', index=False)
            print(f"[WROTE] {tsv_path} (mode={map_mode}, coverage={cov:.3f}, identity={ident:.3f})")

            # Write parquet target aligned to node_feature
            out_parquet = pdb_nodes_dir / args.target_file
            pd.DataFrame({'resId': node_resids, 'score': mapped_scores}).to_parquet(out_parquet, index=False)
            print(f"[WROTE] {out_parquet}")

        except Exception as e:
            msg = str(e)
            print(f"[FAIL] {pdb_id}: {msg}", flush=True)
            failed.append((pdb_id, msg))
            if args.fail_fast:
                raise

    if failed:
        print("\n========== FAILED ==========")
        for pdb_id, msg in failed:
            print(f"  {pdb_id}: {msg}")
        print(f"Total failed: {len(failed)}")
        raise SystemExit(1)
    print("\nAll complexes processed successfully.")


if __name__ == '__main__':
    main()
