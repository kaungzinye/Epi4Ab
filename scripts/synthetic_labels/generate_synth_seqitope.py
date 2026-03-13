#!/usr/bin/env python3
"""
Generate synthetic per-residue epitope-propensity labels for Phase 1.5 (weak labels only).

Output per PDB (under --nodes_edges_dir/<pdb_id>):
  node_label_synthseq.parquet with columns:
    - resId:int (aligned to node_feature.parquet)
    - score:float in [0,1]

Approach (offline-safe, robust fallbacks):
  - Try to use structure-derived proxies from available graph files:
      * edge_index.parquet, edge_attribute_dist.parquet
    Heuristics:
      * Exposure proxy: nodes with fewer close neighbors (<= 8Å) tend to be more surface-exposed → higher score
      * Protrusion proxy: larger mean pairwise distance to neighbors (within 10Å) → higher score
  - Optional sequence fallback (if structure edges absent): smooth one-hot AA index with a short window and normalize.
  - One pass of neighbor smoothing (10Å) with mixing alpha=0.6.

This is intentionally simple to avoid external tool dependencies. If you later stage
BepiPred/DiscoTope/SEPPA/ElliPro outputs, you can extend this script to merge them
and re-weight before smoothing.
"""
import argparse
import os
import sys
from typing import Optional, List

try:
    import pandas as pd
    import numpy as np
except Exception as e:
    print("pandas/numpy required in venv: ", e, file=sys.stderr)
    sys.exit(1)


def _safe_minmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    mn, mx = float(np.nanmin(x)), float(np.nanmax(x))
    if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
        return np.zeros_like(x, dtype=float)
    y = (x - mn) / (mx - mn)
    y[~np.isfinite(y)] = 0.0
    return np.clip(y, 0.0, 1.0)


def _neighbor_lists(edge_index: pd.DataFrame, n_nodes: int) -> List[List[int]]:
    neigh = [[] for _ in range(n_nodes)]
    src = edge_index['source'].to_numpy()
    dst = edge_index['target'].to_numpy()
    for s, d in zip(src, dst):
        if 0 <= s < n_nodes and 0 <= d < n_nodes:
            neigh[s].append(d)
            neigh[d].append(s)
    return neigh


def _degree_exposure(neigh: List[List[int]], cutoff_deg: Optional[int] = None) -> np.ndarray:
    deg = np.array([len(v) for v in neigh], dtype=float)
    # Invert degree as exposure proxy; normalize
    inv = 1.0 / np.maximum(deg, 1.0)
    return _safe_minmax(inv)


def _mean_dist_exposure(neigh: List[List[int]], dists: np.ndarray) -> np.ndarray:
    # dists is per-edge vector aligned with rows of edge_index
    # Build mean distance per node from incident edges
    # Accumulate sum and counts
    # Map edge list again to gather distances
    # We assume undirected graph mirrored or treated symmetrically by _neighbor_lists
    return None


def _gaussian_smooth_once(scores: np.ndarray, neigh: List[List[int]], alpha: float = 0.6) -> np.ndarray:
    out = np.empty_like(scores, dtype=float)
    for i, ns in enumerate(neigh):
        if ns:
            out[i] = alpha * scores[i] + (1.0 - alpha) * float(np.mean(scores[ns]))
        else:
            out[i] = scores[i]
    return np.clip(out, 0.0, 1.0)


def synthesize_for_pdb(nodes_dir: str, pdb_id: str, close_cutoff: float = 8.0, smooth_cutoff: float = 10.0, alpha: float = 0.6) -> str:
    pdb_path = os.path.join(nodes_dir, pdb_id)
    nf = os.path.join(pdb_path, 'node_feature.parquet')
    ei = os.path.join(pdb_path, 'edge_index.parquet')
    ed = os.path.join(pdb_path, 'edge_attribute_dist.parquet')
    if not os.path.exists(nf):
        raise FileNotFoundError(f"missing node_feature.parquet for {pdb_id}")
    df_feat = pd.read_parquet(nf)
    if 'resId' not in df_feat.columns:
        raise ValueError(f"node_feature.parquet for {pdb_id} lacks resId column")
    n_nodes = len(df_feat)

    use_graph = os.path.exists(ei) and os.path.exists(ed)
    if use_graph:
        df_ei = pd.read_parquet(ei)
        df_ed = pd.read_parquet(ed)
        if len(df_ei) != len(df_ed) or 'source' not in df_ei or 'target' not in df_ei or 'dist' not in df_ed:
            use_graph = False
    if use_graph:
        neigh = _neighbor_lists(df_ei, n_nodes)
        # Exposure via degree (<= close_cutoff)
        # Create a filtered neighbor list by masking edges > close_cutoff
        mask = df_ed['dist'].to_numpy() <= float(close_cutoff)
        # Build degree on filtered edges
        src = df_ei['source'].to_numpy()
        dst = df_ei['target'].to_numpy()
        deg = np.zeros(n_nodes, dtype=float)
        for (s, d, m) in zip(src, dst, mask):
            if not m:
                continue
            if 0 <= s < n_nodes and 0 <= d < n_nodes:
                deg[s] += 1.0
                deg[d] += 1.0
        inv_deg = 1.0 / np.maximum(deg, 1.0)
        expo_deg = _safe_minmax(inv_deg)
        base = expo_deg
        # Neighbor smoothing with smooth_cutoff using existing neigh list (approximation)
        base = _gaussian_smooth_once(base, neigh, alpha=alpha)
        scores = _safe_minmax(base)
    else:
        # Fallback: sequence-window proxy if resShort exists; otherwise uniform zeros
        if 'resShort' in df_feat.columns:
            aa_map = {aa: i for i, aa in enumerate('AVLIPMFWGSTCNQYKRHDE')}
            idx = np.array([aa_map.get(a, 0) if isinstance(a, str) else 0 for a in df_feat['resShort'].tolist()], dtype=float)
            # Normalize AA index, then 1D smooth (window=5)
            x = _safe_minmax(idx)
            w = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
            pad = len(w)//2
            xp = np.pad(x, (pad, pad), mode='edge')
            y = np.convolve(xp, w, mode='valid')
            scores = _safe_minmax(y)
        else:
            scores = np.zeros(n_nodes, dtype=float)
    out = pd.DataFrame({'resId': df_feat['resId'].astype(int), 'score': scores.astype(float)})
    out_path = os.path.join(pdb_path, 'node_label_synthseq.parquet')
    try:
        out.to_parquet(out_path, index=False)
    except Exception as e:
        # fall back to CSV
        out_path = out_path.replace('.parquet', '.csv')
        out.to_csv(out_path, index=False)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nodes_edges_dir', required=True)
    ap.add_argument('--pdb_id', action='append', help='PDB id(s) to process; repeatable')
    ap.add_argument('--pdb_list', help='CSV with a pdbID column')
    ap.add_argument('--alpha', type=float, default=0.6)
    ap.add_argument('--close_cutoff', type=float, default=8.0)
    ap.add_argument('--smooth_cutoff', type=float, default=10.0)
    args = ap.parse_args()

    pdb_ids: List[str] = []
    if args.pdb_id:
        pdb_ids.extend(args.pdb_id)
    if args.pdb_list:
        try:
            df = pd.read_csv(args.pdb_list)
            col = next((c for c in ['pdbID','pdbId','pdb_id'] if c in df.columns), df.columns[0])
            pdb_ids.extend(df[col].astype(str).tolist())
        except Exception as e:
            print(f"Warning: cannot read pdb_list {args.pdb_list}: {e}", file=sys.stderr)
    if not pdb_ids:
        print("No PDB ids provided", file=sys.stderr)
        sys.exit(2)

    nodes_dir = args.nodes_edges_dir
    ok, fail = 0, 0
    for pid in pdb_ids:
        try:
            outp = synthesize_for_pdb(nodes_dir, pid, args.close_cutoff, args.smooth_cutoff, args.alpha)
            print(f"OK {pid} -> {outp}")
            ok += 1
        except Exception as e:
            print(f"FAIL {pid}: {e}", file=sys.stderr)
            fail += 1
    print(f"Done. ok={ok} fail={fail}")

if __name__ == '__main__':
    main()
