import argparse
import csv
import os
from dataclasses import dataclass
from datetime import datetime

import pandas as pd


@dataclass
class GateResult:
    pdb_id: str
    ok: bool
    stage: str
    missing: str
    error: str


def _now() -> str:
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def _safe_read_parquet(path: str, columns=None) -> bool:
    try:
        # Use a narrow read to validate file integrity.
        pd.read_parquet(path, columns=columns)
        return True
    except Exception:
        return False


def _columns_for_label(label: str):
    # Keep reads narrow for speed and to avoid loading large parquets.
    if label == 'pdb_profile.parquet':
        return ['resId', 'resShort']
    if label == 'depth_result.parquet':
        return ['resId']
    if label == 'angle_result.parquet':
        return ['resId']
    if label == 'charge_result.parquet':
        return ['resId']
    if label == 'aac_result.parquet':
        return ['resName', 'chainId']
    if label == 'cc_result.parquet':
        return ['resName', 'chainId']
    if label.startswith('edge_index') or label == 'edge_index.parquet':
        return ['source', 'target']
    if 'edge_attribute_dist' in label:
        return ['dist']
    if 'edge_attribute_charge' in label:
        return ['qi*qj']
    if label == 'node_feature.parquet':
        return ['resId', 'resShort']
    return None


def _required_paths_for_stage(pdb_id: str, stage: str, processed_dir: str, nodes_edges_dir: str):
    """Return list of (label, path, check_mode).

    check_mode:
      - 'exists' : os.path.exists
      - 'parquet': exists + parquet readable
    """
    if stage == 'preprocess' and not processed_dir:
        raise ValueError('processed_dir is required for stage=preprocess')
    if stage in ('nodes_edges', 'fill_edge') and not nodes_edges_dir:
        raise ValueError(f'nodes_edges_dir is required for stage={stage}')

    processed_pdb = os.path.join(processed_dir, pdb_id) if processed_dir else ''
    nodes_pdb = os.path.join(nodes_edges_dir, pdb_id) if nodes_edges_dir else ''

    if stage == 'preprocess':
        # Strict gate: require all upstream artifacts used by downstream graph building.
        return [
            ('lig.pdb', os.path.join(processed_pdb, 'lig.pdb'), 'exists'),
            ('pdb2pqr_result.pqr', os.path.join(processed_pdb, 'pdb2pqr', 'pdb2pqr_result.pqr'), 'exists'),
            ('depth_result.parquet', os.path.join(processed_pdb, 'depth', 'depth_result.parquet'), 'parquet'),
            ('angle_result.parquet', os.path.join(processed_pdb, 'angle', 'angle_result.parquet'), 'parquet'),
            ('charge_result.parquet', os.path.join(processed_pdb, 'charge', 'charge_result.parquet'), 'parquet'),
            ('aac_result.parquet', os.path.join(processed_pdb, 'aac', 'aac_result.parquet'), 'parquet'),
            ('cc_result.parquet', os.path.join(processed_pdb, 'charge_composition', 'cc_result.parquet'), 'parquet'),
            ('pdb_profile.parquet', os.path.join(processed_pdb, 'pdb_profile.parquet'), 'parquet'),
            ('antigen_sequence.json', os.path.join(processed_pdb, 'sequence', 'antigen_sequence.json'), 'exists'),
            ('cdr_sequence.json', os.path.join(processed_pdb, 'sequence', 'cdr_sequence.json'), 'exists'),
        ]

    if stage == 'nodes_edges':
        # Require node features + both CA/CB edges before fill_edge.
        return [
            ('node_feature.parquet', os.path.join(nodes_pdb, 'node_feature.parquet'), 'parquet'),
            ('edge_index_CB.parquet', os.path.join(nodes_pdb, 'edge_index_CB.parquet'), 'parquet'),
            ('edge_attribute_dist_CB.parquet', os.path.join(nodes_pdb, 'edge_attribute_dist_CB.parquet'), 'parquet'),
            ('edge_attribute_charge_CB.parquet', os.path.join(nodes_pdb, 'edge_attribute_charge_CB.parquet'), 'parquet'),
            ('edge_index_CA.parquet', os.path.join(nodes_pdb, 'edge_index_CA.parquet'), 'parquet'),
            ('edge_attribute_dist_CA.parquet', os.path.join(nodes_pdb, 'edge_attribute_dist_CA.parquet'), 'parquet'),
            ('edge_attribute_charge_CA.parquet', os.path.join(nodes_pdb, 'edge_attribute_charge_CA.parquet'), 'parquet'),
        ]

    if stage == 'fill_edge':
        return [
            ('edge_index.parquet', os.path.join(nodes_pdb, 'edge_index.parquet'), 'parquet'),
            ('edge_attribute_dist.parquet', os.path.join(nodes_pdb, 'edge_attribute_dist.parquet'), 'parquet'),
            ('edge_attribute_charge.parquet', os.path.join(nodes_pdb, 'edge_attribute_charge.parquet'), 'parquet'),
        ]

    raise ValueError(f"Unknown stage: {stage}")


def _check_paths(required):
    missing = []
    unreadable = []
    for label, path, mode in required:
        if not os.path.exists(path):
            missing.append(label)
            continue
        if mode == 'parquet':
            cols = _columns_for_label(label)
            if not _safe_read_parquet(path, columns=cols):
                unreadable.append(label)
    return missing, unreadable


def _load_metadata(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if 'pdbID' in df.columns:
        return df
    if 'pdbId' in df.columns:
        df = df.rename(columns={'pdbId': 'pdbID'})
        return df
    if 'pdb_id' in df.columns:
        df = df.rename(columns={'pdb_id': 'pdbID'})
        return df
    # Fallback: first column.
    df = df.rename(columns={df.columns[0]: 'pdbID'})
    return df


def main():
    p = argparse.ArgumentParser(description='Gate pipeline steps and emit filtered metadata + manifest')
    p.add_argument('--stage', required=True, choices=['preprocess', 'nodes_edges', 'fill_edge'])
    p.add_argument('--metadata', required=True, help='Input metadata CSV')
    p.add_argument('--processed_dir', default='', help='Processed data dir (required for preprocess gate)')
    p.add_argument('--nodes_edges_dir', default='', help='Nodes/edges dir (required for nodes_edges/fill_edge gate)')
    p.add_argument('--out_dir', required=True, help='Output directory for manifest + filtered metadata')
    p.add_argument('--keep_failed', action='store_true', help='Keep failed rows in filtered metadata (default: drop)')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = _load_metadata(args.metadata)
    pdb_ids = [str(x) for x in df['pdbID'].tolist()]

    results = []
    ok_mask = []
    for pdb_id in pdb_ids:
        try:
            required = _required_paths_for_stage(pdb_id, args.stage, args.processed_dir, args.nodes_edges_dir)
            missing, unreadable = _check_paths(required)
            ok = (len(missing) == 0 and len(unreadable) == 0)
            msg = ''
            if missing:
                msg += f"missing={','.join(missing)}"
            if unreadable:
                if msg:
                    msg += ';'
                msg += f"unreadable={','.join(unreadable)}"

            results.append(GateResult(pdb_id=pdb_id, ok=ok, stage=args.stage, missing=','.join(missing), error=msg))
            ok_mask.append(ok)
        except Exception as e:
            results.append(GateResult(pdb_id=pdb_id, ok=False, stage=args.stage, missing='', error=f"exception={type(e).__name__}:{e}"))
            ok_mask.append(False)

    manifest_path = os.path.join(args.out_dir, f"manifest.{args.stage}.csv")
    with open(manifest_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['timestamp', 'stage', 'pdbID', 'ok', 'missing', 'error'])
        ts = _now()
        for r in results:
            w.writerow([ts, r.stage, r.pdb_id, int(r.ok), r.missing, r.error])

    if args.keep_failed:
        filtered = df.copy()
        filtered['gate_ok'] = [int(x) for x in ok_mask]
    else:
        filtered = df.loc[ok_mask].copy()

    filtered_path = os.path.join(args.out_dir, f"metadata.ok.{args.stage}.csv")
    filtered.to_csv(filtered_path, index=False)

    n_ok = sum(1 for x in ok_mask if x)
    print(f"[{_now()}] Gate stage={args.stage} ok={n_ok}/{len(ok_mask)}")
    print(f"manifest={manifest_path}")
    print(f"filtered_metadata={filtered_path}")


if __name__ == '__main__':
    main()
