#!/usr/bin/env python3
"""Audit preprocessing completeness for the 20 test3A PDBs.

Checks for required artifacts in:
- processed_data/<pdbId>/
- nodes_edges/<pdbId>/

Emits a CSV report to stdout and to logs/preprocess_audit_test3A.csv with columns:
  pdbId,processed_ok,nodes_edges_ok,missing_files
"""

import csv
import os
from pathlib import Path
from typing import List

import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parents[2]
INFERENCE_LIST = PROJ_ROOT / "inference_list_test3A.csv"
# Upstream preprocess root (matches slurm scripts)
UP_ROOT = Path("/leonardo_scratch/fast/AIFAC_F01_302/epi4ab/upstream_preprocess")
PROCESSED_ROOT = UP_ROOT / "processed_data"
NODES_EDGES_ROOT = UP_ROOT / "nodes_edges"

REQUIRED_PROCESSED = [
    "pdb_profile.parquet",
    "node_label_pi.parquet",
    "aac/aac_result.parquet",
    "angle/angle_result.parquet",
    "charge/charge_result.parquet",
    "charge_composition/cc_result.parquet",
    "depth/depth_result.parquet",
]

REQUIRED_NODES_EDGES = [
    "edge_attribute_charge.parquet",
    "edge_attribute_dist.parquet",
    "edge_index.parquet",
    "node_feature.parquet",
    "node_label_pi.parquet",
]


def list_missing(base: Path, required: List[str]) -> List[str]:
    missing = []
    for rel in required:
        if not (base / rel).is_file():
            missing.append(str(rel))
    return missing


def main() -> None:
    if not INFERENCE_LIST.is_file():
        raise SystemExit(f"Inference list not found: {INFERENCE_LIST}")

    df = pd.read_csv(INFERENCE_LIST)
    if "pdbId" not in df.columns:
        raise SystemExit(f"Expected column 'pdbId' in {INFERENCE_LIST}")

    rows = []
    for pdb_id in df["pdbId"].tolist():
        proc_dir = PROCESSED_ROOT / pdb_id
        ne_dir = NODES_EDGES_ROOT / pdb_id

        missing_proc = list_missing(proc_dir, REQUIRED_PROCESSED) if proc_dir.is_dir() else [f"<no processed_data dir {proc_dir}>"]
        missing_ne = list_missing(ne_dir, REQUIRED_NODES_EDGES) if ne_dir.is_dir() else [f"<no nodes_edges dir {ne_dir}>"]

        processed_ok = len(missing_proc) == 0
        nodes_edges_ok = len(missing_ne) == 0

        all_missing = []
        if not processed_ok:
            all_missing.extend(f"processed:{m}" for m in missing_proc)
        if not nodes_edges_ok:
            all_missing.extend(f"nodes_edges:{m}" for m in missing_ne)

        rows.append({
            "pdbId": pdb_id,
            "processed_ok": processed_ok,
            "nodes_edges_ok": nodes_edges_ok,
            "missing_files": ";".join(all_missing) if all_missing else "",
        })

    logs_dir = PROJ_ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)
    out_path = logs_dir / "preprocess_audit_test3A.csv"

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["pdbId", "processed_ok", "nodes_edges_ok", "missing_files"])
        writer.writeheader()
        writer.writerows(rows)

    # Also print to stdout for convenience
    print(f"Wrote audit report to {out_path}")
    print("pdbId,processed_ok,nodes_edges_ok,missing_files")
    for r in rows:
        print(f"{r['pdbId']},{r['processed_ok']},{r['nodes_edges_ok']},{r['missing_files']}")


if __name__ == "__main__":
    main()

