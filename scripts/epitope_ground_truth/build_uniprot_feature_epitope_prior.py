#!/usr/bin/env python3
"""
Tier 2b stub: UniProt feature regions (site / region / antibody binding) → residue mask.

TODO: call UniProt REST (`/uniprotkb/{accession}`) or flat-file hook, parse features,
map via PDBe SIFTS to graph resId (reuse build_iedb_epitope_prior helpers).

This entrypoint documents the CLI only; run with --implement to attempt a future
implementation (currently raises NotImplementedError).

Examples:
  python scripts/epitope_ground_truth/build_uniprot_feature_epitope_prior.py --help
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description="UniProt feature epitope prior (Tier 2b stub).")
    ap.add_argument("--metadata", help="Cohort metadata CSV (pdbID, antigen, UniProt)")
    ap.add_argument("--nodes_edges_dir", help="nodes_edges root")
    ap.add_argument("--processed_dir", help="processed_data root (lig.pdb)")
    ap.add_argument("--pdb_list")
    ap.add_argument("--sifts_cache_dir")
    ap.add_argument(
        "--output_parquet",
        default="node_uniprot_feature_epitope_prior.parquet",
    )
    ap.add_argument("--manifest_out", default="uniprot_feature_prior_manifest.csv")
    ap.add_argument(
        "--implement",
        action="store_true",
        help="Reserved: will run the real mapper when implemented",
    )
    args = ap.parse_args()

    if not args.implement:
        print(
            "Tier 2b stub: pass --implement to run (not yet implemented). "
            "See module docstring and docs/EPITOPE_GROUND_TRUTH_TIERS.md.",
            file=sys.stderr,
        )
        return 0

    raise NotImplementedError(
        "UniProt REST feature parsing + SIFTS alignment — see TODO in module docstring."
    )


if __name__ == "__main__":
    raise SystemExit(main())
