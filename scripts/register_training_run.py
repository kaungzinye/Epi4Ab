#!/usr/bin/env python3
"""Copy train_loss.png to plots hub and append manifest.json (called from SLURM)."""

from __future__ import annotations

import argparse
import os

from epi4ab_plot_paths import copy_train_loss_to_hub, register_run


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--backend", default="")
    ap.add_argument("--split_seed", default="")
    args = ap.parse_args()

    dst = copy_train_loss_to_hub(args.run_id, args.out_dir)
    register_run(
        args.run_id,
        out_dir=args.out_dir,
        test_record_dir=os.path.join(args.out_dir, "test_record"),
        backend=args.backend,
        split_seed=args.split_seed,
    )
    if dst:
        print(f"Copied train_loss -> {dst}")
    print(f"Registered run_id={args.run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
