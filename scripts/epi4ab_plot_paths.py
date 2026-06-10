"""Central paths for Epi4Ab scratch storage and the plots hub."""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

_DEFAULT_SCRATCH = "/leonardo_scratch/fast/EUHPC_D29_035/epi4ab"
_DEFAULT_OUT_BASE_TAG = "20260305_143201"
_DEFAULT_LARGE_ARCHIVE = "/leonardo_scratch/large/userexternal/knaung00/epi4ab_archive"


def scratch_root() -> Path:
    return Path(os.environ.get("EPI4AB_SCRATCH", _DEFAULT_SCRATCH))


def plots_root() -> Path:
    return Path(os.environ.get("EPI4AB_PLOTS_ROOT", str(scratch_root() / "plots")))


def large_archive_root() -> Path:
    return Path(os.environ.get("EPI4AB_LARGE_ARCHIVE", _DEFAULT_LARGE_ARCHIVE))


def out_base() -> Path:
    if tag := os.environ.get("EPI4AB_OUT_BASE"):
        return Path(tag)
    tag = os.environ.get("EPI4AB_OUT_BASE_TAG", _DEFAULT_OUT_BASE_TAG)
    return scratch_root() / "upstream_preprocess_autodetect" / tag


def out_base_tag() -> str:
    env = os.environ.get("EPI4AB_OUT_BASE")
    if env:
        return Path(env).name
    return os.environ.get("EPI4AB_OUT_BASE_TAG", _DEFAULT_OUT_BASE_TAG)


def cache_dir() -> Path:
    return out_base() / "cache"


def cohort_dir(tag: str | None = None) -> Path:
    return plots_root() / "cohort" / (tag or out_base_tag())


def cohort_html_dir(tag: str | None = None) -> Path:
    return cohort_dir(tag) / "html"


def cohort_csv_dir(tag: str | None = None) -> Path:
    return cohort_dir(tag) / "csv"


def cohort_literature_dual_dir(tag: str | None = None) -> Path:
    return cohort_dir(tag) / "literature_dual_by_run"


def run_dir(run_id: str) -> Path:
    return plots_root() / "runs" / run_id


def run_html_dir(run_id: str) -> Path:
    return run_dir(run_id) / "html"


def run_png_dir(run_id: str) -> Path:
    return run_dir(run_id) / "png"


def run_csv_dir(run_id: str) -> Path:
    return run_dir(run_id) / "csv"


def legacy_dir(name: str) -> Path:
    return plots_root() / "legacy" / name


def manifest_path() -> Path:
    return plots_root() / "manifest.json"


def default_cohort_html(name: str, tag: str | None = None) -> Path:
    stem = name if name.endswith(".html") else f"{name}.html"
    return cohort_html_dir(tag) / stem


def default_run_html(run_id: str, name: str) -> Path:
    stem = name if name.endswith(".html") else f"{name}.html"
    return run_html_dir(run_id) / stem


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_run_id(explicit: str | None = None) -> str | None:
    if explicit:
        return explicit
    return os.environ.get("RUN_ID") or None


def load_manifest() -> list[dict]:
    path = manifest_path()
    if not path.is_file():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def save_manifest(entries: list[dict]) -> None:
    ensure_dir(plots_root())
    manifest_path().write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")


def register_run(
    run_id: str,
    *,
    out_dir: str | Path | None = None,
    test_record_dir: str | Path | None = None,
    backend: str = "",
    split_seed: str | int = "",
    label_file: str = "",
    notes: str = "",
) -> None:
    entries = load_manifest()
    rec = {
        "run_id": run_id,
        "backend": backend,
        "split_seed": split_seed,
        "label_file": label_file,
        "out_dir": str(out_dir) if out_dir else "",
        "test_record_dir": str(test_record_dir) if test_record_dir else "",
        "notes": notes,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    entries = [e for e in entries if e.get("run_id") != run_id]
    entries.append(rec)
    save_manifest(entries)


def copy_train_loss_to_hub(run_id: str, out_dir: str | Path) -> Path | None:
    src = Path(out_dir) / "train_loss.png"
    if not src.is_file():
        for sub in sorted(Path(out_dir).glob("*_GNNResNet_*")):
            candidate = sub / "train_loss.png"
            if candidate.is_file():
                src = candidate
                break
        else:
            return None
    dst_dir = ensure_dir(run_png_dir(run_id))
    dst = dst_dir / "train_loss.png"
    shutil.copy2(src, dst)
    return dst


def add_plot_path_args(ap) -> None:
    """Add standard hub CLI flags to an argparse parser."""
    ap.add_argument(
        "--plots_root",
        default=None,
        help=f"Plots hub root (default: {plots_root()})",
    )
    ap.add_argument(
        "--out_base_tag",
        default=None,
        help=f"Cohort tag under plots/cohort/ (default: {out_base_tag()})",
    )
    ap.add_argument(
        "--run_id",
        default=None,
        help="Run id under plots/runs/ (default: $RUN_ID)",
    )
    ap.add_argument(
        "--build-index",
        action="store_true",
        help="Regenerate plots/index.html after writing outputs",
    )


def hub_context(args) -> dict:
    pr = Path(args.plots_root) if getattr(args, "plots_root", None) else plots_root()
    tag = getattr(args, "out_base_tag", None) or out_base_tag()
    rid = resolve_run_id(getattr(args, "run_id", None))
    return {"plots_root": pr, "out_base_tag": tag, "run_id": rid}


def maybe_build_index(args) -> None:
    if not getattr(args, "build_index", False):
        return
    import subprocess
    import sys

    script = Path(__file__).resolve().parent / "build_plots_index.py"
    subprocess.run([sys.executable, str(script)], check=False)
