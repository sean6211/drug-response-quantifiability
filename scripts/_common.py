"""Paths and table writing shared by the analysis scripts."""

from __future__ import annotations

import argparse
from pathlib import Path

from quantifiability.dataset import Cohort

REPO = Path(__file__).resolve().parents[1]
DEFAULT_DATA = REPO / "data" / "prepared"
DEFAULT_EVE = REPO.parent / "evebio_datasets"
DEFAULT_OUT = REPO / "outputs"


def prepared_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser


def load_cohort(path: Path) -> Cohort:
    path = Path(path)
    if not (path / "X.npy").exists():
        raise SystemExit(
            f"No prepared matrix at {path}. Run: python scripts/prepare_dataset.py"
        )
    return Cohort.load(path)


def write_table(out_dir: Path, name: str, frame) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    frame.to_csv(path, index=False)
    print(frame.to_string(index=False))
    print(f"\nWrote {path}")
    return path
