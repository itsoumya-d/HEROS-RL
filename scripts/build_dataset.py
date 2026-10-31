#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Generate or verify the deterministic HEROS-RL migration dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from heros_rl.dataset import (  # noqa: E402
    dataset_stats,
    generate_cases,
    write_dataset,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default=str(ROOT / "data" / "heros_migration_500.jsonl"),
        help="output JSONL path",
    )
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the existing file matches a fresh generation",
    )
    args = parser.parse_args()

    cases = generate_cases(args.count, args.seed)
    destination = Path(args.out)

    if args.check:
        if not destination.exists():
            print(f"missing dataset: {destination}", file=sys.stderr)
            return 1
        expected = [
            json.dumps(case, sort_keys=True) for case in cases
        ]
        actual = [
            line.strip()
            for line in destination.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if expected != actual:
            print("dataset is not reproducible from the generator", file=sys.stderr)
            return 1
        print(json.dumps(dataset_stats(cases), indent=2))
        return 0

    write_dataset(destination, cases)
    print(json.dumps(dataset_stats(cases), indent=2))
    print(f"wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
