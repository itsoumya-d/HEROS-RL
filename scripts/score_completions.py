#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Score pre-generated completions without a GPU.

Input JSONL rows:
  {"id": "heros-mig-0001", "completion": "..."}
or
  {"id": "heros-mig-0001", "completions": ["...", "...", ...]}

This is the same reward path used by training: format tags, SQL safety audit,
and optional forge analysis.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from heros_rl.eval import evaluate_by_group, evaluate_generations  # noqa: E402
from heros_rl.forge import ForgeClient  # noqa: E402
from heros_rl.rewards import RewardFunction  # noqa: E402


def load_rows(path: str) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--completions", required=True)
    parser.add_argument(
        "--dataset", default=str(ROOT / "data" / "heros_migration_500.jsonl")
    )
    parser.add_argument("--output", default=str(ROOT / "outputs" / "scores.json"))
    args = parser.parse_args()

    dataset = load_rows(args.dataset)
    cases_by_id = {case["id"]: case for case in dataset}
    rows = load_rows(args.completions)
    reward_fn = RewardFunction(forge_client=ForgeClient())

    if rows and "completions" in rows[0]:
        samples = {row["id"]: row["completions"] for row in rows}
        metrics = evaluate_by_group(samples, cases_by_id, reward_fn)
    else:
        completions = [row["completion"] for row in rows]
        cases = [cases_by_id[row["id"]] for row in rows]
        metrics = evaluate_generations(completions, cases, reward_fn)

    payload = {"completions": args.completions, **metrics}
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
