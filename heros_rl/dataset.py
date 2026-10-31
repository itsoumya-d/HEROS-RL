# SPDX-License-Identifier: MIT
"""Deterministic 500-case destructive-DDL migration dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from .prompts import build_prompt

BASE_COLUMNS = "COLUMN id bigint NOT_NULL\nCOLUMN email text NOT_NULL\nCOLUMN created_at timestamp NOT_NULL"


def _case(
    index: int,
    family: str,
    old_schema: str,
    target_schema: str,
    *,
    data_loss: bool,
    requires_expand_contract: bool,
    expected_tier: str,
) -> dict:
    case = {
        "id": f"heros-mig-{index:04d}",
        "hazard": family,
        "data_loss": data_loss,
        "requires_expand_contract": requires_expand_contract,
        "expected_tier": expected_tier,
        "old_schema": old_schema,
        "target_schema": target_schema,
    }
    case["prompt"] = build_prompt(case)
    return case


def _family_builders() -> list[Callable[[int, str], dict]]:
    def table(name: str, columns: str) -> str:
        return f"TABLE {name}\n{columns}"

    def drop_column(i: int, table_name: str) -> dict:
        old = table(table_name, f"{BASE_COLUMNS}\nCOLUMN legacy_note text NULLABLE")
        target = table(table_name, BASE_COLUMNS)
        return _case(
            i,
            "drop_column",
            old,
            target,
            data_loss=True,
            requires_expand_contract=True,
            expected_tier="CRITICAL",
        )

    def drop_table(i: int, table_name: str) -> dict:
        audit = table(
            f"{table_name}_audit",
            "COLUMN id bigint NOT_NULL\nCOLUMN payload text NULLABLE",
        )
        old = f"{table(table_name, BASE_COLUMNS)}\n\n{audit}"
        target = table(table_name, BASE_COLUMNS)
        return _case(
            i,
            "drop_table",
            old,
            target,
            data_loss=True,
            requires_expand_contract=True,
            expected_tier="CRITICAL",
        )

    def type_narrowing(i: int, table_name: str) -> dict:
        old = table(table_name, BASE_COLUMNS.replace("bigint", "bigint") + "\nCOLUMN amount bigint NOT_NULL")
        target = table(table_name, BASE_COLUMNS + "\nCOLUMN amount integer NOT_NULL")
        return _case(
            i,
            "type_narrowing",
            old,
            target,
            data_loss=True,
            requires_expand_contract=True,
            expected_tier="HIGH",
        )

    def set_not_null(i: int, table_name: str) -> dict:
        old = table(table_name, BASE_COLUMNS.replace("COLUMN email text NOT_NULL", "COLUMN email text NULLABLE"))
        target = table(table_name, BASE_COLUMNS)
        return _case(
            i,
            "set_not_null",
            old,
            target,
            data_loss=False,
            requires_expand_contract=True,
            expected_tier="HIGH",
        )

    def rename_column(i: int, table_name: str) -> dict:
        old = table(table_name, BASE_COLUMNS + "\nCOLUMN full_name text NULLABLE")
        target = table(table_name, BASE_COLUMNS + "\nCOLUMN display_name text NULLABLE")
        return _case(
            i,
            "rename_column",
            old,
            target,
            data_loss=False,
            requires_expand_contract=True,
            expected_tier="HIGH",
        )

    def add_primary_key(i: int, table_name: str) -> dict:
        old = table(table_name, BASE_COLUMNS)
        target = table(
            table_name,
            BASE_COLUMNS.replace(
                "COLUMN id bigint NOT_NULL",
                "COLUMN id bigint NOT_NULL PRIMARY_KEY",
            ),
        )
        return _case(
            i,
            "add_primary_key",
            old,
            target,
            data_loss=False,
            requires_expand_contract=True,
            expected_tier="HIGH",
        )

    def add_nullable_column(i: int, table_name: str) -> dict:
        old = table(table_name, BASE_COLUMNS)
        target = table(table_name, f"{BASE_COLUMNS}\nCOLUMN bio text NULLABLE")
        return _case(
            i,
            "add_nullable_column",
            old,
            target,
            data_loss=False,
            requires_expand_contract=False,
            expected_tier="SAFE",
        )

    def add_default_column(i: int, table_name: str) -> dict:
        old = table(table_name, BASE_COLUMNS)
        target = table(table_name, f"{BASE_COLUMNS}\nCOLUMN status text NOT_NULL DEFAULT")
        return _case(
            i,
            "add_default_column",
            old,
            target,
            data_loss=False,
            requires_expand_contract=False,
            expected_tier="NOTABLE",
        )

    def widen_type(i: int, table_name: str) -> dict:
        old = table(table_name, BASE_COLUMNS + "\nCOLUMN total bigint NOT_NULL")
        target = table(table_name, BASE_COLUMNS + "\nCOLUMN total numeric NOT_NULL")
        return _case(
            i,
            "widen_type",
            old,
            target,
            data_loss=False,
            requires_expand_contract=False,
            expected_tier="SAFE",
        )

    def create_table(i: int, table_name: str) -> dict:
        old = table(table_name, BASE_COLUMNS)
        events = table(
            f"{table_name}_events",
            "COLUMN id bigint NOT_NULL\nCOLUMN body text NULLABLE",
        )
        target = f"{old}\n\n{events}"
        return _case(
            i,
            "create_table",
            old,
            target,
            data_loss=False,
            requires_expand_contract=False,
            expected_tier="SAFE",
        )

    return [
        drop_column,
        drop_table,
        type_narrowing,
        set_not_null,
        rename_column,
        add_primary_key,
        add_nullable_column,
        add_default_column,
        widen_type,
        create_table,
    ]


def generate_cases(count: int = 500, seed: int = 42) -> list[dict]:
    if count <= 0:
        raise ValueError("count must be positive")
    builders = _family_builders()
    cases: list[dict] = []
    for index in range(1, count + 1):
        builder = builders[(index - 1) % len(builders)]
        suffix = (index * 7919 + seed) % 100000
        table_name = f"entity_{suffix:05d}"
        cases.append(builder(index, table_name))
    return cases


def write_dataset(path: str | Path, cases: list[dict]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case, sort_keys=True) + "\n")


def load_dataset(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def dataset_stats(cases: list[dict]) -> dict:
    families: dict[str, int] = {}
    for case in cases:
        families[case["hazard"]] = families.get(case["hazard"], 0) + 1
    return {
        "count": len(cases),
        "families": dict(sorted(families.items())),
        "data_loss": sum(1 for case in cases if case["data_loss"]),
        "requires_expand_contract": sum(
            1 for case in cases if case["requires_expand_contract"]
        ),
        "tiers": {
            tier: sum(1 for case in cases if case["expected_tier"] == tier)
            for tier in ("SAFE", "NOTABLE", "MEDIUM", "HIGH", "CRITICAL")
        },
    }
