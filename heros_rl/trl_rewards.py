# SPDX-License-Identifier: MIT
"""TRL-compatible reward callbacks."""

from __future__ import annotations

from typing import Any

from .rewards import RewardFunction, format_reward


def normalize_completion(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict):
            return str(last.get("content", ""))
    return str(completion)


def _case(kwargs: dict[str, Any], index: int) -> dict[str, Any]:
    def value(name: str, default: Any = "") -> Any:
        column = kwargs.get(name)
        if isinstance(column, list) and index < len(column):
            return column[index]
        return default

    return {
        "id": value("id", f"row-{index}"),
        "hazard": value("hazard", "unknown"),
        "old_schema": value("old_schema", ""),
        "target_schema": value("target_schema", ""),
        "data_loss": bool(value("data_loss", False)),
        "requires_expand_contract": bool(
            value("requires_expand_contract", False)
        ),
    }


def make_reward_funcs(reward_fn: RewardFunction):
    def format_reward_func(completions, **kwargs):
        return [format_reward(normalize_completion(item)) for item in completions]

    def accuracy_reward_func(completions, **kwargs):
        return [
            reward_fn(normalize_completion(item), _case(kwargs, index)).accuracy
            for index, item in enumerate(completions)
        ]

    return format_reward_func, accuracy_reward_func
