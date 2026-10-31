# SPDX-License-Identifier: MIT
"""Pass@1 and consensus@k evaluation harness."""

from __future__ import annotations

from collections import Counter
from typing import Sequence

from .rewards import RewardBreakdown, RewardFunction


def normalized_answer(text: str) -> str:
    return " ".join(text.lower().split())


def pass_at_1(rewards: Sequence[RewardBreakdown], threshold: float = 0.99) -> float:
    if not rewards:
        return 0.0
    return sum(1 for reward in rewards if reward.accuracy >= threshold) / len(rewards)


def majority_vote(answers: Sequence[str]) -> str:
    if not answers:
        raise ValueError("answers must not be empty")
    counts = Counter(normalized_answer(answer) for answer in answers)
    return counts.most_common(1)[0][0]


def consensus_at_k(
    completions: Sequence[str],
    cases: Sequence[dict],
    reward_fn: RewardFunction,
    threshold: float = 0.99,
) -> float:
    """Fraction of cases where the modal completion is fully correct."""
    if len(completions) != len(cases):
        raise ValueError("completions and cases must have the same length")
    if not cases:
        return 0.0

    grouped: dict[str, list[tuple[str, dict]]] = {}
    for completion, case in zip(completions, cases):
        grouped.setdefault(case["id"], []).append((completion, case))

    correct = 0
    for samples in grouped.values():
        modal = majority_vote([completion for completion, _ in samples])
        chosen = next(
            (completion, case)
            for completion, case in samples
            if normalized_answer(completion) == modal
        )
        reward = reward_fn(*chosen)
        if reward.accuracy >= threshold:
            correct += 1
    return correct / len(grouped)


def evaluate_generations(
    completions: Sequence[str],
    cases: Sequence[dict],
    reward_fn: RewardFunction | None = None,
    threshold: float = 0.99,
) -> dict:
    if len(completions) != len(cases):
        raise ValueError("completions and cases must have the same length")
    reward_fn = reward_fn or RewardFunction()
    rewards = [
        reward_fn(completion, case)
        for completion, case in zip(completions, cases)
    ]
    return {
        "count": len(cases),
        "pass@1": pass_at_1(rewards, threshold),
        "mean_accuracy": sum(reward.accuracy for reward in rewards) / len(rewards)
        if rewards
        else 0.0,
        "mean_format": sum(reward.format for reward in rewards) / len(rewards)
        if rewards
        else 0.0,
        "mean_reward": sum(reward.total for reward in rewards) / len(rewards)
        if rewards
        else 0.0,
        "cons@k": consensus_at_k(completions, cases, reward_fn, threshold),
    }


def evaluate_by_group(
    samples_by_case: dict[str, list[str]],
    cases_by_id: dict[str, dict],
    reward_fn: RewardFunction | None = None,
    threshold: float = 0.99,
) -> dict:
    """Compute Pass@1 and cons@k from k samples per case."""
    reward_fn = reward_fn or RewardFunction()
    flat_completions: list[str] = []
    flat_cases: list[dict] = []
    for case_id, samples in samples_by_case.items():
        case = cases_by_id[case_id]
        for sample in samples:
            flat_completions.append(sample)
            flat_cases.append(case)
    return evaluate_generations(
        flat_completions, flat_cases, reward_fn, threshold
    )


