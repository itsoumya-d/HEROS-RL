# SPDX-License-Identifier: MIT
"""HEROS-RL: critic-free GRPO for zero-downtime PostgreSQL migrations."""

from .grpo import GRPOLossResult, group_relative_advantages, grpo_loss
from .mining import (
    CorpusDocument,
    DomainResult,
    HashedNgramClassifier,
    IterationResult,
    MiningRun,
    SeedDocument,
    mine_domains,
    path_pattern,
    tokenize,
)
from .prompts import SYSTEM_PROMPT, build_prompt
from .rewards import (
    RewardBreakdown,
    RewardFunction,
    extract_sections,
    format_reward,
    split_sql,
)

__all__ = [
    "CorpusDocument",
    "DomainResult",
    "GRPOLossResult",
    "HashedNgramClassifier",
    "IterationResult",
    "MiningRun",
    "SeedDocument",
    "mine_domains",
    "path_pattern",
    "tokenize",
    "RewardBreakdown",
    "RewardFunction",
    "SYSTEM_PROMPT",
    "build_prompt",
    "extract_sections",
    "format_reward",
    "group_relative_advantages",
    "grpo_loss",
    "split_sql",
]
