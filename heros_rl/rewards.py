# SPDX-License-Identifier: MIT
"""Deterministic rule-based rewards for migration rewrites.

Format reward follows DeepSeek-R1-Zero: reasoning must be inside  thinking tags
and the executable answer inside <answer> tags. Accuracy reward combines a
static SQL safety audit with an optional forge risk analysis.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from .forge import ForgeClient, ForgeError, ForgeUnavailable

THINK_RE = re.compile(r"<think\s*>(.*?)</think\s*>", re.DOTALL | re.IGNORECASE)
ANSWER_RE = re.compile(r"<answer\s*>(.*?)</answer\s*>", re.DOTALL | re.IGNORECASE)
SECTION_RE = re.compile(
    r"MIGRATION\s*:\s*(.*?)\s*ROLLBACK\s*:\s*(.*)", re.DOTALL | re.IGNORECASE
)

DESTRUCTIVE_RE = re.compile(
    r"\b("
    r"DROP\s+(COLUMN|TABLE|SCHEMA|INDEX|CONSTRAINT)"
    r"|TRUNCATE\s+TABLE"
    r"|DELETE\s+FROM"
    r"|ALTER\s+COLUMN\s+\w+\s+(SET\s+DATA\s+)?TYPE"
    r"|ALTER\s+COLUMN\s+\w+\s+SET\s+NOT\s+NULL"
    r"|ALTER\s+TABLE\s+\w+\s+RENAME\s+(COLUMN|TO)"
    r"|ALTER\s+TABLE\s+\w+\s+ADD\s+(CONSTRAINT\s+\w+\s+)?PRIMARY\s+KEY"
    r")\b",
    re.IGNORECASE,
)

ALLOWED_START = re.compile(
    r"^\s*(ALTER|CREATE|DROP|UPDATE|INSERT|DELETE|COMMENT|SET|BEGIN|COMMIT|"
    r"DO|REINDEX|VACUUM|ANALYZE|WITH)\b",
    re.IGNORECASE,
)


class RewardParseError(ValueError):
    """The completion does not contain the required tag structure."""


def extract_sections(text: str) -> tuple[str, str]:
    """Return (migration_sql, rollback_sql) from a completion."""
    answer = ANSWER_RE.search(text or "")
    if not answer:
        raise RewardParseError("missing <answer>...</answer> block")
    body = answer.group(1)
    match = SECTION_RE.search(body)
    if not match:
        raise RewardParseError("missing MIGRATION: / ROLLBACK: sections")

    migration = _strip_fence(match.group(1))
    rollback = _strip_fence(match.group(2))
    if not migration.strip():
        raise RewardParseError("MIGRATION section is empty")
    if not rollback.strip():
        raise RewardParseError("ROLLBACK section is empty")
    return migration, rollback


def format_reward(text: str) -> float:
    """Binary format reward: 1.0 only when both tags are present and non-empty."""
    if not text:
        return 0.0
    think = THINK_RE.search(text)
    answer = ANSWER_RE.search(text)
    if not think or not answer:
        return 0.0
    if not think.group(1).strip() or not answer.group(1).strip():
        return 0.0
    return 1.0


def _strip_fence(value: str) -> str:
    value = value.strip()
    value = re.sub(r"^```(?:sql)?\s*", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s*```$", "", value)
    return value.strip()


def split_sql(sql: str) -> list[str]:
    """Split on semicolons outside single/double quotes and dollar quotes."""
    statements: list[str] = []
    buffer: list[str] = []
    quote: str | None = None
    index = 0
    length = len(sql)

    while index < length:
        char = sql[index]
        if quote:
            buffer.append(char)
            if char == quote:
                if index + 1 < length and sql[index + 1] == quote:
                    buffer.append(sql[index + 1])
                    index += 1
                else:
                    quote = None
            index += 1
            continue

        if char in ("'", '"'):
            quote = char
            buffer.append(char)
            index += 1
            continue

        if char == "-" and index + 1 < length and sql[index + 1] == "-":
            newline = sql.find("\n", index)
            if newline == -1:
                break
            index = newline + 1
            buffer.append("\n")
            continue

        if char == "/" and index + 1 < length and sql[index + 1] == "*":
            end = sql.find("*/", index + 2)
            if end == -1:
                break
            index = end + 2
            continue

        if char == ";":
            statement = "".join(buffer).strip()
            if statement:
                statements.append(statement)
            buffer = []
            index += 1
            continue

        buffer.append(char)
        index += 1

    trailing = "".join(buffer).strip()
    if trailing:
        statements.append(trailing)
    return statements


def _strip_leading_comments(statement: str) -> str:
    value = statement.strip()
    while value.startswith("--"):
        newline = value.find("\n")
        if newline == -1:
            return ""
        value = value[newline + 1 :].lstrip()
    return value


def is_destructive(statement: str) -> bool:
    return bool(DESTRUCTIVE_RE.search(statement))


def static_accuracy(migration: str, rollback: str, case: dict[str, Any]) -> float:
    statements = split_sql(migration)
    rollback_statements = split_sql(rollback)
    if not statements or not rollback_statements:
        return 0.0

    cleaned = [_strip_leading_comments(statement) for statement in statements]
    if any(not ALLOWED_START.search(statement) for statement in cleaned):
        return 0.0

    first_destructive = is_destructive(cleaned[0]) if cleaned else False
    any_destructive = any(is_destructive(statement) for statement in cleaned)

    if case.get("requires_expand_contract"):
        if first_destructive:
            return 0.0
        score = 0.0
        if len(statements) >= 2:
            score += 0.5
        if not first_destructive:
            score += 0.3
        if rollback_statements:
            score += 0.2
        return min(score, 1.0)

    score = 0.7
    if not any_destructive:
        score += 0.2
    if rollback_statements:
        score += 0.1
    return min(score, 1.0)


@dataclass
class RewardBreakdown:
    format: float
    accuracy: float
    forge_score: float | None
    total: float
    requires_expand_contract: bool
    first_statement_destructive: bool
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RewardFunction:
    """Callable reward used by TRL and by the evaluation harness."""

    def __init__(
        self,
        forge_client: ForgeClient | None = None,
        format_weight: float = 0.5,
        accuracy_weight: float = 0.5,
    ):
        self.forge_client = forge_client or ForgeClient()
        self.format_weight = format_weight
        self.accuracy_weight = accuracy_weight

    def __call__(self, completion: str, case: dict[str, Any]) -> RewardBreakdown:
        fmt = format_reward(completion)
        forge_score: float | None = None
        error: str | None = None
        first_destructive = False

        try:
            migration, rollback = extract_sections(completion)
            if migration:
                first = _strip_leading_comments(split_sql(migration)[0]) if split_sql(migration) else ""
                first_destructive = is_destructive(first)
            accuracy = static_accuracy(migration, rollback, case)
        except RewardParseError as exc:
            accuracy = 0.0
            error = str(exc)

        if accuracy > 0 and self.forge_client.available:
            try:
                analysis = self.forge_client.analyze(
                    case["old_schema"], case["target_schema"]
                )
                forge_score = analysis.risk_score
                forge_requires_expand = (
                    analysis.decision_required
                    or analysis.has_data_loss
                    or analysis.risk_tier in {"HIGH", "CRITICAL"}
                )
                if forge_requires_expand and not case.get("requires_expand_contract"):
                    accuracy *= 0.5
                if not forge_requires_expand and case.get("requires_expand_contract"):
                    accuracy *= 0.75
            except (ForgeUnavailable, ForgeError) as exc:
                error = f"forge unavailable: {exc}"

        total = self.format_weight * fmt + self.accuracy_weight * accuracy
        return RewardBreakdown(
            format=fmt,
            accuracy=accuracy,
            forge_score=forge_score,
            total=total,
            requires_expand_contract=bool(case.get("requires_expand_contract")),
            first_statement_destructive=first_destructive,
            error=error,
        )

    def scalar(self, completion: str, case: dict[str, Any]) -> float:
        return self(completion, case).total
