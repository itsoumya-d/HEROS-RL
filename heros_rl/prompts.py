# SPDX-License-Identifier: MIT
"""Prompts for the migration-rewrite policy."""

SYSTEM_PROMPT = (
    "You are a PostgreSQL migration safety engineer. Rewrite a requested schema "
    "change as a zero-downtime expand-and-contract migration. Never put a "
    "destructive DDL statement in the first step. Preserve data and provide a "
    "rollback plan."
)

ANSWER_FORMAT = """Respond in exactly this shape:
<answer>
MIGRATION:
-- phase 1: expand
<SQL statements>
-- phase 2: backfill
<SQL statements>
-- phase 3: contract
<SQL statements>
ROLLBACK:
<SQL statements>
</answer>

Rules:
- Use PostgreSQL syntax.
- Every statement ends with a semicolon.
- For destructive or lock-heavy changes, use at least two migration statements
  and keep DROP/TRUNCATE/type narrowing/SET NOT NULL out of the first statement.
- ROLLBACK must restore the old schema shape.
"""


def build_prompt(case: dict) -> str:
    """Build the user prompt for one dataset row."""
    return (
        f"{ANSWER_FORMAT}\n"
        f"Hazard class: {case['hazard']}\n"
        f"Data-loss change: {str(case['data_loss']).lower()}\n"
        f"Requires expand-and-contract: {str(case['requires_expand_contract']).lower()}\n\n"
        f"OLD SCHEMA (Forge format):\n{case['old_schema']}\n\n"
        f"TARGET SCHEMA (Forge format):\n{case['target_schema']}\n"
    )
