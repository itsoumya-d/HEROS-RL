# SPDX-License-Identifier: MIT
import unittest

from heros_rl.forge import ForgeAnalysis
from heros_rl.rewards import (
    RewardFunction,
    extract_sections,
    format_reward,
    split_sql,
    static_accuracy,
)

SAFE_CASE = {
    "old_schema": "TABLE users\nCOLUMN id bigint NOT_NULL",
    "target_schema": "TABLE users\nCOLUMN id bigint NOT_NULL\nCOLUMN bio text NULLABLE",
    "requires_expand_contract": False,
    "data_loss": False,
}

EXPAND_CASE = {
    "old_schema": "TABLE users\nCOLUMN full_name text NULLABLE",
    "target_schema": "TABLE users\nCOLUMN display_name text NULLABLE",
    "requires_expand_contract": True,
    "data_loss": False,
}

GOOD_SAFE = """<think>The column is nullable and has no default, so one ALTER is safe.</think>
<answer>
MIGRATION:
ALTER TABLE users ADD COLUMN bio text;
ROLLBACK:
ALTER TABLE users DROP COLUMN bio;
</answer>"""

GOOD_EXPAND = """<think>Rename with a compatibility window.</think>
<answer>
MIGRATION:
-- phase 1: expand
ALTER TABLE users ADD COLUMN display_name text;
-- phase 2: backfill
UPDATE users SET display_name = full_name;
ROLLBACK:
ALTER TABLE users DROP COLUMN display_name;
</answer>"""

BAD_EXPAND = """<think>Rename directly.</think>
<answer>
MIGRATION:
ALTER TABLE users RENAME COLUMN full_name TO display_name;
ROLLBACK:
ALTER TABLE users RENAME COLUMN display_name TO full_name;
</answer>"""

NO_ROLLBACK = """<think>x</think><answer>
MIGRATION:
ALTER TABLE users ADD COLUMN bio text;
</answer>"""


class FormatRewardTests(unittest.TestCase):
    def test_complete_tags(self):
        self.assertEqual(format_reward(GOOD_SAFE), 1.0)

    def test_missing_tags(self):
        self.assertEqual(format_reward("ALTER TABLE users ADD COLUMN bio text;"), 0.0)
        self.assertEqual(format_reward("<think>x</think>no answer"), 0.0)
        self.assertEqual(format_reward("<answer>MIGRATION: x</answer>"), 0.0)


class SectionTests(unittest.TestCase):
    def test_extract_sections(self):
        migration, rollback = extract_sections(GOOD_SAFE)
        self.assertIn("ADD COLUMN bio", migration)
        self.assertIn("DROP COLUMN bio", rollback)

    def test_missing_rollback_rejected(self):
        with self.assertRaises(Exception):
            extract_sections(NO_ROLLBACK)

    def test_split_sql_ignores_semicolons_in_strings(self):
        statements = split_sql("UPDATE t SET note = 'a; b'; DELETE FROM t;")
        self.assertEqual(len(statements), 2)


class StaticAccuracyTests(unittest.TestCase):
    def test_safe_change_scores_one(self):
        migration, rollback = extract_sections(GOOD_SAFE)
        self.assertAlmostEqual(static_accuracy(migration, rollback, SAFE_CASE), 1.0)

    def test_expand_contract_scores_one(self):
        migration, rollback = extract_sections(GOOD_EXPAND)
        self.assertAlmostEqual(static_accuracy(migration, rollback, EXPAND_CASE), 1.0)

    def test_direct_rename_without_expand_scores_zero(self):
        migration, rollback = extract_sections(BAD_EXPAND)
        self.assertEqual(static_accuracy(migration, rollback, EXPAND_CASE), 0.0)

    def test_single_statement_expand_scores_low(self):
        migration = "ALTER TABLE users ADD COLUMN display_name text;"
        rollback = "ALTER TABLE users DROP COLUMN display_name;"
        score = static_accuracy(migration, rollback, EXPAND_CASE)
        self.assertLess(score, 0.8)
        self.assertGreater(score, 0.0)

    def test_unqualified_statement_scores_zero(self):
        score = static_accuracy("not sql at all;", "not sql either;", SAFE_CASE)
        self.assertEqual(score, 0.0)


class FakeForge:
    available = True

    def analyze(self, old_schema, target_schema):
        return ForgeAnalysis(
            risk_tier="HIGH",
            risk_score=0.25,
            has_data_loss=False,
            decision_required=True,
            raw={},
        )


class RewardFunctionTests(unittest.TestCase):
    def test_total_combines_format_and_accuracy(self):
        reward = RewardFunction(forge_client=FakeForge())(GOOD_EXPAND, EXPAND_CASE)
        self.assertEqual(reward.format, 1.0)
        self.assertEqual(reward.accuracy, 1.0)
        self.assertEqual(reward.forge_score, 0.25)
        self.assertAlmostEqual(reward.total, 1.0)
        self.assertFalse(reward.first_statement_destructive)

    def test_bad_expand_is_penalized(self):
        reward = RewardFunction(forge_client=FakeForge())(BAD_EXPAND, EXPAND_CASE)
        self.assertEqual(reward.accuracy, 0.0)
        self.assertLess(reward.total, 0.6)

    def test_unparseable_completion_gets_zero(self):
        reward = RewardFunction(forge_client=FakeForge())("no tags", EXPAND_CASE)
        self.assertEqual(reward.format, 0.0)
        self.assertEqual(reward.accuracy, 0.0)
        self.assertEqual(reward.total, 0.0)
        self.assertIsNotNone(reward.error)


if __name__ == "__main__":
    unittest.main()
