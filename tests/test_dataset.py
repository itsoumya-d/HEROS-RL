# SPDX-License-Identifier: MIT
import json
import unittest
from pathlib import Path

from heros_rl.dataset import dataset_stats, generate_cases, load_dataset

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "data" / "heros_migration_500.jsonl"


class DatasetTests(unittest.TestCase):
    def test_generated_dataset_is_500_unique_cases(self):
        cases = generate_cases()
        self.assertEqual(len(cases), 500)
        self.assertEqual(len({case["id"] for case in cases}), 500)

    def test_generation_is_deterministic(self):
        first = generate_cases(50, seed=7)
        second = generate_cases(50, seed=7)
        self.assertEqual(first, second)

    def test_family_distribution_and_labels(self):
        stats = dataset_stats(generate_cases())
        self.assertEqual(stats["count"], 500)
        self.assertEqual(len(stats["families"]), 10)
        self.assertEqual(stats["data_loss"], 150)
        self.assertEqual(stats["requires_expand_contract"], 300)
        self.assertEqual(stats["tiers"]["CRITICAL"], 100)
        self.assertEqual(stats["tiers"]["SAFE"], 150)

    def test_committed_jsonl_matches_generator(self):
        rows = load_dataset(DATASET)
        self.assertEqual(len(rows), 500)
        expected = generate_cases()
        for row, case in zip(rows, expected):
            self.assertEqual(row, case)

    def test_prompts_include_both_schemas_and_tag_format(self):
        for case in generate_cases()[:20]:
            self.assertIn(case["old_schema"], case["prompt"])
            self.assertIn(case["target_schema"], case["prompt"])
            self.assertIn("<answer>", case["prompt"])
            self.assertIn("ROLLBACK", case["prompt"])

    def test_jsonl_is_stable_json(self):
        first = DATASET.read_text(encoding="utf-8").splitlines()[0]
        parsed = json.loads(first)
        self.assertEqual(parsed["id"], "heros-mig-0001")


if __name__ == "__main__":
    unittest.main()
