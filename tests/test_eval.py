# SPDX-License-Identifier: MIT
import unittest

from heros_rl.eval import consensus_at_k, evaluate_generations, pass_at_1
from heros_rl.rewards import RewardBreakdown


def reward(accuracy):
    return RewardBreakdown(
        format=1.0,
        accuracy=accuracy,
        forge_score=None,
        total=0.5 + 0.5 * accuracy,
        requires_expand_contract=True,
        first_statement_destructive=False,
    )


def stub_reward(completion, case):
    return reward(1.0 if "GOOD" in completion else 0.0)


class EvalTests(unittest.TestCase):
    def test_pass_at_1(self):
        self.assertAlmostEqual(pass_at_1([reward(1.0), reward(1.0), reward(0.0)]), 2 / 3)

    def test_pass_at_1_empty(self):
        self.assertEqual(pass_at_1([]), 0.0)

    def test_consensus_picks_modal_answer(self):
        case = {"id": "c1", "requires_expand_contract": True}
        completions = ["GOOD plan", "GOOD plan", "bad plan"]
        self.assertEqual(consensus_at_k(completions, [case] * 3, stub_reward), 1.0)

    def test_consensus_fails_when_majority_is_wrong(self):
        case = {"id": "c1", "requires_expand_contract": True}
        completions = ["bad plan", "bad plan", "GOOD plan"]
        self.assertEqual(consensus_at_k(completions, [case] * 3, stub_reward), 0.0)

    def test_evaluate_generations(self):
        cases = [{"id": "c1"}, {"id": "c2"}]
        result = evaluate_generations(["GOOD", "bad"], cases, stub_reward)
        self.assertEqual(result["count"], 2)
        self.assertAlmostEqual(result["pass@1"], 0.5)
        self.assertAlmostEqual(result["mean_accuracy"], 0.5)
        self.assertAlmostEqual(result["mean_format"], 1.0)


if __name__ == "__main__":
    unittest.main()
