# SPDX-License-Identifier: MIT
import math
import unittest

from heros_rl.grpo import group_relative_advantages


class GroupRelativeAdvantageTests(unittest.TestCase):
    def test_known_values(self):
        advantages = group_relative_advantages([1.0, 2.0, 3.0, 4.0])
        expected = [
            (value - 2.5) / (math.sqrt(1.25) + 1e-4)
            for value in (1.0, 2.0, 3.0, 4.0)
        ]
        for actual, wanted in zip(advantages, expected):
            self.assertAlmostEqual(actual, wanted, places=8)

    def test_mean_is_zero(self):
        advantages = group_relative_advantages([0.2, 0.9, 0.4, 0.8, 0.1])
        self.assertAlmostEqual(sum(advantages), 0.0, places=8)

    def test_constant_group_has_zero_advantages(self):
        self.assertEqual(group_relative_advantages([0.7, 0.7, 0.7]), [0.0, 0.0, 0.0])

    def test_empty_group_rejected(self):
        with self.assertRaises(ValueError):
            group_relative_advantages([])


if __name__ == "__main__":
    unittest.main()
