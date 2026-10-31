# SPDX-License-Identifier: MIT
import unittest

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from heros_rl.grpo import grpo_loss


@unittest.skipUnless(HAS_TORCH, "PyTorch is not installed")
class GRPOLossTests(unittest.TestCase):
    def test_on_policy_balanced_advantages_have_zero_loss(self):
        logprobs = torch.zeros(2, 3, requires_grad=True)
        advantages = torch.tensor([1.0, -1.0])
        result = grpo_loss(logprobs, torch.zeros_like(logprobs), torch.zeros_like(logprobs), advantages)
        self.assertAlmostEqual(float(result.total), 0.0, places=6)
        self.assertAlmostEqual(float(result.policy), 0.0, places=6)
        self.assertAlmostEqual(float(result.kl), 0.0, places=6)
        self.assertEqual(result.ratio_mean, 1.0)
        self.assertEqual(result.clip_fraction, 0.0)

    def test_positive_advantage_increases_probability(self):
        logprobs = torch.zeros(1, 1, requires_grad=True)
        result = grpo_loss(
            logprobs,
            torch.zeros_like(logprobs),
            torch.zeros_like(logprobs),
            torch.tensor([1.0]),
        )
        self.assertAlmostEqual(float(result.total), -1.0, places=6)
        result.total.backward()
        self.assertLess(float(logprobs.grad[0, 0]), 0.0)

    def test_ratio_is_clipped(self):
        old = torch.zeros(1, 1)
        logprobs = torch.full((1, 1), 0.6931471805599453)  # ratio = 2
        result = grpo_loss(
            logprobs,
            old,
            torch.zeros_like(old),
            torch.tensor([1.0]),
            clip_eps=0.2,
        )
        self.assertAlmostEqual(float(result.policy), -1.2, places=5)
        self.assertEqual(result.ratio_mean, 2.0)
        self.assertEqual(result.clip_fraction, 1.0)

    def test_kl_penalty_matches_k3_estimator(self):
        logprobs = torch.zeros(1, 1)
        ref = torch.full((1, 1), 0.6931471805599453)
        result = grpo_loss(
            logprobs,
            torch.zeros_like(logprobs),
            ref,
            torch.tensor([0.0]),
            beta=1.0,
        )
        expected = 2.0 - 0.6931471805599453 - 1.0
        self.assertAlmostEqual(float(result.kl), expected, places=5)
        self.assertAlmostEqual(float(result.total), expected, places=5)

    def test_mask_ignores_padding(self):
        logprobs = torch.zeros(1, 3, requires_grad=True)
        mask = torch.tensor([[1.0, 1.0, 0.0]])
        result = grpo_loss(
            logprobs,
            torch.zeros_like(logprobs),
            torch.zeros_like(logprobs),
            torch.tensor([1.0]),
            mask=mask,
        )
        self.assertAlmostEqual(float(result.total), -1.0, places=6)
        self.assertEqual(result.token_count, 2.0)


if __name__ == "__main__":
    unittest.main()
