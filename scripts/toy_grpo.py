#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Toy GRPO ablation used to validate the critic-free update.

This is not an LLM experiment. It is a 12-parameter contextual bandit where the
optimal action differs by hazard context. It checks that group-relative
advantages actually improve the policy, and how the group size affects sample
efficiency. The real 1.5B training script is train_grpo_trl.py / unsloth.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    import torch
except ImportError:  # pragma: no cover
    print(
        "PyTorch is required: python -m pip install torch",
        file=sys.stderr,
    )
    raise SystemExit(2)

from heros_rl.grpo import group_relative_advantages, grpo_loss  # noqa: E402

# Four hazard contexts and three possible actions. The safe action differs by
# context: [0, 1, 2, 0]. This prevents a single global action from solving all
# contexts and forces the policy to condition on the context.
REWARDS = torch.tensor(
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ]
)
CONTEXTS = torch.arange(REWARDS.shape[0])


def policy_accuracy(weights: torch.Tensor) -> float:
    best = torch.argmax(weights, dim=1)
    safe = torch.argmax(REWARDS, dim=1)
    return float((best == safe).float().mean())


def run_experiment(
    group_size: int,
    seed: int,
    steps: int = 300,
    lr: float = 0.1,
    beta: float = 0.01,
    inner_epochs: int = 2,
) -> dict:
    torch.manual_seed(seed)
    weights = torch.zeros(REWARDS.shape[0], REWARDS.shape[1], requires_grad=True)
    reference = weights.detach().clone()
    optimizer = torch.optim.Adam([weights], lr=lr)

    initial_accuracy = policy_accuracy(weights.detach())
    initial_reward = float(REWARDS.mean())
    history: list[float] = []

    for _ in range(steps):
        with torch.no_grad():
            distribution = torch.distributions.Categorical(
                logits=weights.detach()[CONTEXTS]
            )
            sampled = distribution.sample((group_size,)).transpose(0, 1)
            contexts_expanded = CONTEXTS.repeat_interleave(group_size)
            actions = sampled.reshape(-1)
            flat_old = torch.distributions.Categorical(
                logits=weights.detach()[contexts_expanded]
            ).log_prob(actions)
            flat_ref = torch.distributions.Categorical(
                logits=reference[contexts_expanded]
            ).log_prob(actions)
            rewards = (
                REWARDS[contexts_expanded]
                .gather(1, actions.unsqueeze(-1))
                .squeeze(-1)
            )
            advantages = []
            for context_index in range(REWARDS.shape[0]):
                group = rewards[
                    context_index * group_size : (context_index + 1) * group_size
                ]
                advantages.extend(group_relative_advantages(group.tolist()))
            advantage_tensor = torch.tensor(
                advantages, dtype=torch.float32
            ).unsqueeze(-1)
            history.append(float(rewards.mean()))

        flat_old = flat_old.unsqueeze(-1)
        flat_ref = flat_ref.unsqueeze(-1)

        for _ in range(inner_epochs):
            distribution = torch.distributions.Categorical(
                logits=weights[contexts_expanded]
            )
            current = distribution.log_prob(actions).unsqueeze(-1)
            loss = grpo_loss(
                current,
                flat_old,
                flat_ref,
                advantage_tensor,
                beta=beta,
            )
            optimizer.zero_grad()
            loss.total.backward()
            optimizer.step()

    final_accuracy = policy_accuracy(weights.detach())
    final_reward = statistics.mean(history[-20:]) if history else initial_reward
    return {
        "group_size": group_size,
        "seed": seed,
        "initial_accuracy": initial_accuracy,
        "final_accuracy": final_accuracy,
        "initial_reward": initial_reward,
        "final_reward": final_reward,
        "steps": steps,
    }


def summarize(runs: list[dict]) -> list[dict]:
    grouped: dict[int, list[dict]] = {}
    for run in runs:
        grouped.setdefault(run["group_size"], []).append(run)
    summary = []
    for group_size in sorted(grouped):
        values = grouped[group_size]
        summary.append(
            {
                "group_size": group_size,
                "runs": len(values),
                "final_accuracy_mean": statistics.mean(
                    run["final_accuracy"] for run in values
                ),
                "final_accuracy_std": statistics.pstdev(
                    run["final_accuracy"] for run in values
                ),
                "final_reward_mean": statistics.mean(
                    run["final_reward"] for run in values
                ),
                "final_reward_std": statistics.pstdev(
                    run["final_reward"] for run in values
                ),
            }
        )
    return summary


def markdown_table(summary: list[dict]) -> str:
    lines = [
        "| Group size G | Runs | Final safe-action accuracy | Final mean reward |",
        "|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['group_size']} | {row['runs']} | "
            f"{row['final_accuracy_mean']:.3f} ± {row['final_accuracy_std']:.3f} | "
            f"{row['final_reward_mean']:.3f} ± {row['final_reward_std']:.3f} |"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument(
        "--group-sizes",
        type=int,
        nargs="+",
        default=[1, 4, 8, 16, 32],
    )
    parser.add_argument(
        "--out",
        default=str(ROOT / "experiments" / "toy_grpo_results.json"),
    )
    args = parser.parse_args()

    runs = []
    for group_size in args.group_sizes:
        for seed in range(args.seeds):
            runs.append(
                run_experiment(group_size, seed, steps=args.steps)
            )

    summary = summarize(runs)
    output = {
        "task": "contextual-bandit-grpo-validation",
        "note": "Toy validation of the GRPO update; not an LLM result.",
        "steps": args.steps,
        "seeds": args.seeds,
        "runs": runs,
        "summary": summary,
    }
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    print(markdown_table(summary))
    print(f"\nwrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
