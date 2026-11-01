# HEROS-RL

Critic-free **Group Relative Policy Optimization (GRPO)** for rewriting
hazardous PostgreSQL schema migrations into zero-downtime expand-and-contract
plans.

HEROS-RL replaces the PPO value network with a group baseline: sample `G`
completions per migration prompt, normalise their deterministic rewards inside
the group, and optimise the clipped surrogate with an explicit KL penalty to a
frozen reference policy.

This repository contains the algorithm, the deterministic reward stack, a
500-case dataset, evaluation harness, and TRL/Unsloth training entry points for
`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`.

## Research basis

- DeepSeekMath — GRPO (arXiv:2402.03300)
- DeepSeek-R1 — rule-based format/accuracy rewards (arXiv:2501.12948)
- Demystifying GRPO — group-relative advantage as a U-statistic (arXiv:2603.01162)
- Reflexion — verbal self-correction protocol reused by the reward diagnostics
  (arXiv:2303.11366)

## What is implemented

| Component | Status |
|---|---|
| Group-relative advantage `A_i = (r_i - mean(r)) / (std(r) + eps)` | Implemented, unit tested |
| Clipped GRPO surrogate + k3 KL penalty | Implemented in PyTorch, unit tested |
| Strict ` thinking/<answer>` format reward | Implemented |
| SQL safety audit (expand/contract and rollback) | Implemented, unit tested |
| Optional forge-binary risk analysis | Implemented (`HEROS_FORGE_BIN`) |
| 500-case deterministic dataset | Generated and committed |
| Pass@1 / cons@k evaluation | Implemented |
| TRL + Unsloth 1.5B training scripts | Provided (GPU required) |
| 1.5B empirical results | Not run in this environment; see [RESULTS.md](RESULTS.md) |

## Install

```bash
python -m pip install -e .
# GPU training:
python -m pip install -e ".[train]"
```

No runtime dependency is needed for the reward, dataset, or evaluation logic.

## Quick start

```python
from heros_rl import RewardFunction, group_relative_advantages

advantages = group_relative_advantages([0.0, 0.4, 0.9, 1.0])
reward = RewardFunction()
print(reward(completion, case).to_dict())
```

Build or verify the dataset:

```bash
python scripts/build_dataset.py
python scripts/build_dataset.py --check
```

Run the tests:

```bash
python -m unittest discover -s tests -v
```

## Reward design

Every completion is scored on two deterministic axes:

1. **Format reward** — reasoning must be enclosed in ` thinking...</think>` and
   the executable plan in `<answer>...</answer>`. Both must be non-empty.
2. **Accuracy reward** — the answer must contain `MIGRATION:` and `ROLLBACK:`
   sections. The SQL audit:
   - parses statements outside quotes and comments,
   - rejects unknown statement starts,
   - rejects `DROP`, `TRUNCATE`, `DELETE FROM`, type narrowing,
     `SET NOT NULL`, direct `RENAME`, and direct `PRIMARY KEY` additions as the
     first statement when the case requires expand-and-contract,
   - awards partial credit for a multi-phase plan, a safe first step, and a
     non-empty rollback.

When `HEROS_FORGE_BIN` or `forge` on `PATH` is available, the reward also calls
`forge analyze --from <old> --to <target>` and uses its `risk_tier`,
`has_data_loss`, and `decision_required` fields as a second deterministic
signal. The forge call is optional; the static audit is the reproducible
fallback.

A future PostgreSQL sandbox adapter can execute the migration and rollback in a
transaction. It is intentionally not enabled by default so training rewards stay
deterministic and dependency-free.

## Dataset

`data/heros_migration_500.jsonl` contains 500 prompts across 10 families:

| Family | Cases | Data loss | Expand/contract |
|---|---:|---:|---:|
| drop_column | 50 | yes | yes |
| drop_table | 50 | yes | yes |
| type_narrowing | 50 | yes | yes |
| set_not_null | 50 | no | yes |
| rename_column | 50 | no | yes |
| add_primary_key | 50 | no | yes |
| add_nullable_column | 50 | no | no |
| add_default_column | 50 | no | no |
| widen_type | 50 | no | no |
| create_table | 50 | no | no |

Totals: 150 data-loss cases, 300 requiring expand-and-contract, 150 SAFE, 50
NOTABLE, 200 HIGH, 100 CRITICAL. The generator is deterministic and CI verifies
the committed file byte-for-byte.

A Hugging Face dataset card is in `data/dataset_card.md`. Publishing requires
an authenticated Hugging Face account; it is not performed automatically.

## Four-iteration corpus mining

`heros_rl/mining.py` implements a dependency-free fastText-style loop: hashed
1–2 gram logistic regression, domain-level positive-fraction filtering
(`> 10%`), URL path-pattern review, seed expansion, and four iterations.

```bash
python scripts/mine_corpus.py
```

The committed fixture (`experiments/mining_results.json`) is synthetic, not a
web crawl. It discovers `docs.example.com` (4/8 positive) and
`proofs.example.org` (3/8 positive), extracts `/docs/migrations/:id` and
`/proofs/:id`, and grows the seed set from 4 to 11 documents. This is the
official fastText library's *loop*, not the official fastText implementation.

## Training

The default model is `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`.

### TRL (Kaggle/Colab T4)

```bash
python scripts/train_grpo_trl.py \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --group-size 8 \
  --epochs 1 \
  --output outputs/grpo-g8
```

### Unsloth (single GPU, 4-bit)

```bash
python scripts/train_grpo_unsloth.py \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --group-size 8 \
  --output outputs/grpo-unsloth-g8
```

Both scripts use LoRA and TRL's `GRPOTrainer`. The reward callbacks are in
`heros_rl/trl_rewards.py`; no critic model is instantiated.

Group-size ablation: run with `--group-size 4`, `8`, `16`, and `32` and keep
the evaluation JSON for each run.

## Evaluation

Sample `k` completions per case and compute single-sample Pass@1 plus majority
vote consensus:

```bash
python scripts/evaluate.py \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --adapter outputs/grpo-g8 \
  --k 64
```

To score pre-generated completions without a GPU:

```bash
python scripts/score_completions.py --completions samples.jsonl
```

## Toy validation

`scripts/toy_grpo.py` trains a 12-parameter contextual bandit with the same GRPO
update. This is an algorithmic check, not an LLM result. With 5 seeds and 300
steps:

| Group size G | Final safe-action accuracy | Final mean reward |
|---:|---:|---:|
| 1 | 0.500 ± 0.000 | 0.338 ± 0.016 |
| 4 | 1.000 ± 0.000 | 0.997 ± 0.004 |
| 8 | 1.000 ± 0.000 | 0.998 ± 0.001 |
| 16 | 1.000 ± 0.000 | 0.999 ± 0.000 |
| 32 | 1.000 ± 0.000 | 0.999 ± 0.001 |

`G = 1` has zero within-group variance, so the advantage is zero and the policy
does not learn — the expected failure mode of a group baseline with no group.
`G >= 4` solves the toy task. Full details are in [RESULTS.md](RESULTS.md).

## Status and limitations

- The 1.5B GRPO training runs are **not** executed in this repository's current
  environment (no CUDA GPU). The scripts and reward stack are ready; the
  empirical 1.5B table will be filled only after GPU runs.
- The static SQL audit is a deterministic proxy for a PostgreSQL sandbox. It
  rejects dangerous first statements but does not prove semantic equivalence.
- The `forge` analysis is optional; without the binary the static audit is used.
- The dataset is synthetic and balanced for training, not a sample of production
  migrations.

## Repository layout

```
heros_rl/          GRPO math, rewards, forge bridge, dataset, evaluation
scripts/           dataset generation, TRL/Unsloth training, eval, toy ablation
data/              500-case JSONL + Hugging Face dataset card
tests/             40 stdlib unit tests (torch tests skip without torch)
experiments/       toy GRPO ablation results
RESULTS.md         measured tests, toy ablation, GPU-run status
```

## License

MIT. Copyright (c) 2024-2026 Soumya Debnath.
