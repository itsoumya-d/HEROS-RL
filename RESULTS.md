# HEROS-RL Results

## Environment

| Item | Value |
|---|---|
| Date | 2026-10-31 |
| OS | macOS (Apple Silicon) |
| Python | 3.14.0 |
| PyTorch | 2.15.0 (CPU) |
| GPU | None |
| Dataset | `data/heros_migration_500.jsonl` |
| Forge binary | Not available during this run; static SQL audit used |

## Unit tests

```
python -m unittest discover -s tests -v
```

**40 tests passed**, covering:

- group-relative advantages (known values, zero-mean, constant-group zero,
  empty input),
- GRPO loss (on-policy zero loss, positive advantage gradient direction,
  clipping, k3 KL value, padding mask),
- format reward and tag parsing,
- `MIGRATION:` / `ROLLBACK:` extraction, quoted-semicolon splitting,
- static accuracy for safe, expand-contract, direct-destructive, single-step,
  and non-SQL answers,
- forge reward integration with a fake forge returning HIGH/decision-required,
- Pass@1, cons@k, and grouped evaluation,
- dataset size, determinism, labels, prompts, and committed JSONL equality,
- hashed-ngram mining, four-iteration domain discovery, path patterns, and
  deterministic run output.

## Dataset reproducibility

```
python scripts/build_dataset.py --check
```

The committed JSONL matches a fresh generation byte-for-byte.

| Metric | Value |
|---|---:|
| Cases | 500 |
| Families | 10 × 50 |
| Data-loss cases | 150 |
| Expand-and-contract cases | 300 |
| SAFE / NOTABLE / HIGH / CRITICAL | 150 / 50 / 200 / 100 |

## Toy GRPO ablation

`scripts/toy_grpo.py` uses a 12-parameter contextual bandit with four hazard
contexts and three actions. The optimal action differs by context. This isolates
the GRPO update from model quality and tests whether group-relative advantages
learn.

Command:

```bash
python scripts/toy_grpo.py --seeds 5 --steps 300 --group-sizes 1 4 8 16 32
```

| Group size G | Runs | Final safe-action accuracy | Final mean reward |
|---:|---:|---:|---:|
| 1 | 5 | 0.500 ± 0.000 | 0.338 ± 0.016 |
| 4 | 5 | 1.000 ± 0.000 | 0.997 ± 0.004 |
| 8 | 5 | 1.000 ± 0.000 | 0.998 ± 0.001 |
| 16 | 5 | 1.000 ± 0.000 | 0.999 ± 0.000 |
| 32 | 5 | 1.000 ± 0.000 | 0.999 ± 0.001 |

Interpretation:

- `G = 1` is the no-group control. Every advantage is zero by construction, so
  the policy receives no learning signal. Its 0.500 accuracy is the argmax
  tie-break of the zero-initialised policy, not learning.
- `G = 4` is sufficient to solve the toy task.
- `G = 8, 16, 32` converge to marginally higher mean reward, with `G = 16`
  highest in this run. The differences are within a few thousandths and should
  not be over-interpreted.
- This validates the implementation and the group-size mechanism. It is **not**
  evidence of 1.5B model performance on SQL migrations.

Raw data: `experiments/toy_grpo_results.json`.

## Forge reward integration

The reward path was tested with a fake forge client returning
`risk_tier=HIGH`, `decision_required=true`. It correctly:

- records `forge_score = 0.25`,
- keeps a safe expand-contract plan at 1.0 accuracy,
- keeps a direct rename at 0.0 accuracy,
- leaves the format reward binary.

The real forge binary was not present in this environment. Set
`HEROS_FORGE_BIN` to the compiled forge executable to enable the same path
against the deterministic Rust/Zero implementation.

## 1.5B GRPO runs — status: NOT EXECUTED

No CUDA device is available in this environment. No Pass@1 or cons@64 numbers
are reported here, because fabricating them would defeat the purpose of the
deterministic reward stack.

The reproducible commands are:

```bash
# TRL
python scripts/train_grpo_trl.py \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --group-size 8 --epochs 1 --output outputs/grpo-g8

# Unsloth
python scripts/train_grpo_unsloth.py \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --group-size 8 --output outputs/grpo-unsloth-g8

# Evaluation
python scripts/evaluate.py \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --adapter outputs/grpo-g8 \
  --k 64
```

After a run, `scripts/evaluate.py` writes Pass@1, cons@64, mean accuracy, mean
format reward, and per-case samples. Add those numbers to this file rather than
estimating them.

## Four-iteration corpus mining (synthetic fixture)

`python scripts/mine_corpus.py` discovers `docs.example.com` (4/8 positive) and
`proofs.example.org` (3/8 positive) in iteration 1, extracts
`/docs/migrations/:id` and `/proofs/:id`, grows the seed set from 4 to 11, and
rejects `blog.example.com` (1/12 = 8.3%) and `shop.example.com` (0/8). This is a
synthetic fixture with a dependency-free hashed-ngram classifier, not a web
crawl or the official fastText library.

## What would count as success

The project's empirical target is:

- single-sample Pass@1 at or above the SFT/reference baseline,
- cons@64 improving over Pass@1,
- a group-size curve that is flat or improving from G=4 to G=32,
- 100% format compliance on the held-out split.

Those are targets, not results. The current evidence is the algorithm and reward
validation above.
