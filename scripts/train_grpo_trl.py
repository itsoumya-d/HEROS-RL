#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""GRPO training with TRL. Intended for Kaggle/Colab T4 GPUs.

Example:
  python scripts/train_grpo_trl.py \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --group-size 8 --epochs 1 \
    --output outputs/grpo-g8

The reward is deterministic: format tags + SQL safety audit + optional forge
risk analysis. No critic/value network is trained.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer
except ImportError as error:  # pragma: no cover
    print(
        "Training dependencies are missing. Install with:\n"
        "  python -m pip install -e .[train]\n"
        f"Import error: {error}",
        file=sys.stderr,
    )
    raise SystemExit(2)

from heros_rl.forge import ForgeClient  # noqa: E402
from heros_rl.rewards import RewardFunction  # noqa: E402
from heros_rl.trl_rewards import make_reward_funcs  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    )
    parser.add_argument(
        "--dataset", default=str(ROOT / "data" / "heros_migration_500.jsonl")
    )
    parser.add_argument("--output", default=str(ROOT / "outputs" / "grpo"))
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--max-completion-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--push-to-hub", default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print(
            "No CUDA device. Use a Kaggle/Colab T4 runtime or an equivalent GPU.",
            file=sys.stderr,
        )
        return 2

    dataset = load_dataset("json", data_files=args.dataset, split="train")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    reward_fn = RewardFunction(forge_client=ForgeClient())
    format_reward_func, accuracy_reward_func = make_reward_funcs(reward_fn)

    training_args = GRPOConfig(
        output_dir=args.output,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        num_generations=args.group_size,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        beta=args.beta,
        logging_steps=5,
        save_steps=100,
        bf16=True,
        gradient_checkpointing=True,
        report_to="none",
        push_to_hub=bool(args.push_to_hub),
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[format_reward_func, accuracy_reward_func],
        args=training_args,
        train_dataset=dataset,
        peft_config=lora,
    )
    trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)

    if args.push_to_hub:
        trainer.push_to_hub(args.push_to_hub)

    print(f"saved GRPO adapter to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
