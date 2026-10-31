#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Faster single-GPU GRPO with Unsloth + TRL (Kaggle/Colab T4).

Unsloth requires CUDA. Install with:
  pip install "unsloth[cu121-torch250] @ git+https://github.com/unslothai/unsloth.git"
  pip install -e .[train]
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
    from trl import GRPOConfig, GRPOTrainer
    from unsloth import FastLanguageModel
except ImportError as error:  # pragma: no cover
    print(f"Install Unsloth + TRL first. Import error: {error}", file=sys.stderr)
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
    parser.add_argument("--output", default=str(ROOT / "outputs" / "grpo-unsloth"))
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--push-to-hub", default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("Unsloth requires CUDA; no GPU detected.", file=sys.stderr)
        return 2

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        dtype=None,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    dataset = load_dataset("json", data_files=args.dataset, split="train")
    reward_fn = RewardFunction(forge_client=ForgeClient())
    format_reward_func, accuracy_reward_func = make_reward_funcs(reward_fn)

    training_args = GRPOConfig(
        output_dir=args.output,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_generations=args.group_size,
        max_prompt_length=1024,
        max_completion_length=1024,
        num_train_epochs=args.epochs,
        learning_rate=5e-6,
        beta=0.04,
        logging_steps=5,
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
