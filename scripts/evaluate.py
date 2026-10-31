#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Sample a GRPO policy and compute Pass@1 / cons@k on the HEROS-RL dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError as error:  # pragma: no cover
    print(f"Install evaluation dependencies first: {error}", file=sys.stderr)
    raise SystemExit(2)

from heros_rl.eval import evaluate_by_group  # noqa: E402
from heros_rl.forge import ForgeClient  # noqa: E402
from heros_rl.prompts import SYSTEM_PROMPT  # noqa: E402
from heros_rl.rewards import RewardFunction  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    )
    parser.add_argument("--adapter", default=None)
    parser.add_argument(
        "--dataset", default=str(ROOT / "data" / "heros_migration_500.jsonl")
    )
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--output", default=str(ROOT / "outputs" / "eval_results.json")
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("No CUDA device; run on a GPU runtime.", file=sys.stderr)
        return 2

    dataset = load_dataset("json", data_files=args.dataset, split="train")
    cases = list(dataset)
    if args.limit:
        cases = cases[: args.limit]

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    samples_by_case: dict[str, list[str]] = {}
    cases_by_id = {case["id"]: case for case in cases}

    for case in cases:
        prompt = (
            f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\n{case['prompt']}\n<|assistant|>\n"
        )
        encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=0.95,
                num_return_sequences=args.k,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated = tokenizer.batch_decode(
            outputs[:, encoded["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )
        samples_by_case[case["id"]] = generated
        print(f"sampled {case['id']} ({len(generated)} completions)")

    reward_fn = RewardFunction(forge_client=ForgeClient())
    metrics = evaluate_by_group(samples_by_case, cases_by_id, reward_fn)
    payload = {
        "model": args.model,
        "adapter": args.adapter,
        "k": args.k,
        "cases": len(cases),
        **metrics,
        "samples": samples_by_case,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
