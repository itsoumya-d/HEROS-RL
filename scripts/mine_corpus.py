# SPDX-License-Identifier: MIT
"""Run the four-iteration corpus miner on a synthetic seed/corpus fixture.

This demonstrates the domain-discovery loop with a deterministic hashed-ngram
classifier. It is not a crawl of the public web and not the official fastText
implementation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from heros_rl.mining import CorpusDocument, SeedDocument, mine_domains
POSITIVE = "postgres migration schema expand contract rollback sql alter table backfill zero spec"
NEGATIVE = "recipe travel sports movie music garden cooking holiday"


def seeds() -> list[SeedDocument]:
    return [
        SeedDocument(POSITIVE, 1),
        SeedDocument(POSITIVE + " theorem proof", 1),
        SeedDocument(NEGATIVE, 0),
        SeedDocument(NEGATIVE + " kitchen", 0),
    ]


def corpus() -> list[CorpusDocument]:
    documents: list[CorpusDocument] = []
    for index in range(8):
        text = POSITIVE if index < 4 else NEGATIVE
        documents.append(CorpusDocument(
            "docs.example.com",
            f"https://docs.example.com/docs/migrations/{index}",
            f"{text} page {index}",
        ))
    for index in range(12):
        text = POSITIVE if index == 0 else NEGATIVE
        documents.append(CorpusDocument(
            "blog.example.com",
            f"https://blog.example.com/posts/{index}",
            f"{text} article {index}",
        ))
    for index in range(8):
        text = POSITIVE if index < 3 else NEGATIVE
        documents.append(CorpusDocument(
            "proofs.example.org",
            f"https://proofs.example.org/proofs/{index}",
            f"{text} theorem proof {index}",
        ))
    for index in range(8):
        documents.append(CorpusDocument(
            "shop.example.com",
            f"https://shop.example.com/products/{index}",
            f"{NEGATIVE} product {index}",
        ))
    return documents


def main() -> None:
    run = mine_domains(seeds(), corpus(), iterations=4, threshold=0.1, min_pages=2)
    payload = {
        "fixture": "synthetic-mining-corpus",
        "not_a_web_crawl": True,
        "classifier": "hashed 1-2 gram logistic regression (fastText-style)",
        **run.to_dict(),
    }
    output = ROOT / "experiments" / "mining_results.json"
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
