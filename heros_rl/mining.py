# SPDX-License-Identifier: MIT
"""FastText-style corpus mining for migration/reasoning domains.

This is a dependency-free hashed n-gram logistic classifier, not the official
fastText library. It preserves the mining loop: train on a seed corpus, scan
candidate pages, keep domains where more than ``threshold`` of pages score
positive, inspect URL path patterns, add the strongest positives to the seed
corpus, and repeat for a fixed number of iterations.
"""

from __future__ import annotations

import math
import random
import re
import zlib
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence
from urllib.parse import urlparse

TOKEN_RE = re.compile(r"[a-z0-9_]+")
SEGMENT_ID_RE = re.compile(r"^(?:\d+|[0-9a-f]{8,})$", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def _feature_indices(text: str, dimension: int, max_ngram: int = 2) -> list[int]:
    tokens = tokenize(text)
    indices: list[int] = []
    for ngram in range(1, max_ngram + 1):
        for start in range(0, len(tokens) - ngram + 1):
            joined = " ".join(tokens[start:start + ngram])
            indices.append(zlib.crc32(joined.encode("utf-8")) % dimension)
    # Binary features: a word appearing twice is still one feature.
    return list(dict.fromkeys(indices))


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp = math.exp(value)
    return exp / (1.0 + exp)


class HashedNgramClassifier:
    """Binary logistic regression over hashed 1–2 gram features."""

    def __init__(
        self,
        dimension: int = 512,
        epochs: int = 12,
        learning_rate: float = 0.2,
        l2: float = 1e-4,
        seed: int = 7,
        max_ngram: int = 2,
    ) -> None:
        if not isinstance(dimension, int) or dimension < 1:
            raise ValueError("dimension must be a positive integer")
        if not isinstance(epochs, int) or epochs < 1:
            raise ValueError("epochs must be a positive integer")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if l2 < 0:
            raise ValueError("l2 must be non-negative")
        if max_ngram not in (1, 2):
            raise ValueError("max_ngram must be 1 or 2")
        self.dimension = dimension
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.l2 = l2
        self.seed = seed
        self.max_ngram = max_ngram
        self.weights = [0.0] * dimension
        self.bias = 0.0

    def fit(self, examples: Sequence[tuple[str, int]]) -> "HashedNgramClassifier":
        if not examples:
            raise ValueError("training examples must not be empty")
        labels = {int(label) for _, label in examples}
        if labels != {0, 1}:
            raise ValueError("training examples must contain both labels 0 and 1")

        rows = [
            (_feature_indices(text, self.dimension, self.max_ngram), int(label))
            for text, label in examples
        ]
        rng = random.Random(self.seed)
        for _ in range(self.epochs):
            rng.shuffle(rows)
            for indices, label in rows:
                score = self.bias + sum(self.weights[index] for index in indices)
                probability = _sigmoid(score)
                gradient = probability - label
                self.bias -= self.learning_rate * gradient
                for index in indices:
                    current = self.weights[index]
                    self.weights[index] = current - self.learning_rate * (
                        gradient + self.l2 * current
                    )
        return self

    def predict_proba(self, text: str) -> float:
        indices = _feature_indices(text, self.dimension, self.max_ngram)
        return _sigmoid(self.bias + sum(self.weights[index] for index in indices))

    def predict(self, text: str, threshold: float = 0.5) -> int:
        return int(self.predict_proba(text) >= threshold)

    def to_dict(self) -> dict:
        return {
            "dimension": self.dimension,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "l2": self.l2,
            "seed": self.seed,
            "max_ngram": self.max_ngram,
            "bias": self.bias,
            "weights": list(self.weights),
        }

    @classmethod
    def from_dict(cls, value: dict) -> "HashedNgramClassifier":
        classifier = cls(
            dimension=int(value["dimension"]),
            epochs=int(value["epochs"]),
            learning_rate=float(value["learning_rate"]),
            l2=float(value["l2"]),
            seed=int(value["seed"]),
            max_ngram=int(value["max_ngram"]),
        )
        weights = [float(item) for item in value["weights"]]
        if len(weights) != classifier.dimension:
            raise ValueError("weight vector length does not match dimension")
        classifier.weights = weights
        classifier.bias = float(value["bias"])
        return classifier


@dataclass(frozen=True)
class SeedDocument:
    text: str
    label: int


@dataclass(frozen=True)
class CorpusDocument:
    domain: str
    url: str
    text: str


@dataclass(frozen=True)
class DomainResult:
    domain: str
    total: int
    positives: int
    fraction: float
    path_patterns: tuple[str, ...]
    sample_urls: tuple[str, ...]


@dataclass(frozen=True)
class IterationResult:
    iteration: int
    seed_count: int
    domains_scanned: int
    selected_domains: tuple[DomainResult, ...]
    added_documents: int
    threshold: float


@dataclass(frozen=True)
class MiningRun:
    iterations: tuple[IterationResult, ...]
    final_seed_count: int
    classifier: HashedNgramClassifier = field(repr=False)

    def to_dict(self) -> dict:
        return {
            "threshold": self.iterations[0].threshold if self.iterations else None,
            "final_seed_count": self.final_seed_count,
            "iterations": [
                {
                    "iteration": result.iteration,
                    "seed_count": result.seed_count,
                    "domains_scanned": result.domains_scanned,
                    "added_documents": result.added_documents,
                    "selected_domains": [
                        {
                            "domain": domain.domain,
                            "total": domain.total,
                            "positives": domain.positives,
                            "fraction": round(domain.fraction, 6),
                            "path_patterns": list(domain.path_patterns),
                            "sample_urls": list(domain.sample_urls),
                        }
                        for domain in result.selected_domains
                    ],
                }
                for result in self.iterations
            ],
        }


def path_pattern(url: str) -> str:
    """Collapse numeric/hex path segments to ``:id`` for pattern review."""
    segments = [segment for segment in urlparse(url).path.split("/") if segment]
    if not segments:
        return "/"
    collapsed = [
        ":id" if SEGMENT_ID_RE.match(segment) else segment
        for segment in segments[:3]
    ]
    return "/" + "/".join(collapsed)


def _domain_results(
    corpus: Sequence[CorpusDocument],
    classifier: HashedNgramClassifier,
    threshold: float,
    min_pages: int,
) -> tuple[DomainResult, ...]:
    grouped: dict[str, list[tuple[CorpusDocument, float]]] = {}
    for document in corpus:
        probability = classifier.predict_proba(document.text)
        grouped.setdefault(document.domain, []).append((document, probability))

    results = []
    for domain, rows in grouped.items():
        if len(rows) < min_pages:
            continue
        positive_rows = [(document, score) for document, score in rows if score >= 0.5]
        fraction = len(positive_rows) / len(rows)
        if fraction <= threshold:
            continue
        patterns: dict[str, int] = {}
        for document, _ in positive_rows:
            pattern = path_pattern(document.url)
            patterns[pattern] = patterns.get(pattern, 0) + 1
        top_patterns = tuple(
            pattern for pattern, _ in sorted(patterns.items(), key=lambda item: (-item[1], item[0]))[:3]
        )
        sample_urls = tuple(
            document.url
            for document, _ in sorted(positive_rows, key=lambda item: (-item[1], item[0].url))[:3]
        )
        results.append(
            DomainResult(
                domain=domain,
                total=len(rows),
                positives=len(positive_rows),
                fraction=fraction,
                path_patterns=top_patterns,
                sample_urls=sample_urls,
            )
        )
    results.sort(key=lambda result: (-result.fraction, result.domain))
    return tuple(results)


def mine_domains(
    seeds: Sequence[SeedDocument],
    corpus: Sequence[CorpusDocument],
    *,
    iterations: int = 4,
    threshold: float = 0.1,
    min_pages: int = 2,
    max_new_per_domain: int = 3,
    classifier_factory: Callable[[], HashedNgramClassifier] | None = None,
) -> MiningRun:
    if not seeds:
        raise ValueError("seeds must not be empty")
    if not corpus:
        raise ValueError("corpus must not be empty")
    if not isinstance(iterations, int) or iterations < 1:
        raise ValueError("iterations must be a positive integer")
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be between 0 and 1")
    if not isinstance(min_pages, int) or min_pages < 1:
        raise ValueError("min_pages must be a positive integer")
    if not isinstance(max_new_per_domain, int) or max_new_per_domain < 1:
        raise ValueError("max_new_per_domain must be a positive integer")

    factory = classifier_factory or HashedNgramClassifier
    current = list(seeds)
    seen_texts = {seed.text for seed in current}
    history: list[IterationResult] = []
    classifier: HashedNgramClassifier | None = None

    for iteration in range(1, iterations + 1):
        classifier = factory().fit([(seed.text, seed.label) for seed in current])
        selected = _domain_results(corpus, classifier, threshold, min_pages)
        added = 0
        for domain_result in selected:
            added_for_domain = 0
            for document in corpus:
                if document.domain != domain_result.domain:
                    continue
                if classifier.predict_proba(document.text) < 0.5:
                    continue
                if document.text in seen_texts:
                    continue
                current.append(SeedDocument(text=document.text, label=1))
                seen_texts.add(document.text)
                added += 1
                added_for_domain += 1
                if added_for_domain >= max_new_per_domain:
                    break
        history.append(
            IterationResult(
                iteration=iteration,
                seed_count=len(current) - added,
                domains_scanned=len({document.domain for document in corpus}),
                selected_domains=selected,
                added_documents=added,
                threshold=threshold,
            )
        )

    assert classifier is not None
    return MiningRun(
        iterations=tuple(history),
        final_seed_count=len(current),
        classifier=classifier,
    )
