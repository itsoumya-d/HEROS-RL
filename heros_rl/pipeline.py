"""Deterministic data pipeline contracts, feature sanitization, and unsupervised clustering.

Provides lightweight, dependency-free schema enforcement, data quality validation,
and clustering for ML training/inference datasets.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


@dataclass(frozen=True)
class FieldSpec:
    """Specification for an expected schema field."""

    name: str
    dtype: type
    nullable: bool = False
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    allowed_values: Optional[Set[Any]] = None


@dataclass
class ValidationReport:
    """Validation outcome report with data quality metrics."""

    valid: bool
    total_records: int
    errors: List[str] = field(default_factory=list)
    null_counts: Dict[str, int] = field(default_factory=dict)
    summary_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)


class DataSchemaContract:
    """Enforces tabular and record-level schema constraints with quality metrics."""

    def __init__(self, specs: List[FieldSpec]) -> None:
        self.specs = {s.name: s for s in specs}

    def validate(self, records: List[Dict[str, Any]]) -> ValidationReport:
        errors: List[str] = []
        null_counts: Dict[str, int] = {k: 0 for k in self.specs}
        num_vals: Dict[str, List[float]] = {
            k: [] for k, s in self.specs.items() if s.dtype in (int, float)
        }

        for idx, row in enumerate(records):
            for name, spec in self.specs.items():
                val = row.get(name)
                if val is None:
                    null_counts[name] += 1
                    if not spec.nullable:
                        errors.append(f"Row {idx}: missing required field '{name}'")
                    continue

                if not isinstance(val, spec.dtype):
                    errors.append(
                        f"Row {idx}: field '{name}' expected {spec.dtype.__name__}, got {type(val).__name__}"
                    )
                    continue

                if spec.dtype in (int, float):
                    num_vals[name].append(float(val))
                    if spec.min_val is not None and val < spec.min_val:
                        errors.append(
                            f"Row {idx}: field '{name}' value {val} below min {spec.min_val}"
                        )
                    if spec.max_val is not None and val > spec.max_val:
                        errors.append(
                            f"Row {idx}: field '{name}' value {val} exceeds max {spec.max_val}"
                        )

                if spec.allowed_values is not None and val not in spec.allowed_values:
                    errors.append(
                        f"Row {idx}: field '{name}' value '{val}' not in allowed set"
                    )

        summary_stats: Dict[str, Dict[str, float]] = {}
        for col, vals in num_vals.items():
            if vals:
                mean = sum(vals) / len(vals)
                variance = sum((x - mean) ** 2 for x in vals) / len(vals)
                summary_stats[col] = {
                    "count": float(len(vals)),
                    "mean": round(mean, 4),
                    "std": round(math.sqrt(variance), 4),
                    "min": min(vals),
                    "max": max(vals),
                }

        return ValidationReport(
            valid=len(errors) == 0,
            total_records=len(records),
            errors=errors,
            null_counts=null_counts,
            summary_stats=summary_stats,
        )


class PipelineSanitizer:
    """Cleans, normalizes, and transforms raw records before model ingestion."""

    def __init__(self) -> None:
        self.lineage: List[str] = []

    def clean_strings(
        self, records: List[Dict[str, Any]], columns: List[str]
    ) -> List[Dict[str, Any]]:
        self.lineage.append(f"clean_strings: columns={columns}")
        out = []
        for r in records:
            copied = dict(r)
            for c in columns:
                if isinstance(copied.get(c), str):
                    copied[c] = copied[c].strip()
            out.append(copied)
        return out

    def impute_numeric(
        self,
        records: List[Dict[str, Any]],
        column: str,
        strategy: str = "mean",
    ) -> List[Dict[str, Any]]:
        vals = [r[column] for r in records if r.get(column) is not None]
        if not vals:
            fill_val = 0.0
        elif strategy == "median":
            s = sorted(vals)
            mid = len(s) // 2
            fill_val = s[mid] if len(s) % 2 != 0 else (s[mid - 1] + s[mid]) / 2.0
        else:
            fill_val = sum(vals) / len(vals)

        self.lineage.append(f"impute_numeric: column={column}, strategy={strategy}, fill={fill_val}")
        out = []
        for r in records:
            copied = dict(r)
            if copied.get(column) is None:
                copied[column] = fill_val
            out.append(copied)
        return out


class EmbeddingClusterAnalyzer:
    """Lightweight deterministic k-means clustering for embeddings or numeric features."""

    def __init__(self, k: int, max_iter: int = 50, seed: int = 42) -> None:
        if k <= 0:
            raise ValueError("k must be greater than 0")
        self.k = k
        self.max_iter = max_iter
        self.seed = seed
        self.centroids: List[List[float]] = []

    @staticmethod
    def _euclidean_dist(a: List[float], b: List[float]) -> float:
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    def fit_predict(self, vectors: List[List[float]]) -> Tuple[List[int], float]:
        if not vectors:
            return [], 0.0
        if len(vectors) < self.k:
            raise ValueError(f"Need at least {self.k} samples for {self.k} clusters")

        dim = len(vectors[0])
        # Deterministic stratified initial centroids
        step = len(vectors) // self.k
        self.centroids = [list(vectors[i * step]) for i in range(self.k)]

        assignments = [0] * len(vectors)
        for _ in range(self.max_iter):
            # Assignment phase
            changed = False
            for idx, vec in enumerate(vectors):
                best_dist = float("inf")
                best_cluster = 0
                for c_idx, c in enumerate(self.centroids):
                    d = self._euclidean_dist(vec, c)
                    if d < best_dist:
                        best_dist = d
                        best_cluster = c_idx
                if assignments[idx] != best_cluster:
                    assignments[idx] = best_cluster
                    changed = True

            if not changed:
                break

            # Update centroids
            for c_idx in range(self.k):
                cluster_points = [vectors[i] for i, a in enumerate(assignments) if a == c_idx]
                if cluster_points:
                    self.centroids[c_idx] = [
                        sum(p[d] for p in cluster_points) / len(cluster_points)
                        for d in range(dim)
                    ]

        # Calculate total inertia (sum of squared distances to centroid)
        inertia = sum(
            self._euclidean_dist(vectors[i], self.centroids[assignments[i]]) ** 2
            for i in range(len(vectors))
        )
        return assignments, round(inertia, 4)
