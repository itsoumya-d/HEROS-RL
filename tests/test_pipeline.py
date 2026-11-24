"""Unit tests for heros_rl.pipeline module."""

import unittest
from heros_rl.pipeline import (
    DataSchemaContract,
    FieldSpec,
    PipelineSanitizer,
    EmbeddingClusterAnalyzer,
)


class TestDataSchemaContract(unittest.TestCase):
    def setUp(self):
        self.specs = [
            FieldSpec(name="id", dtype=int),
            FieldSpec(name="score", dtype=float, min_val=0.0, max_val=1.0),
            FieldSpec(name="status", dtype=str, allowed_values={"active", "pending"}),
            FieldSpec(name="notes", dtype=str, nullable=True),
        ]
        self.contract = DataSchemaContract(self.specs)

    def test_valid_records(self):
        records = [
            {"id": 1, "score": 0.85, "status": "active", "notes": "ok"},
            {"id": 2, "score": 0.40, "status": "pending", "notes": None},
        ]
        report = self.contract.validate(records)
        self.assertTrue(report.valid)
        self.assertEqual(len(report.errors), 0)
        self.assertEqual(report.null_counts["notes"], 1)
        self.assertIn("score", report.summary_stats)
        self.assertAlmostEqual(report.summary_stats["score"]["mean"], 0.625, places=3)

    def test_invalid_types_and_bounds(self):
        records = [
            {"id": "not_an_int", "score": 1.5, "status": "unknown"},
        ]
        report = self.contract.validate(records)
        self.assertFalse(report.valid)
        self.assertTrue(any("expected int" in e for e in report.errors))
        self.assertTrue(any("exceeds max" in e for e in report.errors))
        self.assertTrue(any("not in allowed set" in e for e in report.errors))


class TestPipelineSanitizer(unittest.TestCase):
    def test_clean_strings(self):
        sanitizer = PipelineSanitizer()
        records = [{"name": "  test  ", "val": 10}]
        cleaned = sanitizer.clean_strings(records, ["name"])
        self.assertEqual(cleaned[0]["name"], "test")
        self.assertEqual(len(sanitizer.lineage), 1)

    def test_impute_numeric(self):
        sanitizer = PipelineSanitizer()
        records = [{"val": 10.0}, {"val": 20.0}, {"val": None}]
        imputed = sanitizer.impute_numeric(records, "val", strategy="mean")
        self.assertEqual(imputed[2]["val"], 15.0)

        imputed_med = sanitizer.impute_numeric(records, "val", strategy="median")
        self.assertEqual(imputed_med[2]["val"], 15.0)


class TestEmbeddingClusterAnalyzer(unittest.TestCase):
    def test_clustering_partition(self):
        analyzer = EmbeddingClusterAnalyzer(k=2, max_iter=20)
        vectors = [
            [1.0, 1.0],
            [1.1, 0.9],
            [10.0, 10.0],
            [10.2, 9.8],
        ]
        labels, inertia = analyzer.fit_predict(vectors)
        self.assertEqual(len(labels), 4)
        self.assertEqual(labels[0], labels[1])
        self.assertEqual(labels[2], labels[3])
        self.assertNotEqual(labels[0], labels[2])
        self.assertGreater(inertia, 0.0)

    def test_k_greater_than_samples(self):
        analyzer = EmbeddingClusterAnalyzer(k=5)
        with self.assertRaises(ValueError):
            analyzer.fit_predict([[1.0], [2.0]])


if __name__ == "__main__":
    unittest.main()
