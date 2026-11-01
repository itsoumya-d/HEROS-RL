# SPDX-License-Identifier: MIT
import unittest

from heros_rl.mining import (
    CorpusDocument,
    HashedNgramClassifier,
    SeedDocument,
    mine_domains,
    path_pattern,
    tokenize,
)

POSITIVE = "postgres migration schema expand contract rollback sql alter table backfill"
NEGATIVE = "recipe travel sports movie music garden cooking holiday"


class TokenizationTests(unittest.TestCase):
    def test_tokens_and_path_patterns(self):
        self.assertEqual(tokenize("Zero-lang SQL Migration!"), ["zero", "lang", "sql", "migration"])
        self.assertEqual(path_pattern("https://docs.example.com/docs/migrations/12345"), "/docs/migrations/:id")
        self.assertEqual(path_pattern("https://example.com/"), "/")


class ClassifierTests(unittest.TestCase):
    def setUp(self):
        self.examples = [(POSITIVE, 1), (POSITIVE + " zero", 1), (NEGATIVE, 0), (NEGATIVE + " kitchen", 0)]
        self.classifier = HashedNgramClassifier(dimension=256, epochs=16, seed=3).fit(self.examples)

    def test_learns_clear_domains(self):
        self.assertGreater(self.classifier.predict_proba(POSITIVE), 0.5)
        self.assertLess(self.classifier.predict_proba(NEGATIVE), 0.5)

    def test_persistence_round_trip(self):
        restored = HashedNgramClassifier.from_dict(self.classifier.to_dict())
        self.assertAlmostEqual(restored.predict_proba(POSITIVE), self.classifier.predict_proba(POSITIVE))
        self.assertAlmostEqual(restored.predict_proba(NEGATIVE), self.classifier.predict_proba(NEGATIVE))

    def test_rejects_invalid_training_data(self):
        with self.assertRaises(ValueError):
            HashedNgramClassifier().fit([])
        with self.assertRaises(ValueError):
            HashedNgramClassifier().fit([(POSITIVE, 1), (POSITIVE, 1)])


class MiningTests(unittest.TestCase):
    def corpus(self):
        documents = []
        for index in range(6):
            label_text = POSITIVE if index < 3 else NEGATIVE
            documents.append(CorpusDocument(
                "docs.example.com",
                f"https://docs.example.com/docs/migrations/{index}",
                f"{label_text} document {index}",
            ))
        for index in range(12):
            label_text = POSITIVE if index == 0 else NEGATIVE
            documents.append(CorpusDocument(
                "blog.example.com",
                f"https://blog.example.com/posts/{index}",
                f"{label_text} article {index}",
            ))
        for index in range(6):
            documents.append(CorpusDocument(
                "shop.example.com",
                f"https://shop.example.com/products/{index}",
                f"{NEGATIVE} product {index}",
            ))
        for index in range(6):
            label_text = POSITIVE if index < 3 else NEGATIVE
            documents.append(CorpusDocument(
                "proofs.example.org",
                f"https://proofs.example.org/proofs/{index}",
                f"{label_text} theorem proof document {index}",
            ))
        return documents

    def seeds(self):
        return [
            SeedDocument(POSITIVE, 1),
            SeedDocument(POSITIVE + " zero", 1),
            SeedDocument(NEGATIVE, 0),
            SeedDocument(NEGATIVE + " kitchen", 0),
        ]

    def test_four_iteration_discovery(self):
        run = mine_domains(self.seeds(), self.corpus(), iterations=4, threshold=0.1, min_pages=2)
        self.assertEqual(len(run.iterations), 4)
        first = run.iterations[0]
        selected = {result.domain for result in first.selected_domains}
        self.assertIn("docs.example.com", selected)
        self.assertIn("proofs.example.org", selected)
        self.assertNotIn("shop.example.com", selected)
        self.assertNotIn("blog.example.com", selected)
        self.assertGreater(first.added_documents, 0)
        self.assertGreater(run.final_seed_count, len(self.seeds()))
        self.assertTrue(any("/docs/migrations/:id" in result.path_patterns for result in first.selected_domains if result.domain == "docs.example.com"))

    def test_mining_is_deterministic(self):
        first = mine_domains(self.seeds(), self.corpus(), iterations=4).to_dict()
        second = mine_domains(self.seeds(), self.corpus(), iterations=4).to_dict()
        self.assertEqual(first, second)

    def test_rejects_invalid_configuration(self):
        with self.assertRaises(ValueError):
            mine_domains([], self.corpus())
        with self.assertRaises(ValueError):
            mine_domains(self.seeds(), [])
        with self.assertRaises(ValueError):
            mine_domains(self.seeds(), self.corpus(), iterations=0)
        with self.assertRaises(ValueError):
            mine_domains(self.seeds(), self.corpus(), threshold=0.0)


if __name__ == "__main__":
    unittest.main()
