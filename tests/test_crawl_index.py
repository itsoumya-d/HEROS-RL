import unittest
from heros_rl.crawl_index import ManifestStaticAnalyzer


class TestManifestStaticAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = ManifestStaticAnalyzer()

    def test_analyzes_permissions_and_dangerous_flags(self):
        manifest = {
            "package": "com.sample.analytics",
            "version_code": 140,
            "version_name": "1.4.0",
            "min_sdk": 24,
            "target_sdk": 34,
            "permissions": [
                "android.permission.INTERNET",
                "android.permission.READ_PHONE_STATE",
                "android.permission.ACCESS_FINE_LOCATION",
            ],
            "components": [
                {"name": "com.sample.ExportedReceiver", "exported": True},
                {"name": "com.sample.InternalService", "exported": False},
            ],
        }
        classes = [
            "com.sample.MainActivity",
            "com.appsflyer.AFKeystore",
            "io.branch.referral.Branch",
        ]

        record = self.analyzer.analyze(manifest, class_dump=classes)

        self.assertEqual(record.package_name, "com.sample.analytics")
        self.assertEqual(record.version_code, 140)
        self.assertEqual(record.dangerous_permissions_count, 2)
        self.assertIn("appsflyer", record.detected_sdks)
        self.assertIn("branch", record.detected_sdks)
        self.assertEqual(record.exported_components, ["com.sample.ExportedReceiver"])
        self.assertGreater(record.attack_surface_score, 1.0)

    def test_postgres_upsert_generation(self):
        manifest = {
            "package": "com.core.app",
            "version_code": 1,
            "version_name": "1.0",
            "permissions": ["android.permission.INTERNET"],
        }
        record = self.analyzer.analyze(manifest)
        sql = self.analyzer.generate_postgres_upsert_sql(record)

        self.assertIn("ON CONFLICT (package_name, version_code) DO UPDATE", sql["query"])
        self.assertEqual(sql["params"][0], "com.core.app")
        self.assertEqual(sql["params"][1], 1)


if __name__ == "__main__":
    unittest.main()
