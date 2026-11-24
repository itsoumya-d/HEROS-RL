"""Static manifest analyzer and PostgreSQL ingestion pipeline for app package telemetry."""
from dataclasses import dataclass, field
import json
import re
from typing import Any, Dict, List, Optional, Set


DANGEROUS_PERMISSIONS = {
    "android.permission.READ_PHONE_STATE",
    "android.permission.ACCESS_FINE_LOCATION",
    "android.permission.ACCESS_COARSE_LOCATION",
    "android.permission.CAMERA",
    "android.permission.RECORD_AUDIO",
    "android.permission.READ_CONTACTS",
    "android.permission.WRITE_EXTERNAL_STORAGE",
    "android.permission.READ_EXTERNAL_STORAGE",
}

TRACKING_SIGNATURES = {
    "appsflyer": r"com\.appsflyer\.",
    "adjust": r"com\.adjust\.sdk\.",
    "branch": r"io\.branch\.",
    "facebook_analytics": r"com\.facebook\.appevents\.",
    "firebase_analytics": r"com\.google\.firebase\.analytics\.",
    "kochava": r"com\.kochava\.",
}


@dataclass
class AppManifestRecord:
    package_name: str
    version_code: int
    version_name: str
    min_sdk: int
    target_sdk: int
    permissions: List[str] = field(default_factory=list)
    exported_components: List[str] = field(default_factory=list)
    detected_sdks: List[str] = field(default_factory=list)
    dangerous_permissions_count: int = 0
    attack_surface_score: float = 0.0


class ManifestStaticAnalyzer:
    """Analyzes APK/IPA manifest declarations and prepares high-throughput PostgreSQL ingestion records."""

    def analyze(self, raw_manifest: Dict[str, Any], class_dump: Optional[List[str]] = None) -> AppManifestRecord:
        package_name = raw_manifest.get("package", "unknown")
        version_code = int(raw_manifest.get("version_code", 0))
        version_name = str(raw_manifest.get("version_name", "0.0.1"))
        min_sdk = int(raw_manifest.get("min_sdk", 21))
        target_sdk = int(raw_manifest.get("target_sdk", 34))

        raw_perms = raw_manifest.get("permissions", [])
        permissions = sorted(list(set(raw_perms)))

        # Dangerous permission extraction
        dangerous = [p for p in permissions if p in DANGEROUS_PERMISSIONS]

        # Exported activities/services without permissions
        exported = []
        for comp in raw_manifest.get("components", []):
            if comp.get("exported", False) and not comp.get("permission"):
                exported.append(comp.get("name", "unnamed"))

        # SDK tracking signature detection
        detected_sdks = set()
        if class_dump:
            for cls in class_dump:
                for sdk_name, pattern in TRACKING_SIGNATURES.items():
                    if re.search(pattern, cls):
                        detected_sdks.add(sdk_name)

        # Attack surface scoring (0.0 to 10.0 scale)
        score = 0.0
        score += min(5.0, len(dangerous) * 0.75)
        score += min(3.0, len(exported) * 0.5)
        if target_sdk < 31:
            score += 2.0  # Outdated target SDK penalty
        attack_surface_score = round(min(10.0, score), 2)

        return AppManifestRecord(
            package_name=package_name,
            version_code=version_code,
            version_name=version_name,
            min_sdk=min_sdk,
            target_sdk=target_sdk,
            permissions=permissions,
            exported_components=exported,
            detected_sdks=sorted(list(detected_sdks)),
            dangerous_permissions_count=len(dangerous),
            attack_surface_score=attack_surface_score,
        )

    def generate_postgres_upsert_sql(self, record: AppManifestRecord) -> Dict[str, Any]:
        """Generates idempotent PostgreSQL upsert statement and parameterized query values."""
        query = (
            "INSERT INTO app_manifest_telemetry ("
            "package_name, version_code, version_name, min_sdk, target_sdk, "
            "permissions, exported_components, detected_sdks, dangerous_count, attack_surface_score, ingested_at"
            ") VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, NOW()) "
            "ON CONFLICT (package_name, version_code) DO UPDATE SET "
            "version_name = EXCLUDED.version_name, "
            "target_sdk = EXCLUDED.target_sdk, "
            "permissions = EXCLUDED.permissions, "
            "exported_components = EXCLUDED.exported_components, "
            "detected_sdks = EXCLUDED.detected_sdks, "
            "dangerous_count = EXCLUDED.dangerous_count, "
            "attack_surface_score = EXCLUDED.attack_surface_score, "
            "ingested_at = NOW();"
        )
        params = [
            record.package_name,
            record.version_code,
            record.version_name,
            record.min_sdk,
            record.target_sdk,
            json.dumps(record.permissions),
            json.dumps(record.exported_components),
            json.dumps(record.detected_sdks),
            record.dangerous_permissions_count,
            record.attack_surface_score,
        ]
        return {"query": query, "params": params}
