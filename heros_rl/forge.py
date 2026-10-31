# SPDX-License-Identifier: MIT
"""Deterministic forge-binary bridge used by the accuracy reward."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass

RISK_SCORE = {
    "SAFE": 1.0,
    "NOTABLE": 0.8,
    "MEDIUM": 0.5,
    "HIGH": 0.25,
    "CRITICAL": 0.0,
}


class ForgeError(RuntimeError):
    """The forge binary ran but returned an unusable response."""


class ForgeUnavailable(RuntimeError):
    """No forge binary is configured or found."""


def to_inline(schema: str) -> str:
    if "|" in schema:
        raise ValueError("Forge schemas must not contain pipe characters")
    return schema.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "|")


@dataclass(frozen=True)
class ForgeAnalysis:
    risk_tier: str
    risk_score: float
    has_data_loss: bool
    decision_required: bool
    raw: dict

    @classmethod
    def from_payload(cls, payload: dict) -> "ForgeAnalysis":
        tier = str(payload.get("risk_tier", "CRITICAL")).upper()
        return cls(
            risk_tier=tier,
            risk_score=float(payload.get("risk_score", RISK_SCORE.get(tier, 0.0))),
            has_data_loss=bool(payload.get("has_data_loss", False)),
            decision_required=bool(payload.get("decision_required", False)),
            raw=payload,
        )


class ForgeClient:
    def __init__(self, binary: str | None = None, timeout: float = 10.0):
        self.binary = (
            binary
            or os.environ.get("HEROS_FORGE_BIN")
            or os.environ.get("FORGE_BIN")
            or shutil.which("forge")
        )
        self.timeout = timeout

    @property
    def available(self) -> bool:
        return bool(self.binary)

    def analyze(self, old_schema: str, target_schema: str) -> ForgeAnalysis:
        if not self.binary:
            raise ForgeUnavailable(
                "forge binary not found; set HEROS_FORGE_BIN to use the accuracy reward"
            )

        command = [
            self.binary,
            "analyze",
            "--from",
            to_inline(old_schema),
            "--to",
            to_inline(target_schema),
        ]
        try:
            process = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise ForgeError(f"forge could not be executed: {error}") from error

        output = (process.stdout or "").strip()
        if not output:
            raise ForgeError(
                f"forge produced no output (exit {process.returncode}): {process.stderr.strip()}"
            )
        try:
            payload = json.loads(output)
        except json.JSONDecodeError as error:
            raise ForgeError("forge produced invalid JSON") from error
        if not isinstance(payload, dict):
            raise ForgeError("forge produced a non-object response")
        if "error_code" in payload:
            raise ForgeError(
                f"forge error {payload.get('error_code')}: {payload.get('error')}"
            )
        return ForgeAnalysis.from_payload(payload)
