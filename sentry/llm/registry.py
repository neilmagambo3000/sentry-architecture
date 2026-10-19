"""Prompt registry. Nothing is frozen until the owner approves the stability check."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from sentry.config import config_path


class UnregisteredPromptError(RuntimeError):
    pass


def prompt_hash(agent: str, system_text: str, schema_version: str, temperature: float, model_id: str,
                gate_yaml_bytes: bytes, masked: bool) -> str:
    h = hashlib.sha256()
    for part in (agent, system_text, schema_version, repr(float(temperature)), model_id):
        h.update(part.encode() + b"\x1f")
    h.update(gate_yaml_bytes + b"\x1f")
    h.update(b"masked=1" if masked else b"masked=0")
    return h.hexdigest()


def is_dev() -> bool:
    return os.environ.get("SENTRY_ENV", "").strip().lower() == "dev"


class PromptRegistry:
    def __init__(self, path: Path | None = None):
        self.path = Path(path or config_path("registry.json"))
        data = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.frozen: dict[str, dict] = dict(data.get("frozen", {}))

    def is_frozen(self, h: str) -> bool:
        return h in self.frozen

    def freeze(self, h: str, agent: str, frozen_on: str) -> None:
        self.frozen[h] = {"agent": agent, "frozen_on": frozen_on}
        self.path.write_text(json.dumps({"frozen": self.frozen}, indent=2, sort_keys=True) + "\n")

    def check(self, h: str) -> bool:
        """True if frozen; False if unfrozen but SENTRY_ENV=dev; raises in prod."""
        if self.is_frozen(h):
            return True
        if is_dev():
            return False
        raise UnregisteredPromptError(f"prompt hash {h[:12]} is not frozen in {self.path.name}")
