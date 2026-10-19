"""strategy_version / strategy_hash.

strategy_hash = sha256(prompt_hash_disc | params_hash | sha256 of each file in HASHED_CONFIGS | ablation_key)

params_hash is supplied by the decision layer (not part of this public snapshot).

STRATEGY_VERSIONS.md carries one machine-readable line per tag:
    <!-- tag: v1.0.0 hash: <64 lowercase hex> -->
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Mapping, Optional

from sentry.config import config_path, file_sha256

TAG_RE = re.compile(r"<!--\s*tag:\s*(v\d+\.\d+\.\d+)\s+hash:\s*([0-9a-f]{64})\s*-->")
HASHED_CONFIGS = ("risk.yaml", "gate.yaml", "universe.yaml", "models.yaml")


class UntaggedStrategyError(RuntimeError):
    pass


def ablation_key(ablation: Mapping[str, str]) -> str:
    return ",".join(f"{k}={ablation[k]}" for k in sorted(ablation))


def strategy_hash(
    prompt_hash_disc: Optional[str], params_hash: str, config_dir: Path | None, ablation: Mapping[str, str]
) -> str:
    parts = [prompt_hash_disc or "", params_hash]
    parts += [file_sha256(config_path(name, config_dir)) for name in HASHED_CONFIGS]
    parts.append(ablation_key(ablation))
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def tags(path: Path) -> dict[str, str]:
    """version -> hash. A tag listed twice with different hashes is an error (never re-point a tag)."""
    out: dict[str, str] = {}
    p = Path(path)
    text = p.read_text() if p.exists() else ""
    for version, h in TAG_RE.findall(text):
        if out.get(version, h) != h:
            raise UntaggedStrategyError(f"{version} is tagged with two different hashes in {path}")
        out[version] = h
    return out


def require_tag(version: str, computed_hash: str, path: Path) -> None:
    """The live loop's startup check: refuse an untagged version or a hash that drifted."""
    known = tags(path)
    if version not in known:
        raise UntaggedStrategyError(f"{version} has no entry in {path}; cut a tag before running")
    if known[version] != computed_hash:
        raise UntaggedStrategyError(
            f"{version} is tagged {known[version][:12]} but the computed strategy_hash is {computed_hash[:12]}: "
            "a live-affecting file changed without a new tag"
        )


def append_tag(path: Path, version: str, hash_: str, note: str) -> None:
    if not re.fullmatch(r"v\d+\.\d+\.\d+", version):
        raise ValueError(f"not a semver tag: {version}")
    if not re.fullmatch(r"[0-9a-f]{64}", hash_):
        raise ValueError("hash must be 64 lowercase hex characters")
    p = Path(path)
    if version in tags(p):
        raise ValueError(f"{version} already tagged in {path}")
    entry = f"\n## {version}\n\n{note.strip()}\n\n<!-- tag: {version} hash: {hash_} -->\n"
    with p.open("a") as f:
        f.write(entry)
