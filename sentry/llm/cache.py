"""Response cache on table llm_cache. Stores the validated AgentView JSON."""
from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timezone
from typing import Optional


def cache_key(agent: str, model_id: str, prompt_hash: str, schema_version: str, symbol: str,
              as_of: datetime, context_hash: str) -> str:
    parts = [agent, model_id, prompt_hash, schema_version, symbol, as_of.isoformat(), context_hash]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def get(conn: sqlite3.Connection, key: str) -> Optional[str]:
    row = conn.execute("SELECT response_json FROM llm_cache WHERE key = ?", (key,)).fetchone()
    return row[0] if row else None


def put(conn: sqlite3.Connection, key: str, response_json: str) -> None:
    # created_at is bookkeeping only; nothing reads it to decide visibility.
    conn.execute("INSERT OR REPLACE INTO llm_cache (key, response_json, created_at) VALUES (?, ?, ?)",
                 (key, response_json, datetime.now(timezone.utc).isoformat()))
    conn.commit()
