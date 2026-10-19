"""SQLite schema. One file, one writer process (the engine).

Timestamps are stored as ISO-8601 UTC strings ("2026-09-30T12:00:00+00:00"),
dates as "YYYY-MM-DD". JSON columns hold dicts/lists.
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS bars (
    symbol TEXT NOT NULL, date TEXT NOT NULL,
    o REAL, h REAL, l REAL, c REAL, v REAL, adj_c REAL,
    PRIMARY KEY (symbol, date));

CREATE TABLE IF NOT EXISTS articles (
    id TEXT PRIMARY KEY, first_seen TEXT NOT NULL, published TEXT,
    source TEXT, headline TEXT, body TEXT, url TEXT);
CREATE TABLE IF NOT EXISTS article_symbols (
    article_id TEXT NOT NULL, symbol TEXT NOT NULL,
    PRIMARY KEY (article_id, symbol));
CREATE INDEX IF NOT EXISTS ix_article_symbols_symbol ON article_symbols(symbol);

CREATE TABLE IF NOT EXISTS filings (
    id TEXT PRIMARY KEY, symbol TEXT NOT NULL, form TEXT, filed_at TEXT NOT NULL,
    period_end TEXT, facts_json TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS ix_filings_symbol ON filings(symbol, filed_at);

CREATE TABLE IF NOT EXISTS macro_vintages (
    series TEXT NOT NULL, vintage TEXT NOT NULL, obs_json TEXT NOT NULL,
    PRIMARY KEY (series, vintage));

CREATE TABLE IF NOT EXISTS agent_calls (
    agent TEXT NOT NULL, symbol TEXT NOT NULL, as_of TEXT NOT NULL,
    prompt_hash TEXT NOT NULL, ablation_key TEXT NOT NULL,
    model_id TEXT, schema_version TEXT, temperature REAL,
    input_tokens INTEGER, cache_creation_input_tokens INTEGER,
    cache_read_input_tokens INTEGER, output_tokens INTEGER,
    cost_usd REAL, cache_hit INTEGER, batch_id TEXT, escalated INTEGER,
    view_json TEXT, error TEXT, latency_ms INTEGER, frozen INTEGER,
    PRIMARY KEY (agent, symbol, as_of, prompt_hash, ablation_key));

CREATE TABLE IF NOT EXISTS llm_cache (
    key TEXT PRIMARY KEY, response_json TEXT NOT NULL, created_at TEXT NOT NULL);

CREATE TABLE IF NOT EXISTS features (
    symbol TEXT NOT NULL, as_of TEXT NOT NULL, strategy_hash TEXT NOT NULL,
    strategy_version TEXT NOT NULL, row_json TEXT NOT NULL,
    PRIMARY KEY (symbol, as_of, strategy_hash));

CREATE TABLE IF NOT EXISTS decisions (
    symbol TEXT NOT NULL, as_of TEXT NOT NULL, direction INTEGER, target_weight REAL,
    stop REAL, score REAL, reason_json TEXT, params_hash TEXT,
    strategy_version TEXT NOT NULL, strategy_hash TEXT NOT NULL,
    PRIMARY KEY (symbol, as_of, strategy_hash));

CREATE TABLE IF NOT EXISTS shadow_decisions (
    symbol TEXT NOT NULL, as_of TEXT NOT NULL, baseline TEXT NOT NULL,
    direction INTEGER, target_weight REAL, strategy_version TEXT NOT NULL,
    PRIMARY KEY (symbol, as_of, baseline, strategy_version));
CREATE TABLE IF NOT EXISTS shadow_equity (
    date TEXT NOT NULL, baseline TEXT NOT NULL, strategy_version TEXT NOT NULL,
    equity REAL NOT NULL, turnover REAL, cost REAL,
    PRIMARY KEY (date, baseline, strategy_version));

CREATE TABLE IF NOT EXISTS orders (
    client_order_id TEXT PRIMARY KEY, broker_order_id TEXT, symbol TEXT, side TEXT,
    notional REAL, qty REAL, type TEXT, time_in_force TEXT, status TEXT,
    submitted_at TEXT, dry_run INTEGER NOT NULL, reason TEXT,
    strategy_version TEXT NOT NULL, strategy_hash TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS fills (
    broker_order_id TEXT NOT NULL, client_order_id TEXT, symbol TEXT, side TEXT,
    qty REAL, price REAL, filled_at TEXT, ref_price REAL,
    strategy_version TEXT NOT NULL, strategy_hash TEXT NOT NULL,
    PRIMARY KEY (broker_order_id, filled_at));
CREATE TABLE IF NOT EXISTS closed_trades (
    symbol TEXT NOT NULL, opened_at TEXT NOT NULL, closed_at TEXT NOT NULL,
    side INTEGER, qty REAL, entry_price REAL, exit_price REAL, pnl REAL, exit_reason TEXT,
    strategy_version TEXT NOT NULL, strategy_hash TEXT NOT NULL,
    PRIMARY KEY (symbol, opened_at, strategy_hash));
CREATE TABLE IF NOT EXISTS positions_snapshot (
    ts TEXT NOT NULL, symbol TEXT NOT NULL, qty REAL, avg_entry_price REAL,
    market_value REAL, equity REAL, strategy_version TEXT NOT NULL,
    PRIMARY KEY (ts, symbol));
CREATE TABLE IF NOT EXISTS equity_history (
    ts TEXT PRIMARY KEY, equity REAL NOT NULL, strategy_version TEXT NOT NULL);

CREATE TABLE IF NOT EXISTS cost_daily (date TEXT PRIMARY KEY, usd REAL NOT NULL);
CREATE TABLE IF NOT EXISTS health (
    cycle_ts TEXT NOT NULL, step TEXT NOT NULL, status TEXT NOT NULL, detail TEXT,
    PRIMARY KEY (cycle_ts, step));
CREATE TABLE IF NOT EXISTS ic_daily (
    date TEXT NOT NULL, strategy_version TEXT NOT NULL, horizon INTEGER NOT NULL,
    signal TEXT NOT NULL, ic REAL, n INTEGER,
    PRIMARY KEY (date, strategy_version, horizon, signal));
"""


def default_db_path() -> Path:
    return Path(os.environ.get("SENTRY_DB", Path(os.environ.get("SENTRY_DATA_DIR", "data")) / "sentry.db"))


def connect(path: str | os.PathLike | None = None) -> sqlite3.Connection:
    """Open (and create if needed) the store. Use ":memory:" in tests."""
    p = str(path if path is not None else default_db_path())
    if p != ":memory:":
        Path(p).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(p, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL" if p != ":memory:" else "PRAGMA journal_mode=MEMORY")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(SCHEMA)
    return conn
