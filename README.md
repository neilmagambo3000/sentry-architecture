# SENTRY (public architecture snapshot)

Personal research project. This is not a product, a service, a trading signal or financial advice.

SENTRY is the plumbing for a research system that combines deterministic market-data features and
one LLM news agent into a daily decision for a **paper** brokerage account (simulated money only).
This repository is a public snapshot of the infrastructure: data handling, the LLM call path, the
risk gate, the paper broker adapter, validation tooling and a read-only dashboard. The decision
logic itself (feature definitions, scoring, parameters, symbol universe and the real prompt) is
private and is not included here.

## What is in this snapshot

| Area | What it shows | Where |
|---|---|---|
| Point-in-time data | One read path for bars, news, filings and macro vintages; nothing first seen after `as_of` is visible, enforced with an explicit raise | `sentry/data/snapshot.py`, `sentry/data/store.py` |
| Data loaders | Alpaca bars and news, SEC EDGAR first-reported XBRL facts, ALFRED macro vintages | `sentry/data/loaders/` |
| LLM call path | One synchronous call path with a prompt registry, a daily cost ceiling checked before any network call, a response cache, a usage log and structured-output validation | `sentry/llm/` |
| Agent context | Cache-friendly prompt layout, ticker and date masking, evidence-id checks that reject a view citing news it was not shown | `sentry/agents/` |
| Risk gate | Hard limits that run after the decision layer and can only reduce exposure (kill switch, stale data, cost ceiling, drawdown halts, max weight, max gross, long only) | `sentry/risk/` |
| Execution | Paper-only broker adapter (no live endpoint, paper key prefix required), fake broker, idempotent orders, reconcile, trade log | `sentry/execution/` |
| Versioning | `strategy_hash` over prompt, parameters and config; the loop refuses an untagged hash | `sentry/features/version.py`, `STRATEGY_VERSIONS.md` |
| Validation | Forward information coefficient with seeded random and 12-1 momentum baselines, prompt stability check, version-scoped paper report with pre-registered criteria | `sentry/layer_a/validation/` |
| Dashboard | Read-only FastAPI + Jinja2 dashboard with a signed-session login and webhook alerts | `sentry/web/` |

Stack: Python 3.11, pandas, NumPy, SciPy, Pydantic, PyYAML, Anthropic SDK, alpaca-py, edgartools,
fredapi, FastAPI, Jinja2, itsdangerous, SQLite, pytest, ruff.

## What is not in this snapshot

- The decision layer, the technical and fundamental feature blocks, the news pre-filter and their tests.
- The live daily loop, scheduler and shadow baselines, which depend on the decision layer.
- The real system prompt. `sentry/agents/prompts/disc.md` is a placeholder.
- The symbol universe, the selection rule and strategy parameters. Config values in `config/` are
  illustrative.
- Deployment files, planning documents and research notes.

## Status

- Built and tested only against mocks, synthetic data and a local fake broker.
- No live trading. No real money. The broker adapter is paper-only in code.
- No performance claims. Nothing in this repository is evidence that any approach makes money.
- Tests: 130 tests in 18 test files, all passing locally on Python 3.11 with the pinned
  requirements, and `ruff check .` clean (2026-10-18). Only tests for the included modules are here.
- No CI is configured.

## Run locally

```
uv venv -p 3.11 .venv
uv pip install -p .venv/bin/python -r requirements-dev.txt
.venv/bin/python -m pytest -q     # no network, no keys
.venv/bin/ruff check .
```

Keys are read from the environment only; names are listed in `.env.example`. Never commit values.

## Disclaimer

Research code shared to show software design. It is not financial, investment, legal or tax advice,
not an offer or solicitation, and makes no claim about past or future performance. Do not use it to
trade real money. Use of any broker or data API is subject to that provider's terms.
