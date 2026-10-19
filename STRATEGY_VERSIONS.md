# Strategy versions

One entry per live-affecting change. The live loop computes `strategy_hash` at startup and refuses
to run unless the version it was started with is tagged below with exactly that hash
(`sentry/features/version.py: require_tag`).

## What the hash covers

`strategy_hash = sha256(prompt_hash_disc | params_hash | sha256 of each config file in HASHED_CONFIGS | ablation_key)`

- `prompt_hash_disc`: the frozen discretionary prompt (model id, system text, schema, gate.yaml bytes).
- `params_hash`: the decision layer's parameters and fixed constants (decision layer not published).
- `risk.yaml`, `gate.yaml`, `universe.yaml`, `models.yaml`.
- `ablation_key`: the ablation arms, for example `debate_rounds=0,escalate=off,trader=off`.

Changing any of these without a new tag is a startup error, not a silent blend.

## Rules

- Each tag starts its own clock. Reporting is per tag, never blended.
- A minimum number of closed trades and trading days under the current tag before any
  live-affecting change, and only when a pre-registered trigger has fired.
- A larger minimum sample under one tag before any statement about edge.
- Before a new tag is deployed, its entry records the trigger, the evidence, the diff and the expected effect.

## Entry format

Each tag is a `## vX.Y.Z` heading, the human-readable note, then exactly one machine-readable line:

    <!-- tag: vX.Y.Z hash: <64 lowercase hex characters> -->

Tags are appended with `sentry.features.version.append_tag` and are never edited or re-pointed.

## Status

No tags in the public snapshot.
