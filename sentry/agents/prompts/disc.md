# Placeholder system prompt

This public snapshot does not include the real discretionary prompt.

The code path is complete: `sentry/agents/context.py` loads this file as the static, cacheable
system block, `sentry/llm/client.py` sends it with the per-day digest and the per-symbol context,
and the response must parse as `AgentView` (`sentry/agents/schema.py`). `sentry/agents/validate.py`
then rejects any view that cites an evidence id that was not in the visible context.

To experiment locally, replace this text with your own instructions. Ask for exactly one
AgentView per call and tell the model to cite only the news ids shown in the context.
