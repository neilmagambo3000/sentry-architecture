"""Layer A / risk may not import Layer B or the LLM SDK, and may not read the clock."""
import ast

from sentry.config import ROOT

BANNED_MODULES = ("sentry.llm", "sentry.agents", "anthropic")
BANNED_CALLS = {("datetime", "now"), ("datetime", "utcnow"), ("datetime", "today"), ("date", "today"), ("time", "time")}


def _banned(name: str) -> bool:
    return any(name == m or name.startswith(m + ".") for m in BANNED_MODULES)


def violations(src: str, label: str = "<src>") -> list[str]:
    out = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            out += [f"{label}: import {a.name}" for a in node.names if _banned(a.name)]
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            mod = node.module or ""
            if _banned(mod) or any(_banned(f"{mod}.{a.name}") for a in node.names):
                out.append(f"{label}: from {mod} import ...")
            if mod in ("time", "datetime") and any((mod, a.name) in BANNED_CALLS for a in node.names):
                out.append(f"{label}: from {mod} import clock")
        elif isinstance(node, ast.Attribute) and isinstance(node.value, (ast.Name, ast.Attribute)):
            base = node.value.id if isinstance(node.value, ast.Name) else node.value.attr
            if (base, node.attr) in BANNED_CALLS:
                out.append(f"{label}:{node.lineno}: {base}.{node.attr}")
    return out


def _files():
    for pkg in ("risk",):
        yield from sorted((ROOT / "sentry" / pkg).rglob("*.py"))


def test_risk_exists():
    assert {"gate.py", "limits.py"} <= {p.name for p in _files()}


def test_no_layer_b_imports_or_clock():
    problems = [v for p in _files() for v in violations(p.read_text(), str(p))]
    assert problems == []


def test_checker_catches_planted_violations():
    src = (
        "import anthropic\nfrom sentry.llm import client\nfrom sentry import agents\nimport datetime\n"
        "x = datetime.datetime.now()\nimport time\ny = time.time()\nfrom time import time\n"
    )
    assert len(violations(src)) == 6
    assert violations("from sentry.features.row import disagreement\nimport math\n") == []
