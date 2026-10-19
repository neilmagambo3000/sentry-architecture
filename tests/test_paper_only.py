"""Paper-only guard. The real TradingClient is never constructed here."""
import inspect
import re
import subprocess
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from alpaca.common.enums import BaseURL

from sentry.config import ROOT
from sentry.execution import alpaca_paper
from sentry.execution.alpaca_paper import PAPER_URL, AlpacaPaperBroker, DryRunError, LiveEndpointError
from sentry.execution.orders import submit_orders
from sentry.types import Order

T0 = datetime(2026, 10, 1, 13, 35, tzinfo=timezone.utc)
ORDER = Order("AAA", "buy", 7.9, None, "sentry-abababab-AAA-20261001-buy", "v1.0.0", "ab" * 32)


class FakeClient:
    def __init__(self, base_url=BaseURL.TRADING_PAPER, sandbox=True, account_number="PA123"):
        self._base_url = base_url
        self._sandbox = sandbox
        self.submitted = []
        self.account_number = account_number

    def submit_order(self, req):
        self.submitted.append(req)
        return SimpleNamespace(id="broker-1")

    def get_orders(self, req=None):
        return []

    def get_account(self):
        return SimpleNamespace(equity="100.0", cash="92.1", last_equity="100.0", account_number=self.account_number)

    def get_portfolio_history(self, req=None):
        return SimpleNamespace(timestamp=[1790000000, 1790086400], equity=[0.0, 100.0])

    def get_all_positions(self):
        return [SimpleNamespace(symbol="AAA", qty="0.5", side=SimpleNamespace(value="short"), avg_entry_price="50",
                                market_value="-25")]


def test_rejects_live_key_prefix():
    with pytest.raises(LiveEndpointError):
        AlpacaPaperBroker(FakeClient(), "AKXXXXXXXX")
    with pytest.raises(LiveEndpointError):
        AlpacaPaperBroker(FakeClient(), "")


def test_rejects_non_paper_client():
    with pytest.raises(LiveEndpointError):
        AlpacaPaperBroker(FakeClient(base_url="https://example.invalid"), "PKXXXXXXXX")
    with pytest.raises(LiveEndpointError):
        AlpacaPaperBroker(FakeClient(sandbox=False), "PKXXXXXXXX")
    assert str(BaseURL.TRADING_PAPER.value) == PAPER_URL
    AlpacaPaperBroker(FakeClient(), "PKXXXXXXXX")


def test_no_way_to_select_live():
    for fn in (AlpacaPaperBroker.__init__, AlpacaPaperBroker.from_env):
        params = set(inspect.signature(fn).parameters)
        assert not params & {"paper", "base_url", "url_override", "url", "live", "sandbox"}
    src = inspect.getsource(alpaca_paper)
    assert "paper=True" in src
    assert re.search(r"paper\s*=\s*(False|0|not\b)", src) is None
    assert "url_override" not in src


def test_repo_has_no_live_endpoint_string():
    live = re.compile(r"(?<!paper-)api\.alpaca\.markets")
    files = subprocess.run(["git", "ls-files", "-co", "--exclude-standard"], cwd=ROOT, capture_output=True, text=True)
    paths = [
        p for p in files.stdout.splitlines()
        if p and not p.startswith((".venv/", "data/")) and not p.rsplit("/", 1)[-1].startswith(".env")  # never read secrets
    ]
    assert paths, "git ls-files returned nothing"
    hits = []
    for p in paths:
        f = ROOT / p
        if f.is_file():
            try:
                text = f.read_text()
            except UnicodeDecodeError:
                continue
            if live.search(text):
                hits.append(p)
    assert hits == []


def test_dry_run_default_on_and_never_calls_client(db, monkeypatch):
    monkeypatch.delenv("SENTRY_DRY_RUN", raising=False)
    client = FakeClient()
    b = AlpacaPaperBroker(client, "PKXXXXXXXX")
    assert b.dry_run is True
    with pytest.raises(DryRunError):
        b.submit(ORDER)
    assert submit_orders(b, [ORDER], db, dry_run=False) == [(ORDER.client_order_id, "dry_run")]
    assert client.submitted == []
    assert db.execute("select dry_run from orders").fetchone()[0] == 1


def test_dry_run_only_off_for_an_explicit_false_value(monkeypatch):
    """A blank or misspelled SENTRY_DRY_RUN (an empty compose interpolation, a typo) must keep dry run ON."""
    for v in ("", " ", "ture", "flase", "none", "null", "2"):
        monkeypatch.setenv("SENTRY_DRY_RUN", v)
        assert AlpacaPaperBroker(FakeClient(), "PKXXXXXXXX").dry_run is True, repr(v)
    for v in ("0", "false", "False", "no", "off", " OFF "):
        monkeypatch.setenv("SENTRY_DRY_RUN", v)
        assert AlpacaPaperBroker(FakeClient(), "PKXXXXXXXX").dry_run is False, repr(v)


def test_dry_run_never_calls_cancel():
    class Client(FakeClient):
        def __init__(self):
            super().__init__()
            self.cancelled = []

        def cancel_order_by_id(self, order_id):
            self.cancelled.append(order_id)

    client = Client()
    with pytest.raises(DryRunError):
        AlpacaPaperBroker(client, "PKXXXXXXXX", dry_run=True).cancel("broker-1")
    assert client.cancelled == []
    AlpacaPaperBroker(client, "PKXXXXXXXX", dry_run=False).cancel("broker-1")
    assert client.cancelled == ["broker-1"]


def test_dry_run_off_submits_notional_day_market(db, monkeypatch):
    monkeypatch.setenv("SENTRY_DRY_RUN", "0")
    client = FakeClient()
    b = AlpacaPaperBroker(client, "PKXXXXXXXX")
    assert b.dry_run is False
    assert submit_orders(b, [ORDER], db, dry_run=False) == [(ORDER.client_order_id, "submitted")]
    req = client.submitted[0]
    assert req.notional == 7.9 and req.qty is None and req.client_order_id == ORDER.client_order_id
    assert req.time_in_force.value == "day" and req.side.value == "buy" and req.type.value == "market"


def test_account_and_positions_mapping():
    b = AlpacaPaperBroker(FakeClient(), "PKXXXXXXXX", dry_run=True)
    a = b.account()
    assert a.equity == 100.0 and a.last_equity == 100.0 and a.is_paper and len(a.equity_history) == 1
    assert b.positions()["AAA"].qty == -0.5
    assert AlpacaPaperBroker(FakeClient(account_number="X1"), "PKXXXXXXXX").account().is_paper is False
