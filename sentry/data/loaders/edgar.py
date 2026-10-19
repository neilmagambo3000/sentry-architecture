"""SEC EDGAR company facts -> filings table (write-only). Network access only through the injected client.

Source: the SEC companyfacts JSON (via edgartools). Each fact entry carries the accession number (`accn`)
of the filing that reported it, so "first reported" = the entry whose accn is the filing that covers that
period (later restatements arrive under later accns and are ignored). Only original 10-Q / 10-K forms
are used; amendments are skipped.

Per filing (accn): period_end = latest `end` among its us-gaap facts. Flow facts are taken at the longest
duration ending at period_end (fiscal YTD: 3/6/9/12 months); that length is stored as facts["months"] and
flows with a different length are dropped. Instants are taken at period_end. shares_outstanding is the
dei cover-page count (latest date in the filing), falling back to diluted weighted-average shares.
ebitda = operating_income + d_and_a when both exist. Missing facts are omitted, never imputed.

companyfacts gives only the filing date, so filed_at is set to 17:30 New York time on that date: EDGAR
dates anything accepted after 17:30 to the next business day, so the real acceptance is never later.
"""
from __future__ import annotations

import os
import sqlite3
from datetime import date, datetime, time
from zoneinfo import ZoneInfo

from sentry.data.writes import insert_filing
from sentry.types import Filing

NY = ZoneInfo("America/New_York")
FORMS = ("10-Q", "10-K")

FLOW = {
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueNet"],
    "net_income": ["NetIncomeLoss"],
    "eps_diluted": ["EarningsPerShareDiluted"],
    "gross_profit": ["GrossProfit"],
    "operating_cash_flow": ["NetCashProvidedByUsedInOperatingActivities"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment"],
    "operating_income": ["OperatingIncomeLoss"],
    "d_and_a": ["DepreciationDepletionAndAmortization", "DepreciationAndAmortization",
                "DepreciationAmortizationAndAccretionNet"],
    "interest_expense": ["InterestExpense", "InterestExpenseNonoperating", "InterestExpenseDebt"],
}
INSTANT = {
    "total_assets": ["Assets"],
    "equity": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    # preference order; LongTermDebtNoncurrent alone understates debt (current portion missing)
    "total_debt": ["DebtLongtermAndShorttermCombinedAmount", "LongTermDebt", "LongTermDebtNoncurrent"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue",
             "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
}


def _entries(facts: dict, taxonomy: str, tag: str) -> list[dict]:
    units = facts.get(taxonomy, {}).get(tag, {}).get("units", {})
    return [e for lst in units.values() for e in lst]


def _mine(facts: dict, accn: str, tag: str, taxonomy: str = "us-gaap") -> list[dict]:
    return [e for e in _entries(facts, taxonomy, tag) if e["accn"] == accn]


def _months(e: dict) -> int:
    return round((date.fromisoformat(e["end"]) - date.fromisoformat(e["start"])).days / 30.44)


def filings_from_companyfacts(symbol: str, cf: dict) -> list[Filing]:
    facts = cf.get("facts", {})
    gaap = facts.get("us-gaap", {})
    accns: dict[str, dict] = {}
    for tag in gaap:
        for e in _entries(facts, "us-gaap", tag):
            if e.get("form") in FORMS:
                a = accns.setdefault(e["accn"], {"form": e["form"], "filed": e["filed"], "end": e["end"]})
                a["end"] = max(a["end"], e["end"])

    out = []
    for accn, meta in sorted(accns.items()):
        end = meta["end"]
        flows: dict[str, tuple[int, float]] = {}
        for name, tags in FLOW.items():
            for tag in tags:
                c = [e for e in _mine(facts, accn, tag) if e["end"] == end and "start" in e]
                if c:
                    best = max(c, key=_months)
                    flows[name] = (_months(best), float(best["val"]))
                    break
        if not flows:
            continue
        months = max(m for m, _ in flows.values())
        x: dict[str, float] = {"months": float(months)}
        x.update({k: v for k, (m, v) in flows.items() if m == months})
        for name, tags in INSTANT.items():
            for tag in tags:
                c = [e for e in _mine(facts, accn, tag) if e["end"] == end and "start" not in e]
                if c:
                    x[name] = float(c[-1]["val"])
                    break
        dei = _mine(facts, accn, "EntityCommonStockSharesOutstanding", "dei")
        if dei:
            last = max(e["end"] for e in dei)
            x["shares_outstanding"] = float(sum({e["val"] for e in dei if e["end"] == last}))
        else:
            c = [e for e in _mine(facts, accn, "WeightedAverageNumberOfDilutedSharesOutstanding") if e["end"] == end]
            if c:
                x["shares_outstanding"] = float(max(c, key=_months)["val"])
        if "operating_income" in x and "d_and_a" in x:
            x["ebitda"] = x["operating_income"] + x["d_and_a"]
        out.append(Filing(
            id=accn, symbol=symbol, form=meta["form"],
            filed_at=datetime.combine(date.fromisoformat(meta["filed"]), time(17, 30), tzinfo=NY),
            period_end=date.fromisoformat(end), facts=x,
        ))
    return out


class EdgartoolsClient:
    """Thin adapter over edgartools: ticker -> raw SEC companyfacts JSON."""

    def __init__(self, identity: str):
        from edgar import set_identity

        set_identity(identity)

    def company_facts(self, ticker: str) -> dict:
        from edgar import Company
        from edgar.entity.entity_facts import download_company_facts_from_sec

        return download_company_facts_from_sec(Company(ticker).cik)


class EdgarLoader:
    def __init__(self, client):
        self.client = client  # anything with company_facts(ticker) -> companyfacts dict

    @classmethod
    def from_env(cls) -> "EdgarLoader":
        return cls(EdgartoolsClient(os.environ["EDGAR_IDENTITY"]))

    def load(self, conn: sqlite3.Connection, symbols: list[str]) -> int:
        n = 0
        for sym in symbols:
            for f in filings_from_companyfacts(sym, self.client.company_facts(sym)):
                insert_filing(conn, f, commit=False)
                n += 1
        conn.commit()
        return n
