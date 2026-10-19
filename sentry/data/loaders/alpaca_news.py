"""Alpaca News -> store (write-only). Network access only through the injected client.

first_seen: historical loads use the vendor's created_at (earliest known publish time); live ingestion
passes `arrival_time`, the moment we actually received the batch.
"""
from __future__ import annotations

import os
import sqlite3
from datetime import datetime

from sentry.data.writes import insert_article
from sentry.types import Article


class AlpacaNewsLoader:
    def __init__(self, client):
        self.client = client  # alpaca.data.historical.news.NewsClient or a fake with get_news

    @classmethod
    def from_env(cls) -> "AlpacaNewsLoader":
        from alpaca.data.historical.news import NewsClient

        return cls(NewsClient(os.environ["ALPACA_PAPER_KEY_ID"], os.environ["ALPACA_PAPER_SECRET"]))

    def load(self, conn: sqlite3.Connection, symbols: list[str], start: datetime, end: datetime,
             arrival_time: datetime | None = None) -> int:
        from alpaca.data.requests import NewsRequest

        n, token = 0, None
        while True:
            # alpaca-py pages internally when limit is unset; next_page_token is honoured in case it is returned
            req = NewsRequest(symbols=",".join(symbols), start=start, end=end, include_content=True,
                              page_token=token)
            page = self.client.get_news(req)
            for item in page.data.get("news", []):
                insert_article(conn, Article(
                    id=str(item.id), symbols=tuple(item.symbols),
                    first_seen=arrival_time or item.created_at, published=item.created_at,
                    source=item.source, headline=item.headline, body=item.content or item.summary or None,
                    url=item.url or "",
                ), commit=False)
                n += 1
            token = page.next_page_token
            if not token:
                break
        conn.commit()
        return n
