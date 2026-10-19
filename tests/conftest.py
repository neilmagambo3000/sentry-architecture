import pytest

from sentry.config import ROOT, load_settings
from sentry.data.store import connect


@pytest.fixture
def settings():
    return load_settings(ROOT / "config")


@pytest.fixture
def db():
    conn = connect(":memory:")
    yield conn
    conn.close()
