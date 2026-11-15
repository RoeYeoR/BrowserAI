import sys
from pathlib import Path

import pytest

from browserai.browser import BrowserSession
from browserai.config import Settings

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from evals.server import LocalServer  # noqa: E402

PAGES = Path(__file__).parent / "pages"


@pytest.fixture
async def page():
    async with BrowserSession(Settings(headless=True)) as session:
        yield session.page


@pytest.fixture
def open_page(page):
    async def _open(name: str):
        await page.goto((PAGES / name).as_uri(), wait_until="load")
        return page

    return _open


@pytest.fixture(scope="session")
def server():
    """Serves tests/pages over HTTP, plus /api/slow?ms=N (answers after N milliseconds)."""
    with LocalServer(PAGES) as srv:
        yield srv.url
