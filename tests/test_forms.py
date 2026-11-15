"""Form tools on a multi-step job-application wizard: native and custom dropdowns, autocomplete,
radios, switches, dates, validation errors, uploads (incl. a button-triggered file chooser)."""

import datetime as dt
import json

import pytest

from browserai.agent.safety import SafetyGuard
from browserai.agent.tools import ToolExecutor
from browserai.browser import BrowserSession
from browserai.browser.forms import best_match, format_date
from browserai.config import Settings
from browserai.perception import DomExtractor

from conftest import PAGES


# ---------- pure helpers ----------

def test_best_match():
    opts = ["Python", "JavaScript", "Java", "Go"]
    assert best_match("java", opts) == 2  # exact beats prefix
    assert best_match("pyth", opts) == 0  # unique prefix
    assert best_match("script", opts) == 1  # unique substring
    assert best_match("ja", opts) is None  # ambiguous prefix
    assert best_match("rust", opts) is None
    assert best_match("  united   KINGDOM ", ["United Kingdom"]) == 0


def test_format_date():
    d = dt.datetime(2025, 6, 7, 9, 5)
    assert format_date(d, "DD/MM/YYYY") == "07/06/2025"
    assert format_date(d, "mm/dd/yyyy") == "06/07/2025"  # lowercase mm = month without hours
    assert format_date(d, "YYYY-MM-DD HH:mm") == "2025-06-07 09:05"
    assert format_date(d, "DD.MM.YY") == "07.06.25"


# ---------- browser fixtures ----------

@pytest.fixture
async def wiz(tmp_path):
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    (uploads / "cv.pdf").write_bytes(b"%PDF-1.4 test")
    (uploads / "portfolio.zip").write_bytes(b"PK test")
    (tmp_path / "secret.txt").write_text("do not upload")

    async with BrowserSession(Settings(headless=True, upload_dir=str(uploads))) as session:
        await session.page.goto((PAGES / "wizard.html").as_uri(), wait_until="load")
        ex = ToolExecutor(session, SafetyGuard(mode="allow"))  # flows the user approved
        extractor = DomExtractor()

        async def snap():
            ex.state = await extractor.snapshot(session.page)
            return ex.state

        async def do(tool, **args):
            r = await ex.execute(tool, args)
            await session.settle()
            await snap()
            return r

        ex.snap, ex.do = snap, do
        await snap()
        yield ex


def idx(state, name):
    hits = [e.index for e in state.elements if e.name == name]
    assert hits, f"{name!r} not in:\n{state.to_prompt()}"
    return hits[0]


async def js(ex, expr):
    return await ex.page.evaluate(f"() => {expr}")


async def fill_step1(wiz, email="roi@example.com"):
    s = wiz.state
    await wiz.do("type_text", index=idx(s, "Full name"), text="Roi Test")
    await wiz.do("type_text", index=idx(s, "Email"), text=email)
    await wiz.do("select_option", index=idx(s, "Country"), option="Israel")
    await wiz.do("set_checkbox", index=idx(s, "1-3 years"), checked=True)
    return await wiz.do("click", index=idx(wiz.state, "Next"))


# ---------- step 1: native select, radios, validation ----------

async def test_native_select_matching(wiz):
    country = idx(wiz.state, "Country")
    r = await wiz.do("select_option", index=country, option="united k")
    assert not r.is_error and "United Kingdom" in r.text
    assert await js(wiz, "document.getElementById('country').value") == "uk"

    assert not (await wiz.do("select_option", index=country, option="de")).is_error  # by value
    assert await js(wiz, "document.getElementById('country').value") == "de"

    r = await wiz.do("select_option", index=country, option="united")
    assert r.is_error and "United Kingdom" in r.text and "United States" in r.text  # lists options


async def test_radio_buttons(wiz):
    r = await wiz.do("set_checkbox", index=idx(wiz.state, "3+ years"), checked=True)
    assert not r.is_error and "now checked" in r.text
    r = await wiz.do("set_checkbox", index=idx(wiz.state, "3+ years"), checked=False)
    assert r.is_error and "radio" in r.text
    r = await wiz.do("set_checkbox", index=idx(wiz.state, "Country"), checked=True)
    assert r.is_error and "not a checkbox" in r.text


async def test_validation_error_is_visible_then_fixed(wiz):
    await fill_step1(wiz, email="not-an-email")
    email = next(e for e in wiz.state.elements if e.name == "Email")
    assert email.invalid and email.error == "Enter a valid email address"
    assert "INVALID: Enter a valid email address" in wiz.state.to_prompt()

    await wiz.do("type_text", index=email.index, text="roi@example.com")
    await wiz.do("click", index=idx(wiz.state, "Next"))
    assert await js(wiz, "document.getElementById('stepinfo').textContent") == "Step 2 of 3"


# ---------- full wizard: custom widgets, dates, uploads, submit ----------

async def test_complete_wizard(wiz):
    await fill_step1(wiz)
    s = wiz.state

    # Step 2
    r = await wiz.do("set_date", index=idx(s, "Earliest start date"), date="2026-01-15")
    assert not r.is_error and "2026-01-15" in r.text
    r = await wiz.do("set_date", index=idx(s, "Graduation date"), date="2025-06-30")  # placeholder DD/MM/YYYY
    assert '"30/06/2025"' in r.text

    r = await wiz.do("select_option", index=idx(s, "Main skill"), option="java")  # custom listbox
    assert not r.is_error and '"Java"' in r.text, r.text
    r = await wiz.do("select_option", index=idx(wiz.state, "City"), option="Haifa")  # async autocomplete
    assert not r.is_error and '"Haifa"' in r.text, r.text

    r = await wiz.do("set_checkbox", index=idx(wiz.state, "Open to remote work"), checked=True)  # role=switch
    assert "now checked" in r.text
    r = await wiz.do("set_checkbox", index=idx(wiz.state, "Open to remote work"), checked=True)
    assert "already checked" in r.text
    await wiz.do("set_checkbox", index=idx(wiz.state, "Send me job alerts"), checked=True)
    await wiz.do("click", index=idx(wiz.state, "Next"))

    # Step 3
    s = wiz.state
    r = await wiz.do("upload_file", index=idx(s, "CV (PDF)"), filename="cv.pdf")
    assert not r.is_error, r.text
    r = await wiz.do("upload_file", index=idx(s, "Add portfolio"), filename="portfolio.zip")  # file chooser
    assert not r.is_error, r.text
    await wiz.do("type_text", index=idx(wiz.state, "Cover letter"), text="Hello!")

    r = await wiz.do("click", index=idx(wiz.state, "Submit application"))
    assert "Please accept the terms" in await js(wiz, "document.getElementById('submit-err').textContent")
    r = await wiz.do("set_checkbox", index=idx(wiz.state, "I accept the terms"), checked=True)  # opacity:0 + label
    assert not r.is_error, r.text
    await wiz.do("click", index=idx(wiz.state, "Submit application"))

    result = json.loads(await js(wiz, "document.getElementById('result').textContent"))
    assert result == {
        "name": "Roi Test", "email": "roi@example.com", "country": "il", "experience": "1-3",
        "start": "2026-01-15", "graduation": "30/06/2025", "skill": "Java", "city": "Haifa",
        "remote": True, "alerts": True, "cv": "cv.pdf", "portfolio": "portfolio.zip",
        "cover": "Hello!", "terms": True,
    }


async def test_upload_is_sandboxed(wiz):
    await fill_step1(wiz)
    await wiz.do("click", index=idx(wiz.state, "Next"))
    cv = idx(wiz.state, "CV (PDF)")
    r = await wiz.do("upload_file", index=cv, filename="../secret.txt")
    assert r.is_error and "uploads folder" in r.text
    r = await wiz.do("upload_file", index=cv, filename="missing.pdf")
    assert r.is_error and "cv.pdf" in r.text and "portfolio.zip" in r.text  # lists what is available


async def test_bad_date_and_wrong_option(wiz):
    await fill_step1(wiz)
    s = wiz.state
    r = await wiz.do("set_date", index=idx(s, "Earliest start date"), date="next tuesday")
    assert r.is_error and "ISO" in r.text
    r = await wiz.do("select_option", index=idx(s, "Main skill"), option="Rust")
    assert r.is_error and "Python" in r.text  # shows the visible options
