"""Tests for DOM perception: the in-page extractor and the Python-side merging/indexing."""

from browserai.perception import DomExtractor, PageState


def by_name(state: PageState, name: str):
    matches = [e for e in state.elements if e.name == name]
    assert matches, f"{name!r} not found in:\n{state.to_prompt()}"
    assert len(matches) == 1, f"{name!r} found {len(matches)} times"
    return matches[0]


def names(state: PageState) -> set[str]:
    return {e.name for e in state.elements}


# ---------- basic elements, names and properties ----------

async def test_finds_standard_controls(open_page):
    page = await open_page("basic.html")
    state = await DomExtractor().snapshot(page)

    assert by_name(state, "About us").kind == "link"
    assert by_name(state, "About us").href == "/about"
    assert by_name(state, "Go").kind == "button"
    assert by_name(state, "Send").kind == "button"  # <input type=submit>
    assert by_name(state, "Message").kind == "textarea"  # name falls back to placeholder


async def test_accessible_names(open_page):
    page = await open_page("basic.html")
    state = await DomExtractor().snapshot(page)

    assert by_name(state, "Close dialog").kind == "button"  # aria-label on icon button
    email = by_name(state, "Email")  # <label for>
    assert email.kind == "input[email]"
    assert email.placeholder == "you@example.com"
    assert email.required
    assert by_name(state, "Password").kind == "input[password]"  # wrapping <label>


async def test_states_and_values(open_page):
    page = await open_page("basic.html")
    state = await DomExtractor().snapshot(page)

    assert by_name(state, "Disabled").disabled
    assert by_name(state, "I agree").checked is True
    country = by_name(state, "Country")
    assert country.kind == "select"
    assert country.value == "UK"
    assert country.options == ["Israel", "UK", "USA"]


async def test_password_value_is_masked(open_page):
    page = await open_page("basic.html")
    state = await DomExtractor().snapshot(page)
    pw = by_name(state, "Password")
    assert pw.value == "••••••"
    assert "secret123" not in state.to_prompt()


async def test_hidden_elements_are_excluded(open_page):
    page = await open_page("basic.html")
    state = await DomExtractor().snapshot(page)
    found = names(state)
    for hidden in [
        "HiddenDisplay", "HiddenVisibility", "Zero", "Transparent",
        "InsideHidden", "InsideTransparent", "Anchor without href",
    ]:
        assert hidden not in found, f"{hidden} should not be extracted"
    assert not any(e.kind == "input[hidden]" for e in state.elements)


async def test_offscreen_elements_are_flagged(open_page):
    page = await open_page("basic.html")
    state = await DomExtractor().snapshot(page)
    assert by_name(state, "Go").in_viewport
    far = by_name(state, "Far below")
    assert not far.in_viewport
    prompt = state.to_prompt()
    assert prompt.index("Below the viewport") < prompt.index('"Far below"')


# ---------- non-standard clickables ----------

async def test_custom_clickables(open_page):
    page = await open_page("clickables.html")
    state = await DomExtractor().snapshot(page)

    assert by_name(state, "Card with onclick").kind == "clickable"
    assert by_name(state, "Role button").kind == "button"
    assert by_name(state, "ARIA checkbox").checked is True
    assert by_name(state, "Focusable span")
    assert "Not focusable by tab" not in names(state)
    editable = by_name(state, "Editable bold text")
    assert editable.kind == "editable"
    assert by_name(state, "Menu").expanded is False


async def test_cursor_pointer_counted_once(open_page):
    page = await open_page("clickables.html")
    state = await DomExtractor().snapshot(page)
    assert by_name(state, "Pointer div inner span")
    assert "inner span" not in names(state)  # child inherits cursor, must not be listed


async def test_invisible_checkbox_uses_label(open_page):
    page = await open_page("clickables.html")
    state = await DomExtractor().snapshot(page)
    fancy = by_name(state, "Fancy checkbox")
    assert fancy.kind == "checkbox"
    assert fancy.via_label
    assert fancy.box.w > 0


async def test_transparent_unlabeled_checkbox(open_page):
    page = await open_page("clickables.html")
    state = await DomExtractor().snapshot(page)
    toggle = by_name(state, "walk dog")  # named from the adjacent label text
    assert toggle.kind == "checkbox"
    assert toggle.checked is False
    assert not toggle.via_label
    assert "Transparent button" not in names(state)  # the exception is only for checkbox/radio/file


# ---------- occlusion ----------

async def test_occluded_by_modal(open_page):
    page = await open_page("overlay.html")
    state = await DomExtractor().snapshot(page)
    assert by_name(state, "Covered button").occluded
    assert not by_name(state, "OK").occluded
    assert "covered by another element" in state.to_prompt()


# ---------- shadow DOM ----------

async def test_shadow_dom(open_page):
    page = await open_page("shadow.html")
    state = await DomExtractor().snapshot(page)

    assert not by_name(state, "Light button").in_shadow
    sbtn = by_name(state, "Shadow button")
    assert sbtn.in_shadow
    assert by_name(state, "Shadow label").in_shadow  # <label for> resolved inside the shadow root
    assert by_name(state, "Deep link").in_shadow  # nested shadow root


async def test_shadow_element_is_locatable(open_page):
    page = await open_page("shadow.html")
    state = await DomExtractor().snapshot(page)
    sbtn = by_name(state, "Shadow button")
    # Playwright CSS selectors pierce open shadow roots, so the tag we set is enough.
    assert await page.locator(sbtn.selector).inner_text() == "Shadow button"


# ---------- iframes ----------

async def test_iframes(open_page):
    page = await open_page("iframes.html")
    state = await DomExtractor().snapshot(page)

    assert by_name(state, "Top button").frame_url is None
    fbtn = by_name(state, "Frame button")
    assert fbtn.frame_url is not None
    # Coordinates are translated to page space: iframe at left=100, top=300, button margin 10px.
    assert 105 <= fbtn.box.x <= 115
    assert 305 <= fbtn.box.y <= 315
    assert by_name(state, "Frame input").frame_url is not None
    assert by_name(state, "Child frame link").box.x >= 600
    assert "Hidden frame button" not in names(state)


async def test_iframe_element_is_locatable(open_page):
    page = await open_page("iframes.html")
    state = await DomExtractor().snapshot(page)
    fbtn = by_name(state, "Frame button")
    assert await fbtn.frame.locator(fbtn.selector).inner_text() == "Frame button"


# ---------- stable indices ----------

async def test_indices_stable_across_dom_changes(open_page):
    page = await open_page("dynamic.html")
    ex = DomExtractor()
    s1 = await ex.snapshot(page)
    a1, b1 = by_name(s1, "A").index, by_name(s1, "B").index

    # Simulate an SPA re-render: insert a new element before the existing ones.
    await page.evaluate(
        "() => { const n = document.createElement('button'); n.textContent = 'New';"
        " document.getElementById('list').prepend(n); }"
    )
    s2 = await ex.snapshot(page)
    assert by_name(s2, "A").index == a1
    assert by_name(s2, "B").index == b1
    assert by_name(s2, "New").index not in (a1, b1)


async def test_indices_reset_on_navigation(open_page):
    page = await open_page("dynamic.html")
    ex = DomExtractor()
    await ex.snapshot(page)
    await page.evaluate("() => document.getElementById('list').prepend(document.createElement('button'))")
    await ex.snapshot(page)

    await page.reload(wait_until="load")
    s = await ex.snapshot(page)
    assert sorted(e.index for e in s.elements) == [1, 2]


async def test_fingerprint_changes_with_state(open_page):
    page = await open_page("basic.html")
    ex = DomExtractor()
    f1 = (await ex.snapshot(page)).fingerprint()
    assert (await ex.snapshot(page)).fingerprint() == f1
    await page.fill("#email", "a@b.co")
    assert (await ex.snapshot(page)).fingerprint() != f1
