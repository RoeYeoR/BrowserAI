# BrowserAI: Build Plan

> **CV line:** Autonomous Web Navigation Agent. Python, Playwright, Claude API, LLM Function Calling, Asyncio.
> Built an autonomous web browsing agent in Python using Playwright and LLM tool calling to carry out
> complex multi-step web workflows from natural-language instructions.
> Engineered robust DOM parsing and visual feedback loops to handle dynamic single-page applications and
> interactive forms.

The goal is that **every phrase in that CV entry maps to code you can point at in an interview.**

| CV claim | Where it lives in the project |
|---|---|
| Python, Asyncio | Fully async codebase (`async_playwright`, `AsyncAnthropic`, `asyncio.gather` for parallel page snapshot and screenshot work, timeouts) |
| Playwright | `browser/` layer: session, actions, waits, tabs, iframes |
| Claude API + LLM function calling | `agent/` loop: tool schemas, a hand-written tool-use loop, tool results fed back to the model |
| Autonomous multi-step workflows from natural language | `python -m browserai "find the cheapest ... and add it to cart"` → plan → act → observe → repeat until `done` |
| Robust DOM parsing | `perception/dom.py`: injected JS extractor for interactive elements, visibility filtering, shadow DOM, iframes, stable element IDs |
| Visual feedback loops | `perception/vision.py`: annotated screenshots (numbered boxes over elements) sent to Claude, plus checking after each action whether the page actually changed |
| Dynamic SPAs | `browser/waits.py`: waits until the DOM stops changing and the network goes quiet, re-finds elements that went stale, handles client-side route changes |
| Interactive forms | Tools for text, select, checkbox and radio, custom dropdowns and comboboxes, date pickers, file upload, plus a form-fill demo |

---

## 1. Architecture

```
            ┌──────────────────────────────────────────────┐
 task ───▶  │  Agent loop (agent/loop.py)                  │
            │   1. observe  → PageState (DOM + screenshot) │
            │   2. think    → Claude (tools + history)     │
            │   3. act      → execute tool call(s)         │
            │   4. verify   → compare before/after state   │
            │   repeat until done / max_steps / error      │
            └──────┬──────────────────────┬────────────────┘
                   │                      │
        ┌──────────▼─────────┐  ┌─────────▼──────────┐
        │ perception/        │  │ browser/           │
        │  dom.py (JS)       │  │  session.py        │
        │  vision.py (SoM)   │  │  actions.py        │
        │  state.py          │  │  waits.py          │
        └────────────────────┘  └────────────────────┘
```

### Observe → think → act → verify
1. **Observe.** Inject `extract.js` into the page (and each same-origin iframe). It returns a compact list of interactive elements:
   `[12] <button> "Add to cart" (visible, in viewport)`. In parallel, take a screenshot and draw numbered boxes on it so the IDs match the list. The model gets **both**: text for precision, the image for layout and visual cues.
2. **Think.** Send the system prompt, the task, a condensed history, the current PageState and the tool definitions to Claude, which returns one or more `tool_use` blocks.
3. **Act.** Look up element ID → stored locator → Playwright action, with retries and a fallback to re-find the element.
4. **Verify.** Re-snapshot and diff (URL, DOM hash, focused element, values in form fields). If nothing changed, tell the model so in the tool result. This is the "visual feedback loop" that lets it recover from clicks that did nothing.

---

## 2. Project layout

```
BrowserAI/
├── pyproject.toml
├── .env.example              # ANTHROPIC_API_KEY=
├── README.md                 # demo GIF, architecture, results table
├── browserai/
│   ├── __main__.py           # CLI (typer + rich live step log)
│   ├── config.py             # model, max_steps, headless, allowed domains
│   ├── agent/
│   │   ├── loop.py           # core async agent loop
│   │   ├── tools.py          # tool JSON schemas + dispatch table
│   │   ├── prompts.py        # system prompt
│   │   ├── memory.py         # history condensing, keeps context small
│   │   └── safety.py         # domain allowlist, confirm before submit/pay
│   ├── browser/
│   │   ├── session.py        # launch, contexts, tabs, popups, dialogs
│   │   ├── actions.py        # click/type/select/scroll/upload/hover/keys
│   │   └── waits.py          # DOM-settle + network-idle heuristics
│   ├── perception/
│   │   ├── extract.js        # in-page DOM walker (shadow DOM, visibility, roles)
│   │   ├── dom.py            # runs extract.js, builds element index
│   │   ├── vision.py         # screenshot + set-of-marks overlay (Pillow)
│   │   └── state.py          # PageState dataclass + diffing
│   └── tracing.py            # per-run folder: screenshots, steps.jsonl, HTML report
├── evals/
│   ├── tasks.yaml            # benchmark tasks with success checks
│   ├── run_evals.py          # runs N tasks concurrently, prints success rate
│   └── sites/                # local test pages: SPA, tricky forms, shadow DOM
└── tests/                    # pytest for extractor, actions, waits
```

---

## 3. Tools the model can call

| Tool | Args | Notes |
|---|---|---|
| `navigate` | `url` | allowlist check |
| `click` | `element_id` | scroll into view, falls back to a JS click |
| `type_text` | `element_id, text, submit?` | clears field first; handles contenteditable |
| `select_option` | `element_id, value` | native `<select>` **and** custom ARIA listbox/combobox |
| `set_checkbox` | `element_id, checked` | |
| `upload_file` | `element_id, path` | sandboxed to `./uploads` |
| `scroll` | `direction, amount` or `element_id` | |
| `press_key` | `key` | Enter, Escape, Tab… |
| `hover` | `element_id` | menus that open on hover |
| `switch_tab` / `go_back` | | |
| `extract` | `query` | pull structured data out of the page |
| `wait` | `seconds` | capped |
| `done` | `success, answer` | ends the run |

Each tool returns a short text result plus the new PageState, e.g. "clicked [12]; URL changed /cart → /checkout".

---

## 4. Technical choices

- **Model:** `claude-opus-5-5` by default at `medium` effort; `--model claude-sonnet-5-5` (or `BROWSERAI_MODEL`) for a cheaper option. Server-side refusal fallbacks are on (`fallbacks: "default"`).
- **SDK:** `anthropic` Python SDK, `AsyncAnthropic`, **hand-written tool-use loop** rather than the SDK's built-in tool runner. Writing the loop yourself is the engineering the CV entry claims.
- **Prompt caching** on the system prompt and tool definitions, which cuts cost a lot on 20+ step runs.
- **Context control:** history is strictly append-only (Opus 5.5 rejects edited history that carries thinking blocks), so old PageStates and screenshots are removed **server-side** with context editing (`clear_tool_uses_20250919`: trigger 50k tokens, keep last 3 tool results). Facts the agent needs later go through `save_note`, which is excluded from clearing.
- **Playwright:** Chromium, `async_api`, persistent context optional (so you can stay logged in to your own test accounts).
- **Libraries:** `playwright`, `anthropic`, `pydantic`, `typer`, `rich`, `Pillow`, `python-dotenv`, `pytest`, `pytest-asyncio`.

---

## 5. Milestones

| # | Milestone | What "done" looks like |
|---|---|---|
| **M1** | Skeleton | `pyproject`, CLI, Playwright opens a page, Claude replies; API key loads from `.env` |
| **M2** | DOM perception | `extract.js` returns indexed interactive elements, including inside shadow DOM and iframes; pytest checks it against local test pages |
| **M3** | Agent loop + core tools | navigate/click/type/scroll/done work; it finishes a simple task end to end ("search Wikipedia for X and tell me the first sentence") |
| **M4** | Visual feedback | set-of-marks screenshots, before/after diff, "no change" recovery; tracing folder with a screenshot per step |
| **M5** | SPA robustness | DOM-settle and network waits, re-finding stale elements, popups, dialogs, new tabs; passes the local SPA test page |
| **M6** | Forms | select/combobox/checkbox/date/upload tools; finishes a multi-page form wizard on a local test site |
| **M7** | Safety + memory | domain allowlist, human confirmation before irreversible actions, history condensing |
| **M8** | Evals + polish | ~15-task benchmark (local sites plus safe public sites), success-rate table, HTML run report, README with demo GIF |

M1–M4 give a working demo; M5–M6 back up the "robust" claims; M8 gives numbers you can quote ("~85% success on a 15-task benchmark").

---

## 6. Demo tasks (for README and interviews)

1. "Go to Hacker News and tell me the top 3 stories with their point counts." (navigation + extraction)
2. "On books.toscrape.com find the cheapest book in the Travel category." (multi-step + pagination)
3. "Fill out the job application form at localhost:8000/form with this profile and stop before submitting." (forms + safety)
4. "In the local TodoMVC app add three items and mark the second as done." (SPA)
5. Shadow DOM / iframe test page (robust parsing)

---

## 7. Safety rules built into the agent
- Domain allowlist (configurable); refuses everything else.
- Pauses and asks the user before: submitting forms, purchases, sending messages, deleting anything.
- Never types passwords or payment details; the user logs in manually through a persistent profile.
- `max_steps` and spending limits on each run.

---

## 8. Open choices (defaults if unanswered)
- **Interface:** CLI with a live step log *(default)*, or also a small web dashboard showing the agent's view live.
- **GitHub:** publish as a public repo with README, GIF and eval table *(recommended for a CV project)*.
