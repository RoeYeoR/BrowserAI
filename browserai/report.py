"""Turn a run's trace folder into a self-contained HTML report and an animated GIF.

    report.html  every step: the annotated screenshot the model saw, its notes, the tool
                 calls with results, and the page changes detected afterwards
    run.gif      the screenshots as a slideshow (handy for READMEs and demos)
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from PIL import Image

_CSS = """
:root { --bg:#f6f7f9; --card:#fff; --text:#1d2330; --muted:#677084; --ok:#1a7f37; --err:#c62828;
        --accent:#3b5bdb; --border:#e3e6ec; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#14171c; --card:#1d2128; --text:#e6e8ec; --muted:#9aa3b2; --ok:#4cc26a; --err:#ff6b6b;
          --accent:#7f9cf5; --border:#2c313a; } }
* { box-sizing: border-box; }
body { margin:0; padding:24px 16px; background:var(--bg); color:var(--text);
       font:15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1100px; margin: 0 auto; }
h1 { font-size: 22px; margin: 0 0 4px; }
.meta { color: var(--muted); font-size: 13px; margin-bottom: 16px; }
.card { background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 16px; margin: 12px 0; }
.answer { white-space: pre-wrap; }
.badge { display:inline-block; padding:2px 8px; border-radius:999px; font-size:12px; font-weight:600; color:#fff; }
.ok { background: var(--ok); } .err { background: var(--err); }
.step { display: grid; grid-template-columns: minmax(0, 1.4fr) minmax(0, 1fr); gap: 16px; }
@media (max-width: 760px) { .step { grid-template-columns: 1fr; } }
.step img { width: 100%; border-radius: 6px; border: 1px solid var(--border); }
.step h2 { font-size: 15px; margin: 0 0 8px; }
.url { color: var(--muted); font-size: 12px; word-break: break-all; }
ul { padding-left: 18px; margin: 6px 0; }
li { margin: 3px 0; }
.thought { color: var(--accent); font-style: italic; }
.call code { font-weight: 600; }
.res-ok { color: var(--ok); } .res-err { color: var(--err); }
.diff { color: var(--muted); font-size: 13px; }
code { font: 13px ui-monospace, SFMono-Regular, Consolas, monospace; word-break: break-word; }
"""


def _esc(v: Any) -> str:
    return html.escape(str(v))


def _load(trace: Path) -> tuple[dict, list[dict], dict]:
    task = json.loads((trace / "task.json").read_text(encoding="utf-8"))
    steps = [json.loads(line) for line in (trace / "steps.jsonl").read_text(encoding="utf-8").splitlines() if line]
    result_file = trace / "result.json"
    result = json.loads(result_file.read_text(encoding="utf-8")) if result_file.exists() else {}
    return task, steps, result


def _step_html(rec: dict) -> str:
    items = []
    for ev in rec.get("events", []):
        kind = ev.get("kind")
        if kind == "thought":
            items.append(f'<li class="thought">{_esc(ev["text"])}</li>')
        elif kind == "tool_call":
            args = ", ".join(f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in ev["input"].items())
            items.append(f'<li class="call">→ <code>{_esc(ev["name"])}</code>({_esc(args)})</li>')
        elif kind == "tool_result":
            cls = "res-err" if ev.get("is_error") else "res-ok"
            first = (ev.get("text") or "").splitlines()[0][:300] if ev.get("text") else ""
            items.append(f'<li class="{cls}">{"✗" if ev.get("is_error") else "✓"} {_esc(first)}</li>')
        elif kind == "verify":
            text = "; ".join(ev.get("changes") or []) or "⚠ no change detected"
            items.append(f'<li class="diff">Δ {_esc(text[:600])}</li>')
        elif kind == "context":
            items.append(f'<li class="diff">context editing cleared {_esc(ev.get("cleared_tool_uses"))} old results</li>')
    title = "Initial page" if rec["step"] == 0 else f"Step {rec['step']}"
    shot = f'<img src="{_esc(rec["screenshot"])}" alt="{title} screenshot" loading="lazy">' if rec.get("screenshot") else ""
    return (
        f'<section class="card step"><div>{shot}</div><div><h2>{title}</h2>'
        f'<div class="url">{_esc(rec.get("url", ""))}</div><ul>{"".join(items)}</ul></div></section>'
    )


def write_report(trace_dir: str | Path) -> Path:
    trace = Path(trace_dir)
    task, steps, result = _load(trace)
    ok = result.get("success")
    badge = (f'<span class="badge {"ok" if ok else "err"}">{"Success" if ok else "Not completed"}</span>'
             if result else "")
    usage = result.get("usage") or {}
    cost = result.get("cost_usd")
    meta = " · ".join(filter(None, [
        _esc(task.get("model", "")),
        f"{result.get('steps')} steps" if result else "",
        f"{result.get('seconds', 0):.0f}s" if result else "",
        f"~${cost:.3f}" if cost is not None else "",
        f"{usage.get('cache_read_tokens', 0):,} cached input tokens" if usage else "",
    ]))
    safety = "".join(
        f"<li>{_esc(e.get('blocked') or (('approved' if e.get('approved') else 'refused') + ': ' + e.get('action', '')))}</li>"
        for e in result.get("safety_log", [])
    )
    body = [
        f"<h1>{_esc(task['task'])}</h1>",
        f'<div class="meta">{badge} {meta}</div>',
        f'<section class="card"><strong>Answer</strong><div class="answer">{_esc(result.get("answer", "(run did not finish)"))}</div>'
        + (f"<strong>Safety</strong><ul>{safety}</ul>" if safety else "") + "</section>",
        *(_step_html(r) for r in steps),
    ]
    page = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>BrowserAI run</title><style>{_CSS}</style></head>"
        f"<body><main>{''.join(body)}</main></body></html>"
    )
    out = trace / "report.html"
    out.write_text(page, encoding="utf-8")
    return out


def write_gif(trace_dir: str | Path, width: int = 960, frame_ms: int = 1400) -> Path | None:
    trace = Path(trace_dir)
    shots = sorted(trace.glob("step_*.jpg"))
    if not shots:
        return None
    frames = []
    for p in shots:
        img = Image.open(p).convert("RGB")
        if img.width > width:
            img = img.resize((width, round(img.height * width / img.width)), Image.LANCZOS)
        frames.append(img.convert("P", palette=Image.ADAPTIVE, colors=128))
    out = trace / "run.gif"
    durations = [frame_ms] * (len(frames) - 1) + [frame_ms * 2]  # linger on the final state
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)
    return out
