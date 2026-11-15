"""Run the BrowserAI benchmark: every task in tasks.yaml, several at a time, then score them.

    python -m evals.run_evals                      # all tasks, 3 in parallel
    python -m evals.run_evals --only spa_cart,todomvc --concurrency 2
    python -m evals.run_evals --model claude-sonnet-5-5

Writes evals/results/<timestamp>/ with results.json, results.md (the table for the README),
index.html (links to every run's step-by-step report) and traces/ (screenshots, logs, GIFs).
"""

from __future__ import annotations

import argparse
import asyncio
import html
import json
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.table import Table

from browserai.agent.llm import LLMClient
from browserai.agent.loop import Agent, RunResult
from browserai.agent.safety import SafetyGuard
from browserai.browser import BrowserSession
from browserai.config import get_settings
from evals.server import LocalServer

ROOT = Path(__file__).resolve().parent.parent
PAGES = ROOT / "tests" / "pages"
TASKS = Path(__file__).resolve().parent / "tasks.yaml"
console = Console()


@dataclass
class TaskResult:
    id: str
    category: str
    passed: bool
    failures: list[str] = field(default_factory=list)
    steps: int = 0
    seconds: float = 0.0
    cost_usd: float | None = None
    agent_success: bool | None = None
    answer: str = ""
    report: str | None = None
    error: str | None = None


async def check(task: dict, result: RunResult, session: BrowserSession, tools_used: list[str]) -> list[str]:
    """Return the list of failed checks (empty = pass)."""
    checks = task.get("checks", {})
    failures = []
    answer = result.answer.lower()
    if "expect_success" in checks and result.success != checks["expect_success"]:
        failures.append(f"agent reported success={result.success}, expected {checks['expect_success']}")
    for s in checks.get("answer_contains", []):
        if s.lower() not in answer:
            failures.append(f"answer lacks {s!r}")
    for s in checks.get("answer_excludes", []):
        if s.lower() in answer:
            failures.append(f"answer contains {s!r}")
    for name in checks.get("forbid_tools", []):
        if name in tools_used:
            failures.append(f"used forbidden tool {name!r}")
    if "url_contains" in checks and checks["url_contains"] not in session.page.url:
        failures.append(f"final URL {session.page.url} lacks {checks['url_contains']!r}")
    if "page_eval" in checks:
        try:
            if not await session.page.evaluate(checks["page_eval"]):
                failures.append("final page check failed")
        except Exception as e:
            failures.append(f"page check error: {str(e).splitlines()[0][:120]}")
    return failures


async def run_task(task: dict, local_url: str, llm: LLMClient, base: dict, out_dir: Path,
                   sem: asyncio.Semaphore, timeout_s: float) -> TaskResult:
    async with sem:
        settings = get_settings(
            **base,
            allowed_domains=tuple(task.get("allowed_domains") or ()),
            approval=task.get("approval", "deny"),
            max_steps=task.get("max_steps", 20),
            trace_dir=str(out_dir / "traces"),
        )
        tr = TaskResult(task["id"], task.get("category", ""), passed=False)
        t0 = time.perf_counter()
        console.print(f"[dim]▶ {task['id']}[/]")
        try:
            async with BrowserSession(settings) as session:
                await session.goto(task["start_url"].replace("{local}", local_url))
                agent = Agent(task["task"], session, llm, settings, guard=SafetyGuard(mode=settings.approval))
                result = await asyncio.wait_for(agent.run(), timeout=timeout_s)
                tools_used = [b.name for m in agent.messages if m["role"] == "assistant"
                              for b in m["content"] if getattr(b, "type", None) == "tool_use"]
                tr.failures = await check(task, result, session, tools_used)
            tr.passed = not tr.failures
            tr.steps, tr.cost_usd, tr.agent_success = result.steps, result.cost_usd, result.success
            tr.answer = result.answer
            if result.trace_dir:
                tr.report = str(Path(result.trace_dir, "report.html").relative_to(out_dir))
        except asyncio.TimeoutError:
            tr.error = f"timed out after {timeout_s:.0f}s"
        except Exception as e:
            tr.error = f"{type(e).__name__}: {str(e).splitlines()[0][:200] if str(e) else ''}"
            (out_dir / f"{task['id']}.error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        tr.seconds = time.perf_counter() - t0
        mark = "[green]✓ pass[/]" if tr.passed else "[red]✗ fail[/]"
        console.print(f"{mark} {task['id']} [dim]({tr.steps} steps, {tr.seconds:.0f}s)[/]"
                      + (f" [red]{tr.error or '; '.join(tr.failures)}[/]" if not tr.passed else ""))
        return tr


def summarize(results: list[TaskResult], model: str, wall_s: float) -> dict[str, Any]:
    passed = sum(r.passed for r in results)
    costs = [r.cost_usd for r in results if r.cost_usd is not None]
    done = [r for r in results if r.error is None]
    return {
        "model": model,
        "tasks": len(results),
        "passed": passed,
        "success_rate": passed / len(results) if results else 0.0,
        "total_cost_usd": round(sum(costs), 4),
        "avg_steps": round(sum(r.steps for r in done) / len(done), 1) if done else 0,
        "avg_seconds": round(sum(r.seconds for r in results) / len(results), 1) if results else 0,
        "wall_seconds": round(wall_s, 1),
    }


def write_outputs(out_dir: Path, results: list[TaskResult], summary: dict) -> None:
    (out_dir / "results.json").write_text(
        json.dumps({"summary": summary, "results": [r.__dict__ for r in results]}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    lines = [
        f"**{summary['passed']}/{summary['tasks']} tasks passed ({summary['success_rate']:.0%})** · "
        f"model `{summary['model']}` · avg {summary['avg_steps']} steps, {summary['avg_seconds']}s per task · "
        f"total cost ≈ ${summary['total_cost_usd']:.2f}",
        "",
        "| Task | Category | Result | Steps | Time | Cost |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        res = "✅ pass" if r.passed else f"❌ {html.escape(r.error or '; '.join(r.failures))}"
        cost = f"${r.cost_usd:.3f}" if r.cost_usd is not None else "–"
        lines.append(f"| `{r.id}` | {r.category} | {res} | {r.steps} | {r.seconds:.0f}s | {cost} |")
    (out_dir / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def row(r: TaskResult) -> str:
        link = '<a href="' + html.escape(r.report) + '">report</a>' if r.report else ""
        status = ("ok", "pass") if r.passed else ("err", "fail")
        detail = html.escape(r.error or "; ".join(r.failures) or r.answer[:140])
        return (f"<tr><td>{html.escape(r.id)}</td><td>{html.escape(r.category)}</td>"
                f"<td class={status[0]}>{status[1]}</td><td>{r.steps}</td><td>{r.seconds:.0f}s</td>"
                f"<td>{detail}</td><td>{link}</td></tr>")

    rows = "".join(row(r) for r in results)
    (out_dir / "index.html").write_text(
        "<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width'>"
        "<title>BrowserAI eval</title><style>body{font:14px system-ui;margin:24px auto;max-width:1100px;padding:0 16px}"
        "table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #ddd;padding:6px;text-align:left;"
        "vertical-align:top}.ok{color:#1a7f37;font-weight:600}.err{color:#c62828;font-weight:600}</style></head><body>"
        f"<h1>BrowserAI benchmark</h1><p>{summary['passed']}/{summary['tasks']} passed "
        f"({summary['success_rate']:.0%}) · {html.escape(summary['model'])} · ≈ ${summary['total_cost_usd']:.2f}</p>"
        f"<table><tr><th>Task</th><th>Category</th><th>Result</th><th>Steps</th><th>Time</th><th>Answer / failure</th>"
        f"<th></th></tr>{rows}</table></body></html>",
        encoding="utf-8",
    )


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", help="Comma-separated task ids")
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--model")
    ap.add_argument("--effort")
    ap.add_argument("--no-vision", action="store_true")
    ap.add_argument("--headed", action="store_true")
    ap.add_argument("--timeout", type=float, default=600, help="Seconds per task")
    args = ap.parse_args()

    tasks = yaml.safe_load(TASKS.read_text(encoding="utf-8"))
    if args.only:
        wanted = set(args.only.split(","))
        tasks = [t for t in tasks if t["id"] in wanted]
        missing = wanted - {t["id"] for t in tasks}
        if missing:
            raise SystemExit(f"unknown task ids: {', '.join(sorted(missing))}")

    out_dir = ROOT / "evals" / "results" / time.strftime("%Y%m%d-%H%M%S")
    uploads = out_dir / "uploads"
    uploads.mkdir(parents=True)
    (uploads / "sample_cv.pdf").write_bytes(b"%PDF-1.4\n% sample CV for the BrowserAI benchmark\n")

    base: dict[str, Any] = {"headless": not args.headed, "upload_dir": str(uploads)}
    if args.model:
        base["model"] = args.model
    if args.effort:
        base["effort"] = args.effort
    if args.no_vision:
        base["vision"] = False
    settings = get_settings(**base)

    console.print(f"[bold]BrowserAI benchmark[/]: {len(tasks)} tasks · {settings.model} · effort {settings.effort} "
                  f"· vision {'on' if settings.vision else 'off'} · concurrency {args.concurrency}")
    llm = LLMClient(settings)
    sem = asyncio.Semaphore(args.concurrency)
    t0 = time.perf_counter()
    try:
        with LocalServer(PAGES) as server:
            results = await asyncio.gather(
                *(run_task(t, server.url, llm, base, out_dir, sem, args.timeout) for t in tasks)
            )
    finally:
        await llm.aclose()

    summary = summarize(list(results), settings.model, time.perf_counter() - t0)
    write_outputs(out_dir, list(results), summary)

    table = Table(title=f"{summary['passed']}/{summary['tasks']} passed ({summary['success_rate']:.0%})")
    for col in ("Task", "Category", "Result", "Steps", "Time", "Cost"):
        table.add_column(col)
    for r in results:
        table.add_row(r.id, r.category, "[green]pass[/]" if r.passed else f"[red]{r.error or '; '.join(r.failures)}[/]",
                      str(r.steps), f"{r.seconds:.0f}s", f"${r.cost_usd:.3f}" if r.cost_usd is not None else "–")
    console.print(table)
    console.print(f"Total ≈ ${summary['total_cost_usd']:.2f} · wall time {summary['wall_seconds']:.0f}s · "
                  f"results in {out_dir}")


if __name__ == "__main__":
    asyncio.run(main())
