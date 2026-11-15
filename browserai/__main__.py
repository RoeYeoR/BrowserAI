"""Command-line entry point: `browserai ...` or `python -m browserai ...`."""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm

from browserai import __version__
from browserai.agent.llm import LLMClient, LLMError
from browserai.agent.loop import Agent, Event
from browserai.agent.safety import SafetyGuard
from browserai.browser import BrowserSession
from browserai.config import get_settings
from browserai.perception import DomExtractor
from browserai.perception.vision import annotated_screenshot

app = typer.Typer(add_completion=False, help="BrowserAI - autonomous web navigation agent")
console = Console()


@app.callback()
def main() -> None:
    """BrowserAI command-line interface."""


@app.command()
def version() -> None:
    """Print the BrowserAI version."""
    console.print(f"browserai {__version__}")


@app.command()
def smoke(
    url: str = typer.Option("https://www.wikipedia.org", help="Page to open"),
    headless: bool = typer.Option(None, "--headless/--headed", help="Override BROWSERAI_HEADLESS"),
    model: str = typer.Option(None, help="Override BROWSERAI_MODEL"),
    no_llm: bool = typer.Option(False, "--no-llm", help="Only test the browser, skip Claude"),
) -> None:
    """M1 smoke test: open a page with Playwright and have Claude summarize it."""
    settings = get_settings(headless=headless, model=model)
    try:
        asyncio.run(_smoke(url, settings, no_llm))
    except LLMError as e:
        console.print(f"[red]LLM error:[/] {e}")
        raise typer.Exit(1)


async def _smoke(url: str, settings, no_llm: bool) -> None:
    # Create the client first so missing credentials fail fast, before launching a browser.
    llm = None if no_llm else LLMClient(settings)

    t0 = time.perf_counter()
    async with BrowserSession(settings) as session:
        await session.goto(url)
        title = await session.title()
        text = await session.visible_text()
        console.print(f"[green]✓[/] Browser opened [bold]{session.page.url}[/] "
                      f"({time.perf_counter() - t0:.1f}s) - title: {title!r}")

    if llm is None:
        return

    try:
        t1 = time.perf_counter()
        summary = await llm.ask(
            f"Page title: {title}\nPage text:\n{text}\n\n"
            "In one sentence, what is this page for?",
            system="You are the perception module of a web-browsing agent. Be concise.",
        )
    finally:
        await llm.aclose()
    console.print(f"[green]✓[/] Claude replied via {settings.model} ({time.perf_counter() - t1:.1f}s)")
    console.print(Panel(summary, title="Claude's summary", expand=False))


@app.command()
def run(
    task: str = typer.Argument(..., help="What the agent should do, in plain English"),
    url: str = typer.Option(None, "--url", "-u", help="Optional page to start on"),
    headless: bool = typer.Option(None, "--headless/--headed", help="Override BROWSERAI_HEADLESS"),
    model: str = typer.Option(None, help="Override BROWSERAI_MODEL"),
    effort: str = typer.Option(None, help="low | medium | high | xhigh | max"),
    max_steps: int = typer.Option(None, help="Override BROWSERAI_MAX_STEPS"),
    vision: bool = typer.Option(None, "--vision/--no-vision", help="Send annotated screenshots (default on)"),
    trace: bool = typer.Option(True, "--trace/--no-trace", help="Save screenshots and a step log under runs/"),
    approval: str = typer.Option(None, help="Risky actions (submit, buy, delete, upload...): ask | allow | deny"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Approve risky actions automatically (= --approval allow)"),
    allow_domain: list[str] = typer.Option(None, "--allow-domain", "-d",
                                           help="Only allow navigation to this domain (repeatable)"),
    max_cost: float = typer.Option(None, help="Stop when the estimated cost exceeds this many USD"),
) -> None:
    """Run the agent on a natural-language task."""
    settings = get_settings(headless=headless, model=model, effort=effort, max_steps=max_steps, vision=vision,
                            approval="allow" if yes else approval, max_cost_usd=max_cost,
                            allowed_domains=tuple(allow_domain) if allow_domain else None)
    if not trace:
        settings = get_settings(**{**settings.__dict__, "trace_dir": ""})
    if settings.approval == "ask" and not sys.stdin.isatty():
        console.print("[yellow]No terminal to ask for approvals: risky actions will be refused "
                      "(use --yes to approve them automatically).[/]")
        settings = get_settings(**{**settings.__dict__, "approval": "deny"})
    try:
        result = asyncio.run(_run(task, url, settings))
    except LLMError as e:
        console.print(f"[red]LLM error:[/] {e}")
        raise typer.Exit(1)
    raise typer.Exit(0 if result.success else 2)


def _print_event(event: Event) -> None:
    d = event.data
    if event.kind == "step":
        console.rule(f"[bold]Step {d['step']}[/]  [dim]{d['url']}[/]", align="left")
    elif event.kind == "thought":
        console.print(f"[italic cyan]💭 {d['text'].strip()}[/]", highlight=False)
    elif event.kind == "tool_call":
        args = ", ".join(f"{k}={v!r}" for k, v in d["input"].items())
        console.print(f"[bold yellow]→ {d['name']}[/]({args})", highlight=False)
    elif event.kind == "tool_result":
        first = d["text"].splitlines()[0] if d["text"] else ""
        style = "red" if d["is_error"] else "green"
        console.print(f"  [{style}]{'✗' if d['is_error'] else '✓'} {first[:160]}[/]", highlight=False)
    elif event.kind == "verify":
        if d["changed"]:
            summary = "; ".join(d["changes"])
            console.print(f"  [dim]Δ {summary[:180]}[/]", highlight=False)
        else:
            console.print("  [bold magenta]⚠ no change detected on the page[/]")
    elif event.kind == "context":
        console.print(f"[dim]  (context editing cleared {d['cleared_tool_uses']} old tool results, "
                      f"{d['cleared_input_tokens']} tokens)[/]")


async def _ask_user(description: str) -> bool:
    """Human-in-the-loop approval for a risky action (runs the blocking prompt off the event loop)."""
    def prompt() -> bool:
        console.print()
        console.print(f"[bold red]🛡 The agent wants to {description}.[/]")
        return Confirm.ask("Allow this action?", default=False)
    return await asyncio.to_thread(prompt)


async def _run(task: str, url: str | None, settings):
    console.print(Panel(task, title=f"Task · {settings.model} · effort {settings.effort}", expand=False))
    scope = ", ".join(settings.allowed_domains) if settings.allowed_domains else "any domain"
    console.print(f"[dim]Approvals: {settings.approval} · navigation: {scope} · "
                  f"budget: ${settings.max_cost_usd:.2f}, {settings.max_steps} steps[/]")
    llm = LLMClient(settings)
    try:
        async with BrowserSession(settings) as session:
            if url:
                await session.goto(url)
            guard = SafetyGuard(mode=settings.approval, approver=_ask_user)
            result = await Agent(task, session, llm, settings, on_event=_print_event, guard=guard).run()
    finally:
        await llm.aclose()

    u = result.usage
    style = "green" if result.success else "red"
    console.print(Panel(result.answer, title=f"[{style}]{'Success' if result.success else 'Not completed'}[/]",
                        expand=False))
    if result.notes:
        console.print("[dim]Notes:[/] " + " | ".join(result.notes), highlight=False)
    console.print(
        f"[dim]{result.steps} steps · {result.seconds:.0f}s · stop: {result.stop_reason} · "
        f"tokens in {u.input_tokens:,} (cache read {u.cache_read_tokens:,}, write {u.cache_write_tokens:,}) "
        f"out {u.output_tokens:,}"
        + (f" · ~${result.cost_usd:.3f}" if result.cost_usd is not None else "") + "[/]"
    )
    for entry in result.safety_log:
        if "blocked" in entry:
            console.print(f"[dim]🛡 blocked: {entry['blocked'][:150]}[/]", highlight=False)
        else:
            verdict = "approved" if entry["approved"] else "refused"
            console.print(f"[dim]🛡 {verdict} ({entry['mode']}): {entry['action']}[/]", highlight=False)
    if result.trace_dir:
        console.print(f"[dim]Trace: {result.trace_dir}[/]")
    return result


@app.command()
def report(trace_dir: str = typer.Argument(..., help="A run folder under runs/")) -> None:
    """(Re)build report.html and run.gif for a saved run."""
    from browserai.report import write_gif, write_report

    console.print(f"Report: {write_report(trace_dir)}")
    gif = write_gif(trace_dir)
    console.print(f"GIF: {gif}" if gif else "[dim]No screenshots in this run (it used --no-vision).[/]")


@app.command()
def inspect(
    url: str = typer.Argument(..., help="Page to open"),
    headless: bool = typer.Option(True, "--headless/--headed"),
    wait: float = typer.Option(1.0, help="Seconds to wait after load for SPAs to render"),
    screenshot: str = typer.Option(None, "--screenshot", "-s", help="Also save the annotated screenshot here"),
) -> None:
    """Show exactly what the agent will see on a page: the indexed interactive elements."""
    settings = get_settings(headless=headless)
    asyncio.run(_inspect(url, settings, wait, screenshot))


async def _inspect(url: str, settings, wait: float, screenshot: str | None) -> None:
    async with BrowserSession(settings) as session:
        await session.goto(url)
        if wait:
            await asyncio.sleep(wait)
        t0 = time.perf_counter()
        state = await DomExtractor().snapshot(session.page)
        ms = (time.perf_counter() - t0) * 1000
        if screenshot:
            Path(screenshot).write_bytes(await annotated_screenshot(session.page, state))
            console.print(f"[dim]Annotated screenshot saved to {screenshot}[/]")
    console.print(state.to_prompt(), markup=False, highlight=False)
    frames = len({e.frame_url for e in state.elements if e.frame_url})
    shadow = sum(e.in_shadow for e in state.elements)
    console.print(
        f"\n[dim]{len(state.elements)} elements ({shadow} in shadow DOM, from {frames} iframe(s)) "
        f"extracted in {ms:.0f} ms[/]"
    )


if __name__ == "__main__":
    app()
