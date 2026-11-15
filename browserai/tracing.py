"""Per-run trace folder: what the agent saw and did at every step.

runs/<timestamp>-<task-slug>/
    task.json        task, model, settings
    step_00.jpg      annotated screenshot the model saw before step 1
    step_01.jpg      ... after step 1's actions, and so on
    steps.jsonl      one JSON record per step: thoughts, tool calls, results, page diff
    result.json      final answer, success, usage
    report.html      readable step-by-step report (see browserai.report)
    run.gif          the screenshots as an animation
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def _slug(text: str, n: int = 40) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    return s[:n].rstrip("-") or "task"


class Tracer:
    def __init__(self, root: Path | str, task: str, meta: dict[str, Any]) -> None:
        self.dir = Path(root) / f"{time.strftime('%Y%m%d-%H%M%S')}-{_slug(task)}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._write_json("task.json", {"task": task, **meta})
        self._steps = (self.dir / "steps.jsonl").open("a", encoding="utf-8")

    def screenshot(self, step: int, jpeg: bytes) -> str:
        name = f"step_{step:02d}.jpg"
        (self.dir / name).write_bytes(jpeg)
        return name

    def step(self, record: dict[str, Any]) -> None:
        self._steps.write(json.dumps(record, ensure_ascii=False, default=_default) + "\n")
        self._steps.flush()

    def finish(self, result: Any) -> None:
        self._write_json("result.json", result)
        self._steps.close()
        from browserai.report import write_gif, write_report  # local import: keeps tracing lightweight

        write_report(self.dir)
        write_gif(self.dir)

    def _write_json(self, name: str, data: Any) -> None:
        (self.dir / name).write_text(json.dumps(data, ensure_ascii=False, indent=2, default=_default), encoding="utf-8")


def _default(o: Any) -> Any:
    if is_dataclass(o):
        return asdict(o)
    return str(o)
