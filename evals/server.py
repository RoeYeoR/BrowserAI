"""Tiny threaded HTTP server for local test sites (used by the eval runner and pytest).

Serves a directory, plus /api/slow?ms=N which answers {"ok": true} after N milliseconds,
for pages that simulate API latency.
"""

from __future__ import annotations

import threading
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


class _Handler(SimpleHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        url = urlparse(self.path)
        if url.path == "/api/slow":
            ms = int(parse_qs(url.query).get("ms", ["500"])[0])
            time.sleep(min(ms, 30_000) / 1000)
            body = b'{"ok": true}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass
            return
        super().do_GET()

    def log_message(self, *args):
        pass


class LocalServer:
    def __init__(self, directory: str | Path, port: int = 0) -> None:
        self.httpd = ThreadingHTTPServer(("127.0.0.1", port), partial(_Handler, directory=str(directory)))
        self.httpd.daemon_threads = True
        self._thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def __enter__(self) -> LocalServer:
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
