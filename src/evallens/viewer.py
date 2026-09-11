"""Loopback-only static server for the offline trace viewer.

There is no backend. The viewer is static HTML, CSS, and JavaScript that reads a run's
``record.json`` — the same file written by `evallens demo`, with no transformation in
between, so what the viewer shows is exactly what was recorded.

The server binds loopback only and refuses anything else. It exists so a browser can fetch
local JSON without tripping over `file://` origin rules; it is not a service.
"""

from __future__ import annotations

import http.server
import json
import socketserver
import threading
import webbrowser
from functools import partial
from pathlib import Path

LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def viewer_root() -> Path:
    """Directory holding the static viewer assets."""
    candidate = Path(__file__).resolve().parents[2] / "viewer"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"viewer assets not found at {candidate}; run from a source checkout to use the viewer"
    )


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    """Serves the viewer, with the run directory mounted under /run/."""

    run_dir: Path = Path()

    def translate_path(self, path: str) -> str:
        clean = path.split("?", 1)[0].split("#", 1)[0]
        if clean.startswith("/run/"):
            relative = clean[len("/run/") :].lstrip("/")
            target = (self.run_dir / relative).resolve()
            # Refuse to serve anything outside the run directory.
            if self.run_dir.resolve() not in target.parents and target != self.run_dir.resolve():
                return str(self.run_dir / "__forbidden__")
            return str(target)
        return super().translate_path(path)

    def log_message(self, format: str, *args: object) -> None:
        return


def serve_viewer(
    run_dir: str | Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8777,
    open_browser: bool = True,
) -> int:
    """Serve the viewer against one run directory until interrupted."""
    if host not in LOOPBACK_HOSTS:
        raise ValueError(f"refusing to bind {host!r}: the viewer is loopback-only")

    run = Path(run_dir).resolve()
    record = run / "record.json"
    if not record.exists():
        print(f"no record.json in {run}")
        print("Run `evallens demo --out <dir>` first, then point the viewer at <dir>.")
        return 2
    try:
        json.loads(record.read_text())
    except json.JSONDecodeError as exc:
        print(f"{record} is not valid JSON: {exc}")
        return 2

    root = viewer_root()
    handler = partial(_QuietHandler, directory=str(root))
    _QuietHandler.run_dir = run

    class Server(socketserver.TCPServer):
        allow_reuse_address = True

    with Server((host, port), handler) as httpd:
        url = f"http://{host}:{port}/index.html"
        print(f"EvalLens viewer serving {run}")
        print(f"  {url}")
        print("  loopback only; press Ctrl+C to stop")
        if open_browser:
            threading.Timer(0.4, lambda: webbrowser.open(url)).start()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")
    return 0


__all__ = ["LOOPBACK_HOSTS", "serve_viewer", "viewer_root"]
