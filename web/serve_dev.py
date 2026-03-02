#!/usr/bin/env python3
"""Dev server for mogusprotocol web UI.

Serves index.html and provides /api/sources endpoint so Pyodide can
fetch the Python source files without needing a build step.

Usage: python web/serve_dev.py
"""

import json
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKG_DIR = ROOT / "mogusprotocol"
WEB_DIR = ROOT / "web"
PORT = 8080


def collect_sources() -> dict[str, str]:
    sources: dict[str, str] = {}
    for py_file in sorted(PKG_DIR.rglob("*.py")):
        rel = py_file.relative_to(ROOT)
        sources[str(rel)] = py_file.read_text(encoding="utf-8")
    return sources


class DevHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def do_GET(self):
        if self.path == "/api/sources":
            sources = collect_sources()
            body = json.dumps(sources).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            super().do_GET()


def main():
    server = HTTPServer(("0.0.0.0", PORT), DevHandler)
    print(f"mogusprotocol dev server: http://localhost:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
