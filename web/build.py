#!/usr/bin/env python3
"""Bundle mogusprotocol Python sources into index.html for static deployment."""

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKG_DIR = ROOT / "mogusprotocol"
WEB_DIR = ROOT / "web"
TEMPLATE = WEB_DIR / "index.html"
OUTPUT_DIR = WEB_DIR / "dist"


def collect_sources() -> dict[str, str]:
    """Collect all .py files from the mogusprotocol package."""
    sources: dict[str, str] = {}
    for py_file in sorted(PKG_DIR.rglob("*.py")):
        rel = py_file.relative_to(ROOT)
        sources[str(rel)] = py_file.read_text(encoding="utf-8")
    return sources


def build():
    """Build the self-contained index.html."""
    sources = collect_sources()
    print(f"Collected {len(sources)} Python files:")
    for path in sources:
        print(f"  {path}")

    template = TEMPLATE.read_text(encoding="utf-8")
    sources_json = json.dumps(sources)
    output = template.replace("__PYTHON_SOURCES__", sources_json)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "index.html"
    out_path.write_text(output, encoding="utf-8")
    print(f"\nBuilt: {out_path} ({out_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    build()
