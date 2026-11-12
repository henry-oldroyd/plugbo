"""Unload an Ollama model from VRAM (same effect as `ollama stop <model>`).

    python -m llm.ollama_unload qwen3-coder:30b [base_url]
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request

from .ollama_client import DEFAULT_BASE_URL


def unload_model(model: str, base_url: str | None = None, timeout: float = 30.0) -> bool:
    """Ask Ollama to evict `model` (keep_alive=0). Never raises; returns success."""
    try:
        root = (base_url or os.environ.get("OLLAMA_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        if root.endswith("/v1"):
            root = root[:-3]
        req = urllib.request.Request(
            f"{root}/api/generate",
            data=json.dumps({"model": model, "keep_alive": 0}).encode(),
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=timeout).read()
        return True
    except Exception as exc:  # noqa: BLE001 - cleanup must never mask the real result
        print(f"[ollama_unload] could not unload {model}: {exc}", file=sys.stderr)
        return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: python -m llm.ollama_unload MODEL [BASE_URL]")
    ok = unload_model(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
    print(f"unloaded {sys.argv[1]}" if ok else "unload failed")
