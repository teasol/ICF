"""Where the local model lives, in one place.

Endpoints and model names were hard-coded in four ops scripts. When the three
Qwen servers died on 2026-09-18 and a single DeepSeek server took their GPUs,
every one of those scripts kept posting to dead ports -- and, before the
failure log existed, would have done so silently.

The values live in talks/ops/llm.json so that changing servers is one edit and
so that a caller can print what it is about to use. Not environment variables:
this repository spent a day chasing ICF_* exports that no code read, and the
lesson recorded in D-053 is that a setting you cannot see is a setting you
cannot trust.
"""

from __future__ import annotations

import json
from pathlib import Path

CONFIG = Path(__file__).resolve().parents[2] / "talks" / "ops" / "llm.json"

DEFAULT = {
    "model": "deepseek-v4.1-flash",
    "endpoints": ["http://127.0.0.1:8000/v1/chat/completions"],
}


def load() -> dict:
    if CONFIG.exists():
        raw = json.loads(CONFIG.read_text(encoding="utf-8"))
        return {"model": raw.get("model", DEFAULT["model"]),
                "endpoints": list(raw.get("endpoints") or DEFAULT["endpoints"])}
    return dict(DEFAULT)


def model() -> str:
    return load()["model"]


def endpoints() -> list[str]:
    return load()["endpoints"]


def endpoint(index: int = 0) -> str:
    eps = endpoints()
    return eps[index % len(eps)]


def describe() -> str:
    cfg = load()
    return f"{cfg['model']} @ {', '.join(cfg['endpoints'])}"
