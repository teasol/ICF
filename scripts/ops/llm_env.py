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

A machine whose route to the server differs drops talks/ops/llm.local.json next
to it; that file is git-ignored and wins. describe() reports which of the two it
read, so the override is visible rather than merely effective.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

CONFIG = Path(__file__).resolve().parents[2] / "talks" / "ops" / "llm.json"

#: Machine-local override, git-ignored, wins over CONFIG when present.
#: llm.json travels through git, but the address that works does not: the server
#: sits on the NEXGEM box, and the Slurm login node has no route to its LAN or
#: tailnet address -- it reaches port 8000 only through an SSH tunnel bound to a
#: different host. One tracked file cannot be correct on both machines, so the
#: machine that needs a different address writes it here. This is the split
#: scripts/node_env.sh already uses with node_env.local.sh, and it keeps the rule
#: from D-053 intact: the setting is still a file you can open and describe()
#: still names which one it read.
LOCAL = CONFIG.with_name("llm.local.json")

#: Fallback only. The real values live in talks/ops/llm.json, which is in git,
#: so a machine that checks out this repository gets the right address without
#: editing code. The LAN address rather than localhost: the server runs on the
#: NEXGEM box while experiments are submitted through Slurm from elsewhere, and
#: 127.0.0.1 would mean "the compute node", where nothing is listening.
DEFAULT = {
    "model": "deepseek-v4.1-flash",
    "endpoints": ["http://10.34.5.16:8000/v1/chat/completions"],
}


def source() -> Path | None:
    """Which file load() reads. None means the built-in DEFAULT."""
    for path in (LOCAL, CONFIG):
        if path.exists():
            return path
    return None


def load() -> dict:
    path = source()
    if path is not None:
        raw = json.loads(path.read_text(encoding="utf-8"))
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
    path = source()
    origin = path.name if path is not None else "built-in default"
    return f"{cfg['model']} @ {', '.join(cfg['endpoints'])} (from {origin})"


def base_url(chat_endpoint: str | None = None) -> str:
    """OpenAI-compatible base URL (the part ending in `/v1`) for a chat endpoint.

    A client such as OpenCode wants `baseURL`, not the full
    `/chat/completions` path. Deriving it here keeps that client on the same
    file as every other ops script instead of hard-coding a second address.
    """
    ep = chat_endpoint or endpoints()[0]
    for suffix in ("/chat/completions", "/completions"):
        if ep.rstrip("/").endswith(suffix):
            return ep.rstrip("/")[: -len(suffix)]
    return ep.rstrip("/")


def models_url(chat_endpoint: str | None = None) -> str:
    return base_url(chat_endpoint).rstrip("/") + "/models"


class LocalModelUnavailable(RuntimeError):
    """The configured endpoint does not serve the configured model."""


def probe_models(timeout: float = 5.0,
                 chat_endpoint: str | None = None) -> tuple[bool, list[str], str | None]:
    """GET `/v1/models` from the configured endpoint.

    Returns `(ok, ids, error)`. It never falls back to another host: the whole
    point is to notice that the local server is not there rather than post to
    an external provider and call it local.
    """
    url = models_url(chat_endpoint)
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.load(resp)
    except Exception as exc:  # noqa: BLE001 - unreachable is a state, not a crash
        return False, [], f"{url}: {exc}"
    data = payload.get("data") if isinstance(payload, dict) else None
    ids = [m.get("id") for m in (data or [])
           if isinstance(m, dict) and m.get("id")]
    return True, ids, None


def require_local_model(timeout: float = 5.0) -> str:
    """Verify the configured model is served locally, or raise.

    Callers use this before any automatic delegation so a dead tunnel stops the
    dispatch instead of silently routing the work somewhere else.
    """
    ok, ids, err = probe_models(timeout)
    if not ok:
        raise LocalModelUnavailable(f"로컬 엔드포인트 도달 불가 — {err}")
    want = model()
    if want not in ids:
        raise LocalModelUnavailable(
            f"로컬 서버에 {want!r} 없음 · 제공 모델: {', '.join(ids) or '없음'}")
    return want
