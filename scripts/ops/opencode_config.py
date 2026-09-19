#!/usr/bin/env python3
"""Point OpenCode at the local vLLM server, not an external provider.

Measured defect this replaces. `routine_opencode.sh` passed an OpenCode
*provider* string naming an external service, not the local vLLM. With no
project config, OpenCode resolved that provider over HTTPS. Copying
`llm.local.json` into the worktree did nothing, because nothing in OpenCode
reads that file. The delegated edits were therefore produced by whatever model
that provider served, while every log line and the task docs said "local
DeepSeek".

This module emits the config OpenCode actually reads (`opencode.json`) for an
OpenAI-compatible provider whose `baseURL` is derived from `llm_env`. The model
id becomes `icf-local/<model>`, so the execution path contains no external
provider string. `routine_opencode.sh` writes it into the sandbox worktree
before launching.

    python scripts/ops/opencode_config.py --write ../ICF.worktrees/chore-x

prints the `provider/model` string to pass to `opencode run --model`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import llm_env  # noqa: E402

#: Provider id in the generated config. Deliberately not `deepseek` or `openai`:
#: either would make an external route look local in a log.
LOCAL_PROVIDER = "icf-local"
CONFIG_NAME = "opencode.json"


def build(model: str, base_url: str) -> dict:
    return {
        "$schema": "https://opencode.ai/config.json",
        "provider": {
            LOCAL_PROVIDER: {
                "npm": "@ai-sdk/openai-compatible",
                "name": "ICF local vLLM",
                "options": {"baseURL": base_url},
                "models": {model: {"name": model}},
            },
        },
        "model": provider_model(model),
    }


def provider_model(model: str) -> str:
    return f"{LOCAL_PROVIDER}/{model}"


def write(directory: str | Path, model: str | None = None,
          base_url: str | None = None) -> str:
    """Write `opencode.json` into `directory`; return the provider/model string."""
    model = model or llm_env.model()
    base_url = base_url or llm_env.base_url()
    path = Path(directory) / CONFIG_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build(model, base_url), ensure_ascii=False, indent=2)
                    + "\n", encoding="utf-8")
    return provider_model(model)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", metavar="DIR",
                    help="opencode.json 을 쓸 worktree 디렉터리")
    ap.add_argument("--model", default=None)
    ap.add_argument("--base-url", default=None)
    args = ap.parse_args(argv)
    if not args.write:
        ap.error("--write DIR 가 필요하다")
    print(write(args.write, args.model, args.base_url))
    return 0


if __name__ == "__main__":
    sys.exit(main())
