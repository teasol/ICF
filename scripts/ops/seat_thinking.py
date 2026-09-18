#!/usr/bin/env python3
"""Let standing seats keep thinking between rounds, without letting them agree
with themselves.

The council protocol is strictly turn-taking: a seat that is not in the current
phase produces no tokens at all, and for a language model producing no tokens is
not thinking. Between rounds every seat is simply absent. This gives three of
them a private notebook that is fed back on the next tick, so a line of
reasoning can develop across rounds instead of restarting each time.

The obvious failure is that a model fed only its own prior text becomes more
confident without new evidence -- the cross-round anchoring measured on
2026-09-18, but now inside a single seat and running unattended. Four guards:

  private        one seat never reads another's notebook, so notebooks cannot
                 manufacture agreement between seats
  grounded       every tick also carries the canonical facts digest, so the
                 input is never the seat's own text alone
  tagged         each line must be 관측 / 추론 / 질문; only 관측 may carry a
                 number, and only with a source
  self-revising  each entry must either propose a check or revise an earlier
                 entry. Restating a previous position is explicitly worthless.

Notebook content is never evidence. It becomes evidence only when a 질문 line
is harvested, verified by the orchestrator, and recorded. That harvest is the
point of the whole thing -- see talks/ops/seat_questions.md.

    .venv/bin/python scripts/ops/seat_thinking.py --seat R
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import urllib.request
from datetime import datetime, timedelta, timezone
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import llm_env  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS = PROJECT_ROOT / "talks/ops/notebooks"
QUESTIONS = PROJECT_ROOT / "talks/ops/seat_questions.md"
KST = timezone(timedelta(hours=9))

#: Only the seats whose job is to doubt. A standing proposer notebook would be
#: a machine for generating unfalsifiable ideas nobody asked for.
SEATS = {
    "R": ("반증", llm_env.endpoint(0), 0.5,
          "이 프로젝트의 현재 주장 중 무엇이 틀릴 수 있는지, 더 단순한 설명이 무엇인지 본다."),
    "A": ("감사", llm_env.endpoint(1), 0.2,
          "데이터·평가·계측이 새는 지점을 본다. 기록과 실행이 어긋나는 경로를 찾는다."),
    "T": ("이론", llm_env.endpoint(2), 0.4,
          "현재 접근이 암묵적으로 참이라고 두는 가정을 드러낸다."),
}
TAG_RE = re.compile(r"^-\s*(관측|추론|질문):")
MAX_NOTEBOOK_CHARS = 24000
TAIL_CHARS = 8000


def facts_digest() -> str:
    """Canonical, code-extracted context. Never model-generated."""
    def sh(cmd: str) -> str:
        try:
            return subprocess.run(cmd, shell=True, cwd=PROJECT_ROOT, capture_output=True,
                                  text=True, timeout=30).stdout.strip()
        except Exception:  # noqa: BLE001
            return ""
    status = (PROJECT_ROOT / "docs/current_status.md")
    open_q = (PROJECT_ROOT / "talks/ops/open_questions.md")
    parts = [
        "## 최근 커밋 12건", sh("git log --oneline -12"), "",
        "## 현재 상태 문서", status.read_text(encoding="utf-8") if status.is_file() else "(없음)",
    ]
    if open_q.is_file():
        head = open_q.read_text(encoding="utf-8").split("## 회차별 원문")[0]
        parts += ["", "## 협의체가 판별하지 못한 것 (자동 수집 묶음)", head]
    return "\n".join(parts)


def ask(seat: str, prompt: str, timeout: int) -> str | None:
    label, endpoint, temp, _ = SEATS[seat]
    body = json.dumps({
        "model": llm_env.model(), "temperature": temp, "max_tokens": 32000,
        "messages": [{"role": "user", "content": prompt}],
    }).encode("utf-8")
    req = urllib.request.Request(endpoint, data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.load(resp)
        return ((payload["choices"][0]["message"].get("content") or "").strip()) or None
    except Exception:  # noqa: BLE001 - a thinking tick must never break the queue
        return None


def build_prompt(seat: str, notebook_tail: str) -> str:
    label, _, _, mandate = SEATS[seat]
    return f"""당신은 ICF 연구 협의체의 상설 {label}({seat}) 좌석이다. {mandate}

지금은 회차 중이 아니다. 이 노트는 **당신만 읽는다.** 다른 좌석은 볼 수 없다.

## 규칙 (반드시 지켜라)

- 모든 줄은 `- 관측:` `- 추론:` `- 질문:` 중 하나로 시작한다. 다른 형식은 버려진다.
- **`관측`에만 수치를 쓸 수 있고, 반드시 출처(파일·문서 절)를 함께 적는다.** 아래 자료에
  없는 수치는 어떤 태그로도 쓰지 마라.
- **이전 노트의 입장을 되풀이하는 것은 가치가 없다.** 각 항목은 (a) 확인 가능한 검사를
  제안하거나 (b) 이전 항목을 수정·철회해야 한다. 둘 다 아니면 쓰지 마라.
- `질문`은 **무엇을 확인하면 답이 갈리는지**가 분명해야 한다. 막연한 의문은 쓰지 마라.
- 6줄을 넘기지 마라. 서론과 맺음말을 쓰지 마라.

## 정본 자료 (코드가 추출한 것)

{facts_digest()}

## 당신의 이전 노트 (최근 부분)

{notebook_tail or "(아직 없음)"}

## 지금 할 일

위 정본 자료에서 **새로 바뀐 것**에 반응하라. 바뀐 것이 없으면 이전 항목 중 하나를
스스로 공격하라. 6줄 이내로 이어 쓰라.
"""


def harvest_questions() -> None:
    """Pull every 질문 line out of every notebook for the orchestrator to triage."""
    lines = []
    for nb in sorted(NOTEBOOKS.glob("*.md")):
        seat = nb.stem
        for ln in nb.read_text(encoding="utf-8").splitlines():
            if ln.strip().startswith("- 질문:"):
                lines.append(f"- `{seat}` {ln.strip()[len('- 질문:'):].strip()}")
    stamp = datetime.now(KST).strftime("%Y-%m-%d %H:%M KST")
    QUESTIONS.write_text(
        "# 상설 좌석이 낸 검증 가능한 질문 (자동 수집)\n\n"
        f"- 갱신: {stamp} · {len(lines)}건\n"
        "- **이 목록은 증거가 아니다.** 오케스트레이터가 확인해야 비로소 근거가 된다.\n\n"
        + ("\n".join(lines) if lines else "(없음)") + "\n",
        encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seat", choices=sorted(SEATS), help="비우면 세 좌석 모두 한 번씩")
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args()

    NOTEBOOKS.mkdir(parents=True, exist_ok=True)
    for seat in ([args.seat] if args.seat else sorted(SEATS)):
        nb = NOTEBOOKS / f"{seat}.md"
        body = nb.read_text(encoding="utf-8") if nb.is_file() else ""
        text = ask(seat, build_prompt(seat, body[-TAIL_CHARS:]), args.timeout)
        if text is None:
            print(f"[{seat}] 호출 실패 — 건너뜀")
            continue
        kept = [ln for ln in text.splitlines() if TAG_RE.match(ln.strip())][:6]
        if not kept:
            print(f"[{seat}] 형식을 지킨 줄이 없어 버림")
            continue
        stamp = datetime.now(KST).strftime("%Y-%m-%d %H:%M")
        body += f"\n### {stamp}\n\n" + "\n".join(kept) + "\n"
        # Oldest entries are dropped rather than summarised: a model-written
        # compaction would quietly become the seat's only memory of what it saw.
        if len(body) > MAX_NOTEBOOK_CHARS:
            body = "> (앞부분은 길이 제한으로 잘렸다)\n" + body[-MAX_NOTEBOOK_CHARS:]
        nb.write_text(body, encoding="utf-8")
        print(f"[{seat}] {len(kept)}줄 추가")
    harvest_questions()
    print(f"질문 수집: {QUESTIONS.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
