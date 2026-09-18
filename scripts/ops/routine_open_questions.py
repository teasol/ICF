#!/usr/bin/env python3
"""Keep a running list of what the council said it could not determine.

Every round writes '문서에 없음', '미확인', '판별 불가' in dozens of places. Those
lines are the project's real backlog, but they are buried in 100 KB reports and
get rediscovered instead of tracked -- the same failure that left seven GF runs
undocumented for two days.

This is a chore, not an experiment: it produces a checklist, makes no research
claim, and nothing downstream depends on anyone reading it promptly. The local
model is used only to group near-duplicate lines; the lines themselves are
extracted by code so nothing is invented.

    .venv/bin/python scripts/ops/routine_open_questions.py
"""

from __future__ import annotations

import json
import re
import urllib.request
from datetime import datetime, timedelta, timezone
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import llm_env  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COUNCIL = PROJECT_ROOT / "talks/council"
OUT = PROJECT_ROOT / "talks/ops/open_questions.md"
ENDPOINT = llm_env.endpoint()
KST = timezone(timedelta(hours=9))
MARKERS = ("문서에 없음", "미확인", "판별 불가", "실행 무효", "미측정", "미수행")
PATTERN = re.compile("|".join(map(re.escape, MARKERS)))


def harvest() -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for report in sorted(COUNCIL.glob("C-*/report.md")):
        lines = [
            ln.strip(" -•\t")
            for ln in report.read_text(encoding="utf-8").splitlines()
            if PATTERN.search(ln) and 30 < len(ln.strip()) < 400
        ]
        if lines:
            found[report.parent.name] = lines
    return found


def group(lines: list[str]) -> str | None:
    """Ask the local model to group near-duplicates. Extraction is code's job."""
    prompt = (
        "아래는 여러 회차 보고서에서 '미확인 / 판별 불가 / 문서에 없음'으로 표시된 문장들이다.\n"
        "비슷한 것끼리 묶어 8~15개의 항목으로 정리하라.\n"
        "규칙: 원문에 없는 내용을 더하지 마라. 해결책을 제안하지 마라. 각 항목은 한 줄로 쓰고 "
        "'- '로 시작하라. 서론과 맺음말을 쓰지 마라.\n\n" + "\n".join(f"- {l}" for l in lines)
    )
    body = json.dumps({"model": llm_env.model(), "temperature": 0.2, "max_tokens": 32000,
                       "messages": [{"role": "user", "content": prompt}]}).encode("utf-8")
    req = urllib.request.Request(ENDPOINT, data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=900) as resp:
            payload = json.load(resp)
        return ((payload["choices"][0]["message"].get("content") or "").strip()) or None
    except Exception:  # noqa: BLE001 - a chore must not fail the queue
        return None


def main() -> None:
    found = harvest()
    flat = [ln for lines in found.values() for ln in lines]
    print(f"회차 {len(found)}개에서 미확정 문장 {len(flat)}건 수집")
    grouped = group(flat[:400]) if flat else None
    stamp = datetime.now(KST).strftime("%Y-%m-%d %H:%M KST")
    body = [
        "# 협의체가 판별하지 못한 것 (자동 수집)", "",
        f"- 갱신: {stamp} · 회차 {len(found)}개 · 원문 {len(flat)}건",
        "- 수집은 코드가, 묶음은 로컬 모델이 한다. **연구 주장이 아니라 점검 목록이다.**", "",
    ]
    body += ["## 묶음", "", grouped or "(모델 호출 실패 — 아래 원문만 보라)", ""]
    body += ["## 회차별 원문", ""]
    for rid, lines in found.items():
        body += [f"### {rid} ({len(lines)}건)", ""] + [f"- {l}" for l in lines[:40]] + [""]
    OUT.write_text("\n".join(body), encoding="utf-8")
    print(f"기록: {OUT.relative_to(PROJECT_ROOT)} ({'묶음 성공' if grouped else '묶음 실패'})")


if __name__ == "__main__":
    main()
