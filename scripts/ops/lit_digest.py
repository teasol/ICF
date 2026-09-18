#!/usr/bin/env python3
"""Turn a downloaded paper into a brief, using the local model rather than the
orchestrator's context.

The split is the point. Fetching needs internet, which only the orchestrator
has; reading needs a long context, which the local servers have three of and
which costs nothing. So the orchestrator spends tokens on a URL and a byte
count, and a 12,000-word paper is read here.

Three guards, because a summary that invents a number is worse than no summary:

  sourced     every claim must carry the paper's own section or figure label;
              claims that cannot be located are dropped by the model on request
              and flagged by the reviewer if they survive
  external    the brief states in its own header that nothing in it is our
              measurement. External results cannot satisfy our gates (D-047);
              they produce hypotheses only
  unverified  the file is marked as a local-model summary. The orchestrator has
              not read the paper, and the brief says so

    .venv/bin/python scripts/ops/lit_digest.py --slug icmil
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LIT = PROJECT_ROOT / "talks/lit"
RAW = LIT / "raw"
KST = timezone(timedelta(hours=9))
ENDPOINTS = [
    "http://127.0.0.1:8000/v1/chat/completions",
    "http://127.0.0.1:8001/v1/chat/completions",
    "http://127.0.0.1:8003/v1/chat/completions",
]
MAX_CHARS = 200000  # the servers carry 262144 tokens; this leaves room for the answer

PROMPT = """당신은 ICF 연구팀에 외부 문헌을 공급하는 역할이다. 아래 논문 전문을 읽고 브리프를 쓴다.

## 이 프로젝트의 맥락 (브리프를 우리 문제에 연결하기 위한 최소 정보)

- 목표: 병리 WSI 벤치마크에서 학습 기반 MIL 기준선(ABMIL)을 넘어서는 것.
- 현재 접근: fold context에서 closed-form ridge로 푸는 in-context 분류기. 최근까지
  "학습 파라미터 0개"를 정체성으로 삼았으나 그 제약은 폐기됐다.
- 현재 겪는 문제: 브랜치 7개의 유효 랭크가 3.52/7로 중복이 크다. 확률적 적합을 쓰면
  적합 분산이 승격 기준의 최대 2.8배로 커진다.

## 규칙 (어기면 브리프가 폐기된다)

- **본문에 있는 것만 쓴다.** 수치·데이터셋·기준선 이름은 본문에서 찾은 것만 적고, 어느
  절·표·그림에서 왔는지 함께 적는다. 찾지 못하면 "본문에서 확인 못함"이라고 쓴다.
- **이것은 외부 주장이지 우리의 관측이 아니다.** 우리 성능이나 우리 결론처럼 쓰지 마라.
- 우리 프로젝트와의 관련은 **가설로만** 적는다. 단정하지 마라.
- 한국어로 쓴다. 서론과 맺음말을 쓰지 마라.

## 출력 형식 (이 제목들을 그대로 쓴다)

## 1. 저자들이 주장하는 것
- 항목마다 한 줄, 끝에 (출처: 절/표/그림)

## 2. 방법의 핵심
- 구조, 학습 방식, 추론 방식. 재현에 필요한 구체적 설정이 본문에 있으면 적는다.

## 3. 평가 설정
- 데이터셋, 비교 기준선, 지표. ABMIL이 기준선에 있으면 명시한다. 없으면 없다고 쓴다.

## 4. 정량 결과
- 본문에 있는 수치만. 표 번호와 함께. 없으면 "본문에서 확인 못함".

## 5. 우리 프로젝트와의 관련 (가설)
- 우리가 옮겨올 수 있는 것 / 옮길 수 없는 것과 이유

## 6. 이 논문이 답하지 않는 것
- 우리 문제에 대해 이 논문으로는 알 수 없는 것

## 논문 전문

"""


def digest(slug: str, endpoint: str, timeout: int) -> str | None:
    raw = (RAW / f"{slug}.txt")
    if not raw.is_file():
        raise SystemExit(f"원문이 없다: {raw}")
    text = raw.read_text(encoding="utf-8")[:MAX_CHARS]
    body = json.dumps({
        "model": "Qwen3.8-27B", "temperature": 0.2, "max_tokens": 32000,
        "messages": [{"role": "user", "content": PROMPT + text}],
    }).encode("utf-8")
    req = urllib.request.Request(endpoint, data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.load(resp)
        return ((payload["choices"][0]["message"].get("content") or "").strip()) or None
    except Exception as exc:  # noqa: BLE001
        print(f"[{slug}] 호출 실패: {exc}")
        return None


def pick_endpoint() -> str:
    """Round-robin through a counter file rather than hashing the slug.

    Hashing looked like free distribution and was not: on 2026-09-18 three
    digests launched together and all three hashed to the same server, so two
    of the three seat servers sat idle while one queued the work. A counter
    cannot collide that way.
    """
    counter = PROJECT_ROOT / "talks/ops/.endpoint_rr"
    try:
        n = int(counter.read_text(encoding="utf-8").strip())
    except Exception:  # noqa: BLE001 - a missing or corrupt counter just restarts
        n = 0
    counter.parent.mkdir(parents=True, exist_ok=True)
    counter.write_text(str((n + 1) % 10_000), encoding="utf-8")
    return ENDPOINTS[n % len(ENDPOINTS)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slug", required=True)
    ap.add_argument("--endpoint", default=None, help="비우면 slug 해시로 분산")
    ap.add_argument("--timeout", type=int, default=1800)
    args = ap.parse_args()

    endpoint = args.endpoint or pick_endpoint()
    text = digest(args.slug, endpoint, args.timeout)
    if text is None:
        raise SystemExit(1)

    source = (RAW / f"{args.slug}.source")
    meta = source.read_text(encoding="utf-8").strip() if source.is_file() else "(출처 기록 없음)"
    stamp = datetime.now(KST).strftime("%Y-%m-%d %H:%M KST")
    out = LIT / f"{args.slug}_digest.md"
    out.write_text(
        f"# 문헌 브리프 — {args.slug} (로컬 모델 요약)\n\n"
        f"```\n{meta}\n```\n\n"
        f"- 요약: Qwen3.8-27B @ `{endpoint.split('//')[1].split('/')[0]}` · {stamp}\n"
        "- **오케스트레이터는 원문을 읽지 않았다.** 이 브리프는 로컬 모델의 요약이며 검증되지 않았다.\n"
        "- **외부 주장이지 우리의 관측이 아니다.** 어떤 수치도 우리 게이트·승격의 근거가 될 수 없다\n"
        "  (`D-047`). 여기서 나오는 것은 가설뿐이며, 우리 결론이 되려면 `Primary 7`에서 직접 측정한다.\n\n"
        "---\n\n" + text + "\n",
        encoding="utf-8")
    print(f"브리프: {out.relative_to(PROJECT_ROOT)} ({len(text)}자, {endpoint})")


if __name__ == "__main__":
    main()
