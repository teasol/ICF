#!/usr/bin/env python3
"""Convene an ICF council round over local chat-completions servers (AGENTS.md, D-046).

Seats are *not* hardcoded here. Every round is driven by a round card written in
Phase 0 -- before any seat runs -- so that the roster, the data each seat sees,
the budget and the highest authority tier the round may reach are all on record
before results exist. `scripts/council.py plan` scaffolds one.

    scripts/council.py plan  --round-id C-20260917-1 --question "..." > card.json
    scripts/council.py check card.json          # rules + live endpoint probe
    scripts/council.py run   card.json

What this runner enforces that a plain prompt loop does not:

  Phase 1 isolation   seats never see one another's Phase 1 text
  mandatory refuter   a card without an R seat is rejected before any call
  abstention record   an empty body is recorded as an abstention and reported,
                      never silently dropped
  void rounds         if the P or S seat abstains the round is declared void
                      instead of being written up as a thin result
  budget              call count is capped by the card, checked before each phase
  preflight           every distinct endpoint answers a short arithmetic probe
                      first, because a misconfigured server returns HTTP 200
                      with fluent nonsense rather than failing

The output is a blackboard (every seat's raw text) plus a report. Both are
local-model artefacts and are not verified: figures quoted by a seat must be
checked against the canonical documents before they enter any conclusion.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.council.spec import (  # noqa: E402
    ARCHETYPES,
    RoundCard,
    Seat,
    SeatResult,
    classify_response,
    round_invalid_reason,
    validate_card,
)

KST = timezone(timedelta(hours=9))


def _shown(path: Path) -> str:
    """Repo-relative when possible; an out-dir outside the repo is still legal."""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _post(endpoint: str, body: dict[str, Any], timeout: int) -> dict[str, Any]:
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def ask(seat: Seat, prompt: str, max_tokens: int, timeout: int) -> SeatResult:
    """One seat, one turn. Transport failures become abstentions, not exceptions.

    A single seat failing must not sink the round -- but it must not vanish
    either, so every failure path produces a SeatResult carrying its reason.
    """
    body = {
        "model": seat.model,
        "messages": [
            {"role": "system", "content": seat.system_prompt()},
            {"role": "user", "content": prompt},
        ],
        "temperature": seat.temperature,
        "max_tokens": max_tokens,
    }
    try:
        payload = _post(seat.endpoint, body, timeout)
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        return SeatResult(
            seat=seat.name, archetype=seat.archetype, abstained=True,
            reason=f"요청 실패: {exc}",
        )
    return classify_response(seat, payload)


def run_phase(jobs: list[tuple[Seat, str]], max_tokens: int, timeout: int) -> list[SeatResult]:
    """Run every job concurrently. Submitted together so no seat sees another first."""
    results: list[SeatResult] = []
    if not jobs:
        return results
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        futures = {pool.submit(ask, seat, prompt, max_tokens, timeout): seat
                   for seat, prompt in jobs}
        for fut in concurrent.futures.as_completed(futures):
            res = fut.result()
            results.append(res)
            if res.ok():
                print(f"    [{res.seat}] {len(res.text)}자", flush=True)
            else:
                print(f"    [{res.seat}] 기권: {res.reason}", flush=True)
    # Deterministic order for the record, independent of completion order.
    order = {seat.name: i for i, (seat, _) in enumerate(jobs)}
    results.sort(key=lambda r: order[r.seat])
    return results


def preflight(card: RoundCard, timeout: int) -> list[str]:
    """Probe every distinct endpoint before spending the budget.

    A server loaded against a mismatched config answers 200 with fluent
    nonsense rather than crashing, so 'the port is open' is not a readiness
    check. One arithmetic question with a checkable answer is.
    """
    problems: list[str] = []
    seen: dict[str, str] = {}
    for seat in card.seats:
        if seat.endpoint in seen:
            continue
        seen[seat.endpoint] = seat.model
        body = {
            "model": seat.model,
            "messages": [{"role": "user", "content": "17 곱하기 23은? 숫자만 답하십시오."}],
            "temperature": 0,
            "max_tokens": 2000,
        }
        try:
            payload = _post(seat.endpoint, body, timeout)
        except Exception as exc:  # noqa: BLE001 - any failure is a preflight failure
            problems.append(f"{seat.endpoint}: 요청 실패 ({exc})")
            continue
        try:
            text = (payload["choices"][0]["message"].get("content") or "")
        except (KeyError, IndexError, TypeError):
            problems.append(f"{seat.endpoint}: 응답 형식이 예상과 다르다")
            continue
        if "391" not in text:
            problems.append(
                f"{seat.endpoint}: 확인 질의 오답 (받은 응답: {text.strip()[:60]!r}). "
                "설정 불일치로 정상 응답처럼 보이는 무의미한 출력일 수 있다"
            )
    return problems


def cmd_plan(args: argparse.Namespace) -> int:
    """Emit a round-card skeleton. The convener fills it in before running."""
    endpoints = [e.strip() for e in args.endpoints.split(",") if e.strip()]
    roster = [c.strip().upper() for c in args.seats.split(",") if c.strip()]
    seats = []
    for i, code in enumerate(roster):
        arch = ARCHETYPES.get(code)
        seats.append({
            "name": f"{code}1" if roster.count(code) == 1 else f"{code}{i + 1}",
            "archetype": code,
            "endpoint": endpoints[i % len(endpoints)] if endpoints else "",
            "model": args.model,
            "temperature": arch["default_temperature"] if arch else 0.3,
            "data_slice": [],
        })
    card = {
        "round_id": args.round_id,
        "question": args.question,
        "ru_type": args.ru_type,
        "prefixed_criteria": "",
        "composition_rationale": "",
        "max_authority": args.max_authority,
        "convener": args.convener,
        "state_documents": [d.strip() for d in args.state.split(",") if d.strip()],
        "max_tokens_per_seat": args.max_tokens,
        "max_total_calls": args.max_calls,
        "seats": seats,
        "note": "composition_rationale 과 (확증이면) prefixed_criteria 를 결과 전에 채우십시오.",
    }
    print(json.dumps(card, ensure_ascii=False, indent=2))
    return 0


def _load(path: str) -> RoundCard:
    raw = json.loads((PROJECT_ROOT / path).read_text(encoding="utf-8")
                     if not Path(path).is_absolute() else Path(path).read_text(encoding="utf-8"))
    return RoundCard.from_dict(raw)


def cmd_check(args: argparse.Namespace) -> int:
    card = _load(args.card)
    errors = validate_card(card)
    for e in errors:
        print(f"  [규칙 위반] {e}")
    if errors:
        print(f"\n{len(errors)}건의 위반으로 소집할 수 없습니다.")
        return 1
    print("  회차 카드 규칙 통과")
    if args.no_probe:
        return 0
    problems = preflight(card, args.timeout)
    for p in problems:
        print(f"  [엔드포인트] {p}")
    if problems:
        print(f"\n{len(problems)}건의 엔드포인트 문제로 소집할 수 없습니다.")
        return 1
    print("  엔드포인트 확인 질의 통과")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    card = _load(args.card)
    errors = validate_card(card)
    if errors:
        for e in errors:
            print(f"  [규칙 위반] {e}")
        print("\n회차 카드가 규범을 위반합니다. 소집하지 않았습니다.")
        return 1

    problems = preflight(card, args.timeout)
    if problems:
        for p in problems:
            print(f"  [엔드포인트] {p}")
        print("\n사전 확인 질의 실패. 소집하지 않았습니다.")
        return 1
    print("[Phase 0] 회차 카드 검증과 엔드포인트 확인 질의 통과", flush=True)

    out_dir = PROJECT_ROOT / args.out_dir / card.round_id
    out_dir.mkdir(parents=True, exist_ok=True)

    state_text = "\n\n".join(
        f"# 상태 문서 ({d})\n\n{(PROJECT_ROOT / d).read_text(encoding='utf-8')}"
        for d in card.state_documents
    )
    base = f"# 연구 질문\n{card.question}\n\n# RU 유형\n{card.ru_type}\n\n{state_text}\n"

    budget = {"used": 0, "cap": card.max_total_calls}

    def spend(n: int, phase: str) -> bool:
        if budget["used"] + n > budget["cap"]:
            print(f"  예산 상한 도달 ({budget['used']}/{budget['cap']}): {phase} 중단", flush=True)
            return False
        budget["used"] += n
        return True

    t0 = time.time()
    proposers = [s for s in card.seats if s.archetype == "P"]
    synths = [s for s in card.seats if s.archetype == "S"]
    reviewers = [s for s in card.seats if s.archetype not in ("P", "S")]

    # Phase 1 -- every seat but the synthesiser, in isolation. The synthesiser is
    # held back on purpose: it has nothing to synthesise yet, and giving it an
    # early opinion would anchor its own Phase 3 classification.
    phase1_seats = proposers + reviewers
    print(f"[Phase 1] 독립 병렬 — {len(phase1_seats)}좌석 (서로의 답을 보지 못함)", flush=True)
    if not spend(len(phase1_seats), "Phase 1"):
        return 1
    phase1 = run_phase(
        [(s, base + "\n위 상태를 당신의 좌석 임무에 따라 분석하십시오.") for s in phase1_seats],
        card.max_tokens_per_seat, args.timeout,
    )

    void = round_invalid_reason(phase1)
    proposal_texts = [r.text for r in phase1 if r.archetype == "P" and r.ok()]
    phase2: list[SeatResult] = []
    phase2b: list[SeatResult] = []

    if void:
        print(f"\n[회차 무효] {void}", flush=True)
    else:
        proposals = "\n\n".join(f"## 제안 ({r.seat})\n{r.text}" for r in phase1
                                if r.archetype == "P" and r.ok())
        print(f"\n[Phase 2] 교차 — {len(reviewers)}좌석이 제안을 공격", flush=True)
        if spend(len(reviewers), "Phase 2"):
            phase2 = run_phase(
                [(s, base + f"\n# 제안 좌석의 산출\n\n{proposals}\n\n"
                            "위 제안을 당신의 좌석 관점에서 공격하십시오. "
                            "당신의 Phase 1 분석을 반복하지 말고 이 제안에 대해서만 쓰십시오.")
                 for s in reviewers],
                card.max_tokens_per_seat, args.timeout,
            )

        challenges = "\n\n".join(f"## {r.seat}의 반론\n{r.text}" for r in phase2 if r.ok())
        if challenges and spend(len(proposers), "Phase 2b"):
            print(f"\n[Phase 2b] 제안 좌석이 반론에 일괄 응답", flush=True)
            phase2b = run_phase(
                [(s, base + f"\n# 당신의 제안\n{proposals}\n\n# 받은 반론\n{challenges}\n\n"
                            "각 반론에 대해 (1) 수용하고 수정할지 (2) 근거를 들어 반박할지 "
                            "(3) 판별 불가로 남길지 밝히십시오. 반론마다 한 항목씩 쓰십시오.")
                 for s in proposers],
                card.max_tokens_per_seat, args.timeout,
            )

    # The abstention ledger goes into the synthesis prompt verbatim: the S seat
    # is required to state which viewpoints are missing, and it can only do that
    # if it is told.
    labelled = ([("Phase 1", r) for r in phase1] + [("Phase 2", r) for r in phase2]
                + [("Phase 2b", r) for r in phase2b])
    abstentions = [(ph, r) for ph, r in labelled if not r.ok()]
    gap_note = (
        "\n\n# 기권한 좌석 (반드시 결론에 공백으로 명시할 것)\n"
        + "\n".join(f"- [{ph}] {r.seat} ({ARCHETYPES[r.archetype]['label']}): {r.reason}"
                    for ph, r in abstentions)
        if abstentions else "\n\n# 기권한 좌석\n- 없음"
    )

    transcript = "\n\n".join(
        [f"# Phase 1\n"] + [f"## {r.seat}\n{r.text}" for r in phase1 if r.ok()]
        + ([f"# Phase 2 반론"] + [f"## {r.seat}\n{r.text}" for r in phase2 if r.ok()] if phase2 else [])
        + ([f"# Phase 2b 제안 좌석 응답"] + [f"## {r.seat}\n{r.text}" for r in phase2b if r.ok()] if phase2b else [])
    )

    phase3: list[SeatResult] = []
    if spend(len(synths), "Phase 3"):
        print(f"\n[Phase 3] 종합", flush=True)
        phase3 = run_phase(
            [(s, base + "\n" + transcript + gap_note + "\n\n"
                        "위 논의의 증거를 지지 / 반박 / 판별 불가 / 실행 무효로 분류하고, "
                        "좌석 공백과 다음 행동을 지정된 형식으로 쓰십시오. "
                        "좌석 몇 개가 동의했는지를 근거로 쓰지 마십시오.")
             for s in synths],
            card.max_tokens_per_seat, args.timeout,
        )

    all_results = phase1 + phase2 + phase2b + phase3
    void = void or round_invalid_reason(all_results)
    elapsed = round(time.time() - t0, 1)

    blackboard = {
        "round_card": json.loads((PROJECT_ROOT / args.card).read_text(encoding="utf-8"))
        if not Path(args.card).is_absolute() else json.loads(Path(args.card).read_text(encoding="utf-8")),
        "convened": datetime.now(KST).strftime("%Y-%m-%d %H:%M KST"),
        "elapsed_seconds": elapsed,
        "calls_used": budget["used"],
        "calls_cap": budget["cap"],
        "void_reason": void,
        "abstentions": [{"phase": ph, "seat": r.seat, "archetype": r.archetype,
                         "reason": r.reason}
                        for ph, r in labelled + [("Phase 3", r) for r in phase3]
                        if not r.ok()],
        "phases": {
            name: [{"seat": r.seat, "archetype": r.archetype, "text": r.text,
                    "finish_reason": r.finish_reason, "abstained": r.abstained,
                    "reason": r.reason} for r in group]
            for name, group in (("phase1", phase1), ("phase2", phase2),
                                ("phase2b", phase2b), ("phase3", phase3))
        },
    }
    (out_dir / "blackboard.json").write_text(
        json.dumps(blackboard, ensure_ascii=False, indent=1), encoding="utf-8")

    lines = [
        f"# ICF Council — {card.round_id}", "",
        f"- **연구 질문**: {card.question}",
        f"- **RU 유형**: {card.ru_type}",
        f"- **최대 권한 등급**: {card.max_authority}",
        f"- **소집자**: {card.convener or '미기재'}",
        f"- **좌석**: " + ", ".join(
            f"{s.name}({s.archetype}, temp {s.temperature}, {s.endpoint})" for s in card.seats),
        f"- **호출**: {budget['used']}/{budget['cap']} · **소요**: {elapsed}초", "",
        "> 이 문서는 로컬 모델이 생성한 것이며 **검증되지 않았습니다.** 좌석이 인용한 수치는",
        "> 정본 문서와 대조하기 전에는 결론에 넣지 마십시오. 좌석 수와 동의 수는 증거가 아닙니다.", "",
    ]
    if void:
        lines += [f"## ⚠ 회차 무효", "", void, "",
                  "무효 회차의 산출물은 판정의 근거로 쓰지 않습니다. 아래는 기록용입니다.", ""]
    if blackboard["abstentions"]:
        lines += ["## 기권한 좌석", ""]
        lines += [f"- [{a['phase']}] `{a['seat']}` "
                  f"({ARCHETYPES[a['archetype']]['label']}): {a['reason']}"
                  for a in blackboard["abstentions"]]
        lines += [""]
    lines += ["## 편성 근거 (결과 전 기록)", "", card.composition_rationale or "(미기재)", ""]
    if card.prefixed_criteria.strip():
        lines += ["## 사전 고정 기준 (결과 전 기록)", "", card.prefixed_criteria, ""]
    for title, group in (("Phase 1 — 독립", phase1), ("Phase 2 — 교차", phase2),
                         ("Phase 2b — 제안 좌석 응답", phase2b), ("Phase 3 — 종합", phase3)):
        ok = [r for r in group if r.ok()]
        if not ok:
            continue
        lines += ["---", "", f"## {title}", ""]
        for r in ok:
            lines += [f"### {r.seat} ({ARCHETYPES[r.archetype]['label']})", "", r.text, ""]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"\n{'무효 종료' if void else '완료'} ({elapsed}초, 호출 {budget['used']}/{budget['cap']})")
    print(f"  blackboard: {_shown(out_dir / 'blackboard.json')}")
    print(f"  report    : {_shown(out_dir / 'report.md')}")
    return 2 if void else 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan", help="회차 카드 뼈대를 출력한다")
    p.add_argument("--round-id", required=True, help="C-YYYYMMDD-n")
    p.add_argument("--question", required=True)
    p.add_argument("--ru-type", default="탐색", help="탐색 | 진단 | 확증 | 재현")
    p.add_argument("--seats", default="P,R,A,S", help="좌석 유형 목록 (쉼표 구분)")
    p.add_argument("--endpoints", default="", help="chat-completions URL 목록 (쉼표 구분)")
    p.add_argument("--model", default="", help="좌석이 호출할 모델 이름")
    p.add_argument("--state", default="docs/current_status.md")
    p.add_argument("--max-authority", default="T1")
    p.add_argument("--max-tokens", type=int, default=32000)
    p.add_argument("--max-calls", type=int, default=32)
    p.add_argument("--convener", default="")
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("check", help="회차 카드 규칙과 엔드포인트를 확인한다")
    p.add_argument("card")
    p.add_argument("--timeout", type=int, default=300)
    p.add_argument("--no-probe", action="store_true", help="엔드포인트 확인 질의를 건너뛴다")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("run", help="회차를 소집한다")
    p.add_argument("card")
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--out-dir", default="talks/council")
    p.set_defaults(func=cmd_run)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
