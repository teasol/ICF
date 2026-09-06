#!/usr/bin/env python3
"""RU (Research Unit) card tool.

진행 단위를 커밋이 아니라 연구 단위로 되돌리기 위한 도구다.
착수 전 카드를 열고(`open`), 필수 필드를 채우고(`validate`), 종료 시
전수 총람에 append한다(`close`).

배경: 450커밋 총람이 측정한 바 Era 6은 RU당 커밋이 2.6개로 떨어졌고
(Era 3은 14.1개) 78개 중 17개가 단일 커밋이다. 자세한 근거는
docs/decisions.md D-015 참조.

사용법:
    python3 scripts/docs/ru.py open --title "형상 후보 조기 사망 조건 검사"
    python3 scripts/docs/ru.py list
    python3 scripts/docs/ru.py validate --id RU-79
    python3 scripts/docs/ru.py close --id RU-79

의존성 없음 (표준 라이브러리만 사용).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DOCS = REPO / "docs"
CARD_DIR = DOCS / "ru"
ALL_JSON = DOCS / "history" / "research_units_all.json"
ALL_MD = DOCS / "history" / "research_units_all.md"

# 착수 전에 반드시 채워야 하는 여섯 필드. agent_handoff.md §1.1과 일치해야 한다.
REQUIRED_OPEN = ("question", "hypothesis", "criteria", "budget", "kill", "next_actions")
# 종료 시 추가로 필요한 필드.
REQUIRED_CLOSE = ("experiment", "observations", "decision")

FIELD_HELP = {
    "question": "답하려는 연구 질문 하나",
    "hypothesis": "사전에 선언한 예상",
    "criteria": "판정 기준 (결과를 보기 전에 고정한다)",
    "budget": "GPU-h 또는 소요 상한",
    "kill": "중단 조건 — 어떤 관측이 나오면 즉시 종료하는가",
    "next_actions": "결과별 후속 행동 (pass / fail 각각)",
    "experiment": "실제로 수행한 실험과 변경",
    "observations": "관측 결과 (수치는 sign agreement 병기)",
    "decision": "내린 결정",
}


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO), *args],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def _load_units() -> list[dict]:
    if not ALL_JSON.exists():
        return []
    return json.loads(ALL_JSON.read_text(encoding="utf-8")).get("units", [])


def _ru_num(ru_id: str) -> int:
    m = re.fullmatch(r"RU-(\d+)", ru_id)
    return int(m.group(1)) if m else -1


def _next_id() -> str:
    nums = [_ru_num(u["id"]) for u in _load_units()]
    nums += [_ru_num(p.stem) for p in CARD_DIR.glob("RU-*.json")]
    return f"RU-{max(nums, default=0) + 1:02d}"


def _card_path(ru_id: str) -> Path:
    return CARD_DIR / f"{ru_id}.json"


def _missing(card: dict, fields: tuple[str, ...]) -> list[str]:
    return [f for f in fields if not str(card.get(f, "")).strip()]


def cmd_open(args: argparse.Namespace) -> int:
    CARD_DIR.mkdir(parents=True, exist_ok=True)
    ru_id = args.id or _next_id()
    path = _card_path(ru_id)
    if path.exists():
        print(f"[!] {ru_id} 카드가 이미 있다: {path}", file=sys.stderr)
        return 1

    now = datetime.now().astimezone()
    card = {
        "id": ru_id,
        "title": args.title,
        "type": args.type,
        "start_date": now.isoformat(timespec="seconds"),
        "start_sha": _git("rev-parse", "HEAD"),
        **{f: "" for f in REQUIRED_OPEN},
        "experiment": "",
        "observations": "",
        "decision": "",
        "evidence": [],
        "relations": "",
        "uncertainties": "",
    }
    path.write_text(json.dumps(card, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"[+] {ru_id} 카드를 만들었다: {path.relative_to(REPO)}\n")
    print("착수 전에 아래 여섯 필드를 채운다. 하나라도 비어 있으면 착수하지 않는다.")
    for f in REQUIRED_OPEN:
        print(f"  - {f:<13} {FIELD_HELP[f]}")
    print("\nnext_actions가 통과·실패 양쪽에서 같다면 그 작업은 하지 않는다.")
    print(f"\n채운 뒤:  python3 scripts/docs/ru.py validate --id {ru_id}")
    return 0


def cmd_list(_: argparse.Namespace) -> int:
    cards = sorted(CARD_DIR.glob("RU-*.json")) if CARD_DIR.exists() else []
    if not cards:
        print("진행 중 RU 없음.")
        return 0
    for p in cards:
        card = json.loads(p.read_text(encoding="utf-8"))
        miss = _missing(card, REQUIRED_OPEN)
        state = "착수 가능" if not miss else f"미완 {len(miss)}: {', '.join(miss)}"
        print(f"{card['id']}  {card.get('title','')}\n    {state}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    path = _card_path(args.id)
    if not path.exists():
        print(f"[!] {args.id} 카드가 없다.", file=sys.stderr)
        return 1
    card = json.loads(path.read_text(encoding="utf-8"))
    miss = _missing(card, REQUIRED_OPEN)
    if miss:
        print(f"[X] {args.id} 착수 불가 — 비어 있는 필수 필드:", file=sys.stderr)
        for f in miss:
            print(f"      {f:<13} {FIELD_HELP[f]}", file=sys.stderr)
        return 1
    print(f"[OK] {args.id} 착수 가능.")
    return 0


def _render_md(card: dict) -> str:
    ev = card.get("evidence") or []
    ev_md = "\n".join(f"  - {e}" for e in ev) if ev else "  - (없음)"
    rng = card.get("commit_indices") or "-"
    return f"""
### {card['id']}. {card.get('title','')}

- **일자 (Date)**: `{card.get('start_date','')[:10]}` ~ `{card.get('end_date','')[:10]}`
- **커밋 범위**: {rng} (`{card.get('start_sha','')[:8]}` ... `{card.get('end_sha','')[:8]}`)
- **작업 유형**: `{card.get('type','')}`
- **질문 (Question)**: {card.get('question','')}
- **가설 (Hypothesis)**: {card.get('hypothesis','')}
- **판정 기준 (Criteria, 사전 고정)**: {card.get('criteria','')}
- **예산 / 중단 조건**: {card.get('budget','')} / {card.get('kill','')}
- **실험 및 변경 (Experiment)**: {card.get('experiment','')}
- **관측 결과 (Observations)**: {card.get('observations','')}
- **결정 (Decision)**: {card.get('decision','')}
- **결과별 후속 행동**: {card.get('next_actions','')}
- **원문 근거 (Evidence)**:
{ev_md}
- **선행·후속 관계 (Relations)**: {card.get('relations','') or '-'}
- **확인 필요 사항 및 한계 (Uncertainties)**: {card.get('uncertainties','') or '-'}

---
"""


def cmd_close(args: argparse.Namespace) -> int:
    path = _card_path(args.id)
    if not path.exists():
        print(f"[!] {args.id} 카드가 없다.", file=sys.stderr)
        return 1
    card = json.loads(path.read_text(encoding="utf-8"))

    miss = _missing(card, REQUIRED_OPEN + REQUIRED_CLOSE)
    if miss:
        print(f"[X] {args.id} 종료 불가 — 비어 있는 필드:", file=sys.stderr)
        for f in miss:
            print(f"      {f:<13} {FIELD_HELP.get(f, '')}", file=sys.stderr)
        return 1

    now = datetime.now().astimezone()
    card["end_date"] = now.isoformat(timespec="seconds")
    card["end_sha"] = _git("rev-parse", "HEAD")
    card["date_range"] = f"{card['start_date'][:10]} ~ {card['end_date'][:10]}"
    rng = _git("rev-list", "--count", "--first-parent",
               f"{card['start_sha']}..{card['end_sha']}") if card.get("start_sha") else ""
    card["commit_count"] = int(rng) if rng.isdigit() else 0

    payload = json.loads(ALL_JSON.read_text(encoding="utf-8"))
    if any(u["id"] == card["id"] for u in payload["units"]):
        print(f"[!] {card['id']}가 총람에 이미 있다.", file=sys.stderr)
        return 1
    payload["units"].append(card)
    ALL_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")

    with ALL_MD.open("a", encoding="utf-8") as fh:
        fh.write(_render_md(card))

    path.unlink()
    print(f"[+] {card['id']} 종료. 총람에 append했다 (커밋 {card['commit_count']}개).")
    print("    규범·기준·구성이 바뀌었다면 docs/decisions.md에 결정 레코드를 추가한다.")
    print("    축을 닫거나 열었다면 docs/closed_axes.md에 반영한다.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="RU (Research Unit) card tool")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("open", help="착수 전 RU 카드 생성")
    p.add_argument("--title", required=True)
    p.add_argument("--id", default=None, help="지정하지 않으면 자동 부여")
    p.add_argument("--type", default="experiment")
    p.set_defaults(func=cmd_open)

    p = sub.add_parser("list", help="진행 중 RU 목록")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("validate", help="착수 가능 여부 검사")
    p.add_argument("--id", required=True)
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("close", help="종료하고 총람에 append")
    p.add_argument("--id", required=True)
    p.set_defaults(func=cmd_close)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
