"""Seat archetypes and round-card validation for the ICF council (AGENTS.md, D-046).

Everything in this module is pure: no network, no clock, no filesystem writes.
That is deliberate -- the composition rules are the load-bearing safety device
of the whole scheme, so they have to be testable without standing up a server.

Why each rule exists (all of them come from AGENTS.md SS1.2 and SS5):

- A round without an `R` seat is void. With no seat carrying an explicit duty to
  refute, the remaining seats only reinforce one another, and several seats
  backed by the same model reinforce fast.
- The `S` seat may not be the `P` seat. A proposer synthesising its own proposal
  produces a summary, not a judgement.
- Composition rationale is required *before* results. The main hazard of
  per-question seat composition is that the convener's bias lands directly in
  the roster; a rationale written after the fact cannot expose that.
- The per-seat token floor is 32000. Reasoning models spend most of their budget
  on reasoning tokens; below that floor seats return empty bodies, and an empty
  body is easy to mistake for a seat that had nothing to add.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# Anti-consensus preamble. This is the load-bearing part of every seat prompt:
# when one model sits in several seats, agreement is the default failure mode
# rather than a signal, so the instruction not to seek it has to come first.
COMMON_PREAMBLE = """당신은 ICF 연구 협의체의 한 좌석입니다. 배정된 관점의 임무만 수행합니다.

다른 좌석과 의견이 일치하는 것 자체에는 아무런 가치가 없습니다.
합의를 추구하지 마십시오.
다른 좌석의 주장이 옳다면, 그것을 옳게 만드는 구체적 증거나 추론을 특정해서 밝히십시오.
근거가 없다면 몇 좌석이 동의하든 상관없이 근거 없음을 명시하십시오.

연구 규범 (반드시 준수):
- 주어진 문서에 없는 수치를 지어내지 마십시오. 모르면 '문서에 없음'이라고 쓰십시오.
- 성능 관측을 기전의 입증으로 바꾸어 쓰지 마십시오. 불명확하면 '기전 미확인'이라고 쓰십시오.
- 결론은 증거의 범위를 넘지 마십시오. 실패 / 반박 / 판별 불가 / 실행 무효를 구분하십시오.
- 보류 자료(SEAL 10)의 내용을 추측하거나 요구하지 마십시오.
- 한국어로, 지정된 출력 형식만 사용하십시오. 서론과 맺음말을 쓰지 마십시오."""

# Seat archetypes (AGENTS.md SS1.1). `default_temperature` is a starting point the
# round card may override; the recorded card always wins.
ARCHETYPES: dict[str, dict[str, Any]] = {
    "P": {
        "label": "제안",
        "default_temperature": 0.9,
        "mandate": """당신의 좌석: **P (제안)**
핵심 질문: "무엇을 어떻게 바꾸면 나아지는가?"

제안은 **최대 3개**. 수를 늘리지 말고 각각을 깊게 쓰십시오.

## 제안 N: <이름>
- **바꾸는 것**: 무엇을 변경하는가
- **바꾸는 전제**: 지금 참이라고 두고 있는 것 중 무엇을 뒤집는가
- **기대하는 차이**: 어느 지표가 어떻게 (근거 없으면 '정량 예측 불가')
- **기전**: 어떤 경로로 그렇게 되는가 (추측이면 '기전 미확인')
- **실패 조건**: 이 제안이 틀렸다면 가장 그럴듯한 이유""",
    },
    "R": {
        "label": "반증",
        "default_temperature": 0.5,
        "mandate": """당신의 좌석: **R (반증)**
핵심 질문: "이 주장이 틀렸다면 어디서 틀리는가? 더 단순한 설명은 없는가?"

제안을 방어하지 마십시오. 당신의 임무는 공격입니다.
**더 단순한 설명이 같은 관측을 낳는지** 반드시 확인하십시오.

## 반론 N
- **공격 대상**: 어느 주장인가
- **더 단순한 설명**: 같은 관측을 낳는 덜 복잡한 설명 (없으면 '없음')
- **무엇이 참이면 이 주장이 무너지는가**
- **이 반론의 한계**: 당신의 반론이 틀릴 조건""",
    },
    "A": {
        "label": "감사",
        "default_temperature": 0.2,
        "mandate": """당신의 좌석: **A (감사)**
핵심 질문: "관측된 차이가 데이터·평가 설계에서 새어 나온 것은 아닌가?"

제안의 매력에 휘둘리지 말고 **오직 데이터와 평가 절차만** 보십시오.
분할 단위, 중복, 전처리·정규화·통계량의 fit 범위, 라벨 사용, 평가 독립성,
매니페스트 고정, 임계값 선택, 기준선이 정답을 보고 있는지를 점검합니다.

## 점검 N: <항목>
- **확인하려는 것**
- **문서에서 확인되는가**: 확인됨 / 문서에 없음 / 모순됨
- **샌다면 나타날 증상**
- **확인 방법**: 어떤 파일·명령으로 확인하는가""",
    },
    "G": {
        "label": "일반화",
        "default_temperature": 0.3,
        "mandate": """당신의 좌석: **G (일반화)**
핵심 질문: "이 결과가 지금 본 분포 밖에서도 유지되는가?"

보류 자료의 내용을 추측하지 마십시오. 당신은 **무엇이 미확인인지**를 밝힙니다.

## 일반화 위험 N
- **위험**: 무엇이 분포 밖에서 깨지는가
- **현재 증거가 말할 수 있는 범위**
- **현재 증거가 말할 수 없는 것**
- **확인하려면 무엇이 필요한가** (현재 자료로 불가능하면 '현재 자료로 불가')""",
    },
    "T": {
        "label": "이론",
        "default_temperature": 0.4,
        "mandate": """당신의 좌석: **T (이론)**
핵심 질문: "이 접근이 암묵적으로 무엇을 참이라고 두고 있는가?"

수학적으로 그럴듯한 설명을 지어내지 마십시오.
당신의 임무는 **숨은 가정을 드러내는 것**이지 정당화가 아닙니다.

## 이론적 병목 N
- **가정**: 현재 접근이 암묵적으로 참이라고 두는 것
- **이 가정이 깨지는 조건**
- **깨질 때 나타나는 관측 가능한 증상**
- **문서의 어떤 관측이 이 가정과 충돌하는가** (없으면 '해당 관측 없음')""",
    },
    "M": {
        "label": "실측",
        "default_temperature": 0.3,
        "mandate": """당신의 좌석: **M (실측)**
핵심 질문: "실제로 구현 가능한가, 얼마가 드는가, 무엇이 막는가?"

추정을 실측으로 보고하지 마십시오. 확인하지 않은 것은 '미실측'이라고 쓰십시오.

## 실측 N: <대상>
- **건드려야 하는 코드·설정**: 구체적 경로
- **비용**: 시간·메모리·연산 (미확인이면 '미실측')
- **막는 것**: 무엇이 왜 막는가
- **중단 기준**: 어느 지점에서 그만두어야 하는가""",
    },
    "S": {
        "label": "종합",
        "default_temperature": 0.2,
        "mandate": """당신의 좌석: **S (종합)**
핵심 질문: "증거가 무엇을 지지하고 무엇을 반박하며 무엇을 가르지 못하는가?"

네 칸을 반드시 분리하고, 합치거나 비워 두지 마십시오.
**기권한 좌석이 있으면 그 공백을 결론에 명시**하십시오.
좌석 몇 개가 동의했는지를 근거로 쓰지 마십시오. 표결은 증거가 아닙니다.

## 지지 (증거가 뒷받침함)
## 반박 (증거가 반박함)
## 판별 불가 (증거가 가르지 못함)
## 실행 무효 (실행·계측 결함으로 판단 불가)
## 좌석 공백 (기권·실패로 빠진 관점)
## 다음 행동 (권한 등급과 함께)
## 하지 말 것 (이유와 함께)""",
    },
}

MIN_TOKENS_PER_SEAT = 32000
RU_TYPES = ("탐색", "진단", "확증", "재현")
AUTHORITY_TIERS = ("T0", "T1", "T2")
ROUND_ID_RE = re.compile(r"^C-\d{8}-\d+$")

# Paths whose contents no seat may receive (AGENTS.md SS5). Substring match, so a
# nested or renamed path under the same name is still caught.
WITHHELD_MARKERS = ("seal10", "seal_10", "seal-10")

#: A prior round's report may not be a seat input. Feeding one in made round 11
#: reproduce round 10's top recommendation almost verbatim, despite the prompt
#: telling the seats to attack it. Phase 1 isolation only prevents anchoring
#: *within* a round; across rounds the convener's choice of inputs decides it,
#: so the rule has to be enforced where cards are validated rather than left in
#: prose that will be forgotten.
PRIOR_REPORT_RE = re.compile(r"talks/council/C-[\d-]+/(report|blackboard)\.")


@dataclass
class Seat:
    """One seat in one round. Transient by construction -- seats do not persist."""

    name: str
    archetype: str
    endpoint: str
    model: str
    temperature: float
    data_slice: list[str] = field(default_factory=list)

    def system_prompt(self) -> str:
        return COMMON_PREAMBLE + "\n\n" + ARCHETYPES[self.archetype]["mandate"]


@dataclass
class RoundCard:
    """Phase 0 record. Written before any seat runs and never rewritten in place."""

    round_id: str
    question: str
    ru_type: str
    composition_rationale: str
    seats: list[Seat]
    state_documents: list[str]
    max_authority: str = "T1"
    prefixed_criteria: str = ""
    max_tokens_per_seat: int = MIN_TOKENS_PER_SEAT
    max_total_calls: int = 32
    convener: str = ""
    note: str = ""

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> "RoundCard":
        seats = [
            Seat(
                name=s["name"],
                archetype=s["archetype"],
                endpoint=s["endpoint"],
                model=s.get("model", ""),
                temperature=float(
                    s.get("temperature", ARCHETYPES.get(s["archetype"], {}).get("default_temperature", 0.3))
                ),
                data_slice=list(s.get("data_slice", [])),
            )
            for s in raw.get("seats", [])
        ]
        known = {
            k: v for k, v in raw.items()
            if k in RoundCard.__dataclass_fields__ and k != "seats"
        }
        return RoundCard(seats=seats, **known)

    def archetypes(self) -> list[str]:
        return [s.archetype for s in self.seats]


def validate_card(card: RoundCard) -> list[str]:
    """Return every rule violation. Empty list means the round may be convened.

    Returns all errors rather than raising on the first: a convener fixing a
    card wants the whole list, not one error per run.
    """
    errors: list[str] = []

    if not ROUND_ID_RE.match(card.round_id):
        errors.append(f"round_id '{card.round_id}' 형식 위반 (C-YYYYMMDD-n)")
    if not card.question.strip():
        errors.append("question 이 비어 있다")
    if card.ru_type not in RU_TYPES:
        errors.append(f"ru_type '{card.ru_type}' 은 {RU_TYPES} 중 하나여야 한다")
    if card.max_authority not in AUTHORITY_TIERS:
        errors.append(f"max_authority '{card.max_authority}' 은 {AUTHORITY_TIERS} 중 하나여야 한다")

    # Confirmatory rounds must fix their criteria before results exist; that is
    # the only thing separating confirmation from exploration told after the fact.
    if card.ru_type == "확증" and not card.prefixed_criteria.strip():
        errors.append("ru_type 이 '확증' 이면 prefixed_criteria 를 결과 전에 고정해야 한다")

    if not card.composition_rationale.strip():
        errors.append("composition_rationale 이 비어 있다 (동적 편성의 편향 통제 장치)")

    unknown = [s.archetype for s in card.seats if s.archetype not in ARCHETYPES]
    if unknown:
        errors.append(f"정의되지 않은 좌석 유형: {sorted(set(unknown))}")

    names = [s.name for s in card.seats]
    if len(names) != len(set(names)):
        errors.append("좌석 이름이 중복된다")

    kinds = card.archetypes()
    for required, why in (
        ("P", "제안 좌석이 없으면 검토할 대상이 없다"),
        ("R", "반증 좌석이 없는 회차는 무효다 (AGENTS.md SS1.2)"),
        ("S", "종합 좌석이 없으면 증거가 분류되지 않는다"),
    ):
        if required not in kinds:
            errors.append(f"필수 좌석 '{required}' 누락 — {why}")

    if len(card.seats) < 3:
        errors.append("최소 구성은 P + R + S 3좌석이다")

    # The synthesis seat must not be the proposer. Same name, or a single seat
    # carrying both duties, collapses proposal and judgement into one context.
    p_names = {s.name for s in card.seats if s.archetype == "P"}
    s_names = {s.name for s in card.seats if s.archetype == "S"}
    shared = p_names & s_names
    if shared:
        errors.append(f"종합 좌석이 제안 좌석과 같다: {sorted(shared)}")

    if card.max_tokens_per_seat < MIN_TOKENS_PER_SEAT:
        errors.append(
            f"max_tokens_per_seat {card.max_tokens_per_seat} < {MIN_TOKENS_PER_SEAT}: "
            "추론 토큰이 예산을 소진해 좌석이 빈 응답을 반환한다"
        )

    # Phase 1 (every seat) + Phase 2 (every non-P, non-S seat) + Phase 2b + Phase 3
    reviewers = [s for s in card.seats if s.archetype not in ("P", "S")]
    needed = len(card.seats) + len(reviewers) + 1 + 1
    if card.max_total_calls < needed:
        errors.append(
            f"max_total_calls {card.max_total_calls} 가 이 편성의 최소 호출 수 {needed} 보다 작다"
        )

    for seat in card.seats:
        if not 0.0 <= seat.temperature <= 2.0:
            errors.append(f"좌석 '{seat.name}' 의 temperature {seat.temperature} 범위 밖")
        if not seat.endpoint:
            errors.append(f"좌석 '{seat.name}' 에 endpoint 가 없다")

    errors.extend(_withheld_violations(card))
    return errors


def _withheld_violations(card: RoundCard) -> list[str]:
    """Reject withheld hold-out paths and prior council reports."""
    errors: list[str] = []
    candidates = list(card.state_documents) + [p for s in card.seats for p in s.data_slice]
    for path in candidates:
        if PRIOR_REPORT_RE.search(path.replace("\\", "/")):
            errors.append(
                f"앞선 회차의 보고서를 좌석 입력에 넣을 수 없다: '{path}' "
                "(회차 간 앵커링 — 쟁점은 소집자가 자기 말로 다시 쓴다)")
            continue
        lowered = path.lower().replace(" ", "")
        for marker in WITHHELD_MARKERS:
            if marker in lowered:
                errors.append(f"보류 자료로 보이는 경로가 좌석 입력에 있다: '{path}'")
                break
    return errors


@dataclass
class SeatResult:
    """One seat's response, including the abstention case.

    `abstained` is not an error state to be swallowed. A seat that returned
    nothing is a missing viewpoint, and the synthesis seat is told about it.
    """

    seat: str
    archetype: str
    text: str = ""
    finish_reason: str = ""
    abstained: bool = False
    reason: str = ""

    def ok(self) -> bool:
        return not self.abstained and bool(self.text.strip())


def classify_response(seat: Seat, payload: dict[str, Any]) -> SeatResult:
    """Turn a chat-completions payload into a result, recording abstentions.

    A body-less response is the documented failure of reasoning models under a
    tight token budget: the server answers 200, the reasoning trace consumes the
    budget, and `content` comes back empty. That is an abstention, not an answer.
    """
    try:
        choice = payload["choices"][0]
    except (KeyError, IndexError, TypeError):
        return SeatResult(
            seat=seat.name, archetype=seat.archetype, abstained=True,
            reason="응답에 choices 가 없다 (서버 오류 또는 설정 불일치)",
        )
    finish = str(choice.get("finish_reason", ""))
    text = (choice.get("message") or {}).get("content") or ""
    if not text.strip():
        return SeatResult(
            seat=seat.name, archetype=seat.archetype, finish_reason=finish, abstained=True,
            reason=f"본문이 비어 있다 (finish_reason={finish or '미확인'}); "
                   "추론 토큰이 예산을 소진했을 가능성이 높다",
        )
    return SeatResult(
        seat=seat.name, archetype=seat.archetype, text=text.strip(), finish_reason=finish,
    )


def round_invalid_reason(results: list[SeatResult]) -> str:
    """Return why the round is void, or '' if it stands (AGENTS.md SS5).

    A failed proposer leaves nothing to cross-examine; a failed synthesiser
    leaves the evidence unclassified. Either way the later phases are noise, so
    the round is declared void instead of being reported as a thin result.
    """
    for archetype, why in (
        ("P", "제안(P) 좌석이 기권했다 — 교차 검증할 대상이 없다"),
        ("S", "종합(S) 좌석이 기권했다 — 증거가 분류되지 않았다"),
    ):
        present = [r for r in results if r.archetype == archetype]
        if present and not any(r.ok() for r in present):
            return why
    return ""
