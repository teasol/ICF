"""The council's composition rules are the safety device, so they are tested.

These rules only help if they fire before a round is convened -- once seats have
run, a missing refuter cannot be added retroactively. Everything here is pure,
so the whole file runs without a server.
"""

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.council.spec import (  # noqa: E402
    MIN_TOKENS_PER_SEAT,
    RoundCard,
    Seat,
    SeatResult,
    classify_response,
    round_invalid_reason,
    validate_card,
)


def _seat(name, archetype, temp=0.3):
    return Seat(name=name, archetype=archetype, endpoint="http://127.0.0.1:8000/v1/chat/completions",
                model="test-model", temperature=temp)


def _card(**over):
    base = dict(
        round_id="C-20260917-1",
        question="테스트 질문",
        ru_type="탐색",
        composition_rationale="반증과 감사 관점이 필요하다고 판단했다.",
        seats=[_seat("P1", "P"), _seat("R1", "R"), _seat("A1", "A"), _seat("S1", "S")],
        state_documents=["docs/current_status.md"],
        max_tokens_per_seat=MIN_TOKENS_PER_SEAT,
        max_total_calls=32,
    )
    base.update(over)
    return RoundCard(**base)


class TestComposition(unittest.TestCase):
    def test_valid_card_passes(self):
        self.assertEqual(validate_card(_card()), [])

    def test_round_without_refuter_is_rejected(self):
        card = _card(seats=[_seat("P1", "P"), _seat("A1", "A"), _seat("S1", "S")])
        errors = validate_card(card)
        self.assertTrue(any("'R'" in e for e in errors), errors)

    def test_synthesis_seat_may_not_be_the_proposer(self):
        # Same seat name carrying both duties collapses proposal and judgement.
        shared = _seat("X1", "P")
        card = _card(seats=[shared, _seat("R1", "R"), Seat(
            name="X1", archetype="S", endpoint=shared.endpoint, model="m", temperature=0.2)])
        errors = validate_card(card)
        self.assertTrue(any("종합 좌석이 제안 좌석과 같다" in e for e in errors), errors)

    def test_fewer_than_three_seats_is_rejected(self):
        card = _card(seats=[_seat("P1", "P"), _seat("R1", "R")])
        errors = validate_card(card)
        self.assertTrue(any("최소 구성" in e for e in errors), errors)

    def test_missing_composition_rationale_is_rejected(self):
        errors = validate_card(_card(composition_rationale="   "))
        self.assertTrue(any("composition_rationale" in e for e in errors), errors)

    def test_confirmatory_round_requires_prefixed_criteria(self):
        errors = validate_card(_card(ru_type="확증"))
        self.assertTrue(any("prefixed_criteria" in e for e in errors), errors)
        self.assertEqual(validate_card(_card(ru_type="확증", prefixed_criteria="A>B 면 채택")), [])

    def test_token_floor_is_enforced(self):
        errors = validate_card(_card(max_tokens_per_seat=6000))
        self.assertTrue(any("빈 응답" in e for e in errors), errors)

    def test_budget_must_cover_the_roster(self):
        errors = validate_card(_card(max_total_calls=3))
        self.assertTrue(any("max_total_calls" in e for e in errors), errors)

    def test_withheld_data_path_is_rejected(self):
        errors = validate_card(_card(state_documents=["docs/seal10_results.md"]))
        self.assertTrue(any("보류 자료" in e for e in errors), errors)

    def test_unknown_archetype_is_rejected(self):
        card = _card(seats=[_seat("P1", "P"), _seat("R1", "R"), _seat("S1", "S"), _seat("Z1", "Z")])
        errors = validate_card(card)
        self.assertTrue(any("정의되지 않은 좌석" in e for e in errors), errors)


class TestAbstention(unittest.TestCase):
    def test_empty_body_is_an_abstention_not_an_answer(self):
        seat = _seat("R1", "R")
        res = classify_response(seat, {"choices": [{"finish_reason": "length",
                                                    "message": {"content": ""}}]})
        self.assertTrue(res.abstained)
        self.assertFalse(res.ok())
        self.assertIn("length", res.reason)

    def test_normal_body_is_kept(self):
        seat = _seat("R1", "R")
        res = classify_response(seat, {"choices": [{"finish_reason": "stop",
                                                    "message": {"content": " 반론 1 "}}]})
        self.assertTrue(res.ok())
        self.assertEqual(res.text, "반론 1")

    def test_malformed_payload_is_an_abstention(self):
        res = classify_response(_seat("R1", "R"), {"error": "boom"})
        self.assertTrue(res.abstained)


class TestVoidRound(unittest.TestCase):
    def test_proposer_abstention_voids_the_round(self):
        results = [
            SeatResult(seat="P1", archetype="P", abstained=True, reason="빈 응답"),
            SeatResult(seat="R1", archetype="R", text="반론"),
        ]
        self.assertIn("제안(P)", round_invalid_reason(results))

    def test_synthesiser_abstention_voids_the_round(self):
        results = [
            SeatResult(seat="P1", archetype="P", text="제안"),
            SeatResult(seat="S1", archetype="S", abstained=True, reason="빈 응답"),
        ]
        self.assertIn("종합(S)", round_invalid_reason(results))

    def test_reviewer_abstention_does_not_void_the_round(self):
        # A missing reviewer is a recorded gap, not a void round.
        results = [
            SeatResult(seat="P1", archetype="P", text="제안"),
            SeatResult(seat="A1", archetype="A", abstained=True, reason="빈 응답"),
            SeatResult(seat="S1", archetype="S", text="종합"),
        ]
        self.assertEqual(round_invalid_reason(results), "")


if __name__ == "__main__":
    unittest.main()
