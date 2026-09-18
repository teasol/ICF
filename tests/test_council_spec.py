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
    count_dissent,
    count_dissent_declared,
    harvest_hypotheses,
    lexical_overlap,
    round_invalid_reason,
    validate_card,
    verified_quotes,
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


class TestCrossRoundAnchoring(unittest.TestCase):
    """A prior round's report may not be a seat input.

    Round 11 was given round 10's report with an explicit instruction to attack
    it, and reproduced its top recommendation almost verbatim. Prose in AGENTS.md
    did not prevent that; the card validator has to.
    """

    def test_prior_report_is_rejected(self):
        card = _card(state_documents=["talks/council/C-20260918-10/report.md"])
        errors = validate_card(card)
        self.assertTrue(any("앞선 회차의 보고서" in e for e in errors), errors)

    def test_prior_blackboard_is_rejected_too(self):
        seats = [_seat("P1", "P"), _seat("R1", "R"), _seat("S1", "S")]
        seats[0].data_slice = ["talks/council/C-20260917-5/blackboard.json"]
        card = _card(seats=seats)
        errors = validate_card(card)
        self.assertTrue(any("앞선 회차의 보고서" in e for e in errors), errors)

    def test_round_card_of_a_prior_round_is_allowed(self):
        # The card states the question and composition, not the synthesis, so it
        # carries no conclusion to anchor on.
        card = _card(state_documents=["talks/council/C-20260918-10_card.json"])
        self.assertEqual(validate_card(card), [])


# --- Free dialogue boundary --------------------------------------------------
# The whole point of Phase B is that the transcript does NOT cross into the
# council. These tests pin the boundary, because a future edit that "helpfully"
# passes the reasoning along would silently undo the anchoring protection
# without breaking anything else.

def test_harvest_takes_only_the_title_not_the_argument():
    transcript = [
        "여기까지 길게 논증했다. 근거는 A, B, C이고 따라서 이렇게 본다.\n"
        "가설: context PCA는 train split에서만 적합한다\n",
    ]
    got = harvest_hypotheses(transcript)
    assert got == ["context PCA는 train split에서만 적합한다"]
    assert "근거" not in " ".join(got)


def test_harvest_dedupes_and_truncates_titles_carrying_arguments():
    long_title = "가설: " + "왜냐하면 " * 40
    got = harvest_hypotheses([long_title, "가설: 같은 것\n", "가설: 같은 것\n"])
    assert got.count("같은 것") == 1
    assert all(len(g) <= 161 for g in got)


def test_harvest_ignores_attribution():
    got = harvest_hypotheses(["## B1 (턴 1)\n가설: 임계값을 사전 고정한다\n"])
    assert got == ["임계값을 사전 고정한다"]


def test_dissent_counts_turns_not_phrases():
    turns = ["동의하지 않는다. 그리고 동의하지 않는다.", "좋은 생각이다", "그건 아니다"]
    assert count_dissent(turns) == 2


def test_echo_chamber_registers_as_zero_dissent():
    assert count_dissent(["좋다", "훌륭하다", "그 위에 얹겠다"]) == 0


# --- Document answers may only quote -----------------------------------------

def test_unverifiable_quotes_are_dropped_not_trusted():
    corpus = "fold-mean AUROC: 0.4426   n_folds=50\n승격 임계는 +0.0030이다"
    answer = "fold-mean AUROC: 0.4426   n_folds=50\n승격 임계는 +0.0500이다"
    kept, dropped = verified_quotes(answer, corpus)
    assert kept == ["fold-mean AUROC: 0.4426   n_folds=50"]
    assert dropped == 1


def test_a_fabricated_answer_survives_nothing():
    kept, dropped = verified_quotes("이 값은 문서에 분명히 적혀 있습니다", "전혀 다른 내용")
    assert kept == []
    assert dropped == 1


# --- Overlap metric ----------------------------------------------------------

def test_identical_proposals_score_full_overlap():
    assert lexical_overlap("PCA fold 적합", "PCA fold 적합") == 1.0


def test_disjoint_proposals_score_zero():
    assert lexical_overlap("PCA 적합 범위", "subsample 임계 발동") == 0.0


# --- Card compatibility ------------------------------------------------------

def test_card_without_brainstorm_still_loads():
    card = RoundCard.from_dict({
        "round_id": "C-1", "question": "q", "ru_type": "탐색",
        "composition_rationale": "r", "state_documents": [], "seats": [],
    })
    assert card.brainstorm == {}
    assert card.brainstorm_seats() == []


def test_brainstorm_seats_are_separate_from_council_seats():
    card = RoundCard.from_dict({
        "round_id": "C-1", "question": "q", "ru_type": "탐색",
        "composition_rationale": "r", "state_documents": [],
        "seats": [{"name": "P1", "archetype": "P", "endpoint": "e"}],
        "brainstorm": {"turns": 2,
                       "seats": [{"name": "B1", "endpoint": "e"},
                                 {"name": "B2", "endpoint": "e"}]},
    })
    assert [s.name for s in card.brainstorm_seats()] == ["B1", "B2"]
    assert [s.name for s in card.seats] == ["P1"]


def test_brainstorm_seats_do_not_inherit_the_proposal_mandate():
    """Round 17's failure: brainstorm seats got the P mandate and wrote reports."""
    card = RoundCard.from_dict({
        "round_id": "C-1", "question": "q", "ru_type": "탐색",
        "composition_rationale": "r", "state_documents": [], "seats": [],
        "brainstorm": {"turns": 2, "seats": [{"name": "B1", "endpoint": "e"}]},
    })
    prompt = card.brainstorm_seats()[0].system_prompt()
    assert "바꾸는 전제" not in prompt
    assert "보고서를 쓰는 자리가 아닙니다" in prompt
    assert "이견:" in prompt


def test_declared_dissent_is_not_fooled_by_the_word_appearing_in_prose():
    loose = ["그것이 아니라 이것이다. 반대로 보면 다르다."]
    assert count_dissent(loose) == 0          # keyword form: no marker present
    assert count_dissent_declared(loose) == 0  # declared form: no 이견 line
    assert count_dissent_declared(["이견: 전제가 틀렸다"]) == 1
