"""The metrics decide whether Phase B stays, so a parser bug is a silent lie.

Round 21 reported supported_claims=0 while its 지지 section held numbered bold
items, and Phase Q reported 0 answers from 20 questions because the reply did
not use the exact bracket form the prompt asked for. Both were parser defects,
not findings -- pinned here so they cannot recur unnoticed.
"""

import re
import unittest

ITEM = re.compile(r"^\s*(?:[-*+]\s+|\*\*\d{1,2}[.)]|\d{1,2}[.)]\s)")
HEAD = re.compile(r"^\s*(?:\*\*)?\[?(\d{1,2})\]?[.)\]]?\s*(?:\*\*)?")


class TestSupportedClaimCounting(unittest.TestCase):
    def test_counts_every_numbering_style_seats_actually_use(self):
        for line in ("- 항목", "* 항목", "**1. 굵은 번호**", "2. 평범한 번호", "3) 괄호"):
            self.assertTrue(ITEM.match(line), line)

    def test_does_not_count_prose(self):
        for line in ("본문 문장이다.", "", "   ", "따라서 3개가 남는다"):
            self.assertFalse(ITEM.match(line), repr(line))

    def test_round_21_style_section_is_not_zero(self):
        section = "\n".join([
            "**1. 첫 항목**", "설명 문장.", "", "**2. 둘째 항목**", "설명 문장."])
        self.assertEqual(sum(1 for l in section.splitlines() if ITEM.match(l)), 2)


class TestQuestionBlockHeads(unittest.TestCase):
    def test_accepts_the_forms_models_produce(self):
        for line, want in (("[1]", "1"), ("1.", "1"), ("**2.**", "2"),
                           ("3)", "3"), ("[10] 답", "10")):
            m = HEAD.match(line)
            self.assertIsNotNone(m, line)
            self.assertEqual(m.group(1), want, line)

    def test_rejects_a_line_with_no_leading_number(self):
        self.assertIsNone(HEAD.match("질문 4").group(1) if HEAD.match("질문 4") else None)


if __name__ == "__main__":
    unittest.main()
