"""문서 정합성 회귀 테스트.

문서 drift를 사람의 주의력이 아니라 테스트로 막는다.

이 테스트가 존재하는 이유 (2026-09-06 재구성 시점에 실제로 발견된 결함):
  - README.md 는 "Active v120 6-branch, Primary 7 macro 0.6265" 라고 적고 있었으나
    공식 비교 기준은 v121 5-branch 0.6171 이었다. 정본 세 곳이 서로 다른 값을 말했다.
  - agent_handoff.md 는 회귀 스위트를 "119 tests" 로 적고 있었으나 실제는 137개였다.
    (그 뒤 이 파일이 추가되며 146개가 됐고 문서는 다시 137에 머물렀다 — 그래서
    TestDocumentedTestCount 로 사람의 주의력 대신 테스트가 지키게 했다.)
  - current_status.md 는 190줄까지 자라 상태·결정·정정·환경이 뒤섞였다.
  - §226 에서 낡은 서술 2건이 브리핑 팩을 통해 7개 에이전트 전부에 전파됐다.

자세한 배경은 docs/history/archive.md 의 결정 이력 D-015 참조.
"""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs"

PROJECT_MD = DOCS / "PROJECT.md"
CLOSED_AXES_MD = DOCS / "closed_axes.md"
DECISIONS_MD = DOCS / "history" / "archive.md"   # 결정 이력은 archive 말미로 통합됐다
STATUS_MD = DOCS / "current_status.md"
AGENT_HANDOFF_MD = DOCS / "agent_handoff.md"

#: PROJECT.md 만 선언할 수 있는 수치. 다른 Living 문서에 나타나면 drift다.
GUARDED_NUMBERS = (
    "0.6171",  # 공식 비교 기준선 (v121 5-branch)
    "0.6265",  # v120 6-branch 계보 기록값
    "0.6972",  # v120 SEAL 10 hold-out
    "0.6681",  # v120 전체 17
    "0.7266",  # SEAL ABMIL 목표
    "0.7125",  # SEAL MeanMIL
)

#: 위 수치를 담아서는 안 되는 Living 문서.
#: closed_axes.md 는 과거 측정값을 근거로 인용하므로 면제한다
#: (현행 기준을 선언하지 않는다). 결정 이력은 history/archive.md 로 통합됐다.
GUARDED_FILES = (
    "README.md",
    "agent_handoff.md",
    "current_status.md",
    "current_architecture.md",
)

#: current_status.md 가 다시 자라지 않도록 하는 상한. 문서 자체가 선언한 값과 같다.
STATUS_MAX_LINES = 80

#: 닫힌 축 상세 항목이 반드시 갖춰야 하는 필드.
AXIS_REQUIRED_FIELDS = ("기각 기전", "경계 안", "경계 밖", "재개 조건")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


class TestSingleSourceOfTruth(unittest.TestCase):
    """수치는 PROJECT.md 에서만 선언한다."""

    def test_project_md_declares_the_numbers(self):
        text = _read(PROJECT_MD)
        for num in GUARDED_NUMBERS:
            self.assertIn(
                num, text,
                f"PROJECT.md 가 기준 수치 {num} 를 더 이상 선언하지 않는다. "
                f"수치를 옮겼다면 이 테스트의 GUARDED_NUMBERS 도 함께 갱신할 것.",
            )

    def test_no_duplicate_numbers_in_living_docs(self):
        for name in GUARDED_FILES:
            path = DOCS / name
            if not path.exists():
                continue
            text = _read(path)
            for num in GUARDED_NUMBERS:
                # assertNotIn 은 실패 시 파일 전문을 덤프하므로 assertTrue 로 쓴다.
                self.assertTrue(
                    num not in text,
                    f"{name} 이 기준 수치 {num} 를 직접 적고 있다. "
                    f"수치는 docs/PROJECT.md 에서만 선언하고 나머지는 링크한다 "
                    f"(README 가 0.6265 를, handoff 가 0.6171 을 각각 다르게 적던 "
                    f"drift 를 막기 위한 규칙이다).",
                )


class TestReferentialIntegrity(unittest.TestCase):
    """문서가 인용하는 축 ID·결정 ID 가 실제로 존재해야 한다."""

    @staticmethod
    def _defined_axis_ids() -> set[str]:
        # §1 일람표의 `CA-xx` 백틱 표기를 정의로 본다.
        return set(re.findall(r"`(CA-[A-Z0-9]+)`", _read(CLOSED_AXES_MD)))

    @staticmethod
    def _defined_decision_ids() -> set[str]:
        return set(re.findall(r"^## (D-\d{3})", _read(DECISIONS_MD), re.MULTILINE))

    def test_referenced_axis_ids_exist(self):
        defined = self._defined_axis_ids()
        self.assertTrue(defined, "closed_axes.md 에 축 정의가 하나도 없다.")
        for path in sorted(DOCS.glob("*.md")):
            for ref in set(re.findall(r"`(CA-[A-Z0-9]+)`", _read(path))):
                self.assertIn(
                    ref, defined,
                    f"{path.name} 이 존재하지 않는 축 {ref} 를 인용한다.",
                )

    def test_referenced_decision_ids_exist(self):
        defined = self._defined_decision_ids()
        self.assertTrue(defined, "archive.md 에 결정 레코드가 하나도 없다.")
        for path in sorted(DOCS.glob("*.md")):
            for ref in set(re.findall(r"`(D-\d{3})`", _read(path))):
                self.assertIn(
                    ref, defined,
                    f"{path.name} 이 존재하지 않는 결정 {ref} 를 인용한다.",
                )


class TestClosedAxesRegistry(unittest.TestCase):
    """축을 닫으려면 경계를 함께 적어야 한다 (§226 벤더 충돌 3건의 원인)."""

    def test_every_axis_defines_its_boundary(self):
        text = _read(CLOSED_AXES_MD)
        # §2 축별 상세만 검사한다. §1 은 일람표, §3 은 미확정 목록이다.
        try:
            body = text.split("## 2. 축별 상세", 1)[1].split("## 3. 경계 미확정", 1)[0]
        except IndexError:  # pragma: no cover - 구조가 바뀌면 즉시 실패시킨다
            self.fail("closed_axes.md 의 §2/§3 구조가 바뀌었다.")

        sections = re.split(r"^### ", body, flags=re.MULTILINE)[1:]
        self.assertTrue(sections, "closed_axes.md §2 에 축 상세가 없다.")
        for sec in sections:
            axis = sec.splitlines()[0].strip()
            if "REOPENED" in axis:
                # 철회되어 다시 열린 축은 경계를 요구하지 않는다. 대신 무엇이 해소돼야
                # 활용 가능한지를 적는다.
                self.assertTrue(
                    "재개 조건" in sec,
                    f"축 '{axis}' 은 REOPENED 인데 재개 조건이 없다.",
                )
                continue
            for field in AXIS_REQUIRED_FIELDS:
                self.assertTrue(
                    field in sec,
                    f"축 '{axis}' 에 '{field}' 가 없다. 축을 닫으려면 "
                    f"기각 기전·경계 안·경계 밖·재개 조건이 모두 필요하다.",
                )

    def test_axis_table_and_details_agree(self):
        text = _read(CLOSED_AXES_MD)
        table = text.split("## 2. 축별 상세", 1)[0]
        listed = set(re.findall(r"\| `(CA-[A-Z0-9]+)`", table))
        detailed = set(re.findall(r"^### `(CA-[A-Z0-9]+)`", text, re.MULTILINE))
        self.assertEqual(
            listed, detailed,
            "closed_axes.md §1 일람표와 §2 상세의 축 목록이 다르다: "
            f"표에만 {sorted(listed - detailed)}, 상세에만 {sorted(detailed - listed)}",
        )


class TestLivingDocsAreSelfContained(unittest.TestCase):
    """living 문서는 이력을 열지 않아도 읽히도록 쓴다.

    결정 이력(`D-xxx`)은 **출처 표기**로만 쓴다. 규칙의 내용을 이력 ID에 위임하면
    ("자세한 것은 D-021에 있다") 독자가 링크를 타야 현행 규칙을 알 수 있고, 그것이
    §226에서 낡은 사실이 전파된 경로였다. 아래 형태를 금지한다.
    """

    #: 규칙 내용을 이력 ID에 위임하는 서술 형태.
    DELEGATING = (
        r"`D-\d{3}`[을를] 따른다",
        r"`D-\d{3}`에 (있다|적혀 있다|기록돼 있다)",
        r"`D-\d{3}`[을를] (참조|참고)",
        r"`D-\d{3}` (참조|참고)",
        r"(자세한|상세(한)?|구체적(인)?)[^.\n]{0,20}`D-\d{3}`",
    )

    #: 현행 사실을 선언하는 living 문서. history/ 와 reports/ 는 기록물이므로 제외한다.
    LIVING = ("PROJECT.md", "current_status.md", "agent_handoff.md", "closed_axes.md",
              "README.md", "current_architecture.md", "research_directions.md")

    def test_no_rule_content_delegated_to_decision_ids(self):
        for name in self.LIVING:
            path = DOCS / name
            if not path.exists():
                continue
            for pattern in self.DELEGATING:
                hit = re.search(pattern, _read(path))
                self.assertIsNone(
                    hit,
                    f"{name} 이 규칙 내용을 결정 이력 ID에 위임하고 있다: "
                    f"{hit.group(0) if hit else ''!r}. living 문서의 각 항목은 "
                    f"이력을 열지 않아도 무엇을 어떻게 할지 알 수 있게 쓴다. "
                    f"ID 는 출처 표기로만 남긴다.",
                )


class TestStatusDocStaysShort(unittest.TestCase):
    """current_status.md 가 다시 190줄로 자라지 않게 한다."""

    def test_line_budget(self):
        n = len(_read(STATUS_MD).splitlines())
        self.assertLessEqual(
            n, STATUS_MAX_LINES,
            f"current_status.md 가 {n}줄이다 (상한 {STATUS_MAX_LINES}). "
            f"종료된 절과 결정 이력은 docs/history/archive.md 로 옮긴다.",
        )


class TestDocumentedTestCount(unittest.TestCase):
    """문서에 적힌 회귀 스위트 규모가 실제 수집 개수와 같아야 한다."""

    #: "146 tests" 형태를 찾는다. 두 Living 문서가 같은 수를 말해야 한다.
    _PATTERN = re.compile(r"(\d+) tests")

    @staticmethod
    def _discovered() -> int:
        # run_tests.sh 와 같은 방식으로 수집한다: `unittest discover -s tests -p 'test_*.py'`.
        # 실행하지 않고 개수만 센다. 모듈은 이 스위트가 이미 임포트한 것이라 sys.modules 에 있다.
        suite = unittest.TestLoader().discover(str(REPO / "tests"), pattern="test_*.py")
        return suite.countTestCases()

    def test_docs_state_the_real_count(self):
        actual = self._discovered()
        for path in (STATUS_MD, AGENT_HANDOFF_MD):
            found = self._PATTERN.findall(_read(path))
            self.assertTrue(
                found, f"{path.name} 에 '<n> tests' 표기가 없다. 스위트 규모를 명시한다.",
            )
            for n in found:
                self.assertEqual(
                    int(n), actual,
                    f"{path.name} 는 회귀 스위트를 {n} tests 로 적었으나 실제는 {actual}개다. "
                    f"테스트를 추가·삭제했으면 두 문서의 수치를 함께 고친다.",
                )


class TestResearchUnitsLedger(unittest.TestCase):
    """연구 이력 총람이 기계적으로 읽히는 상태를 유지해야 한다."""

    @staticmethod
    def _units() -> list[dict]:
        path = DOCS / "history" / "research_units_all.json"
        return json.loads(path.read_text(encoding="utf-8"))["units"]

    def test_ids_are_unique_and_contiguous(self):
        ids = [u["id"] for u in self._units()]
        self.assertEqual(len(ids), len(set(ids)), "RU ID 가 중복된다.")
        nums = sorted(int(i.split("-")[1]) for i in ids)
        # 아직 열려 있는 카드(docs/ru/RU-NN.json)는 총람에 없는 것이 정상이다 --
        # ru.py 는 close 시점에 append 하므로 종료 순서가 번호 순서와 다를 수 있다.
        # 그 번호만 빼고 1..max 가 빠짐없이 채워졌는지 검사한다.
        open_ids = {int(f.stem.split("-")[1]) for f in (DOCS / "ru").glob("RU-*.json")}
        expected = [n for n in range(1, max(nums) + 1) if n not in open_ids]
        self.assertEqual(
            nums, expected,
            "RU 번호가 연속이 아니다 (열린 카드 제외). "
            "ru.py 를 거치지 않고 편집했을 가능성이 있다.",
        )

    def test_units_carry_a_decision(self):
        for u in self._units():
            self.assertTrue(
                str(u.get("decision", "")).strip(),
                f"{u['id']} 에 decision 이 없다. 결정 없이 종료된 RU 는 기록하지 않는다.",
            )


if __name__ == "__main__":
    unittest.main()
