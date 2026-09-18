"""talks/ops/llm.local.json이 실제로 llm.json을 이긴다는 것만 확인한다.

주소를 코드가 아니라 파일이 정하는 구조라(D-053), 우선순위가 조용히 뒤집히면
잡무 스크립트는 예외 없이 죽은 포트로 POST한다. 이 테스트는 그 우선순위와
describe()가 출처를 밝히는지를 고정한다.
"""

import importlib
import json
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "ops"))
import llm_env  # noqa: E402


class LocalOverride(unittest.TestCase):
    def setUp(self):
        self.saved = {}
        for attr in ("CONFIG", "LOCAL"):
            self.saved[attr] = getattr(llm_env, attr)

    def tearDown(self):
        for attr, value in self.saved.items():
            setattr(llm_env, attr, value)

    @staticmethod
    def _write(path: Path, endpoint: str):
        # No cleanup hook: every caller writes inside a TemporaryDirectory that
        # is removed first, and unlinking after that raises FileNotFoundError.
        path.write_text(json.dumps({"model": "m", "endpoints": [endpoint]}),
                        encoding="utf-8")

    def test_local_wins_over_tracked(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            llm_env.CONFIG = Path(d) / "llm.json"
            llm_env.LOCAL = Path(d) / "llm.local.json"
            self._write(llm_env.CONFIG, "http://tracked:8000/v1/chat/completions")
            self._write(llm_env.LOCAL, "http://local:8000/v1/chat/completions")
            self.assertEqual(llm_env.endpoint(), "http://local:8000/v1/chat/completions")
            self.assertIn("llm.local.json", llm_env.describe())

    def test_tracked_used_when_no_local(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            llm_env.CONFIG = Path(d) / "llm.json"
            llm_env.LOCAL = Path(d) / "llm.local.json"
            self._write(llm_env.CONFIG, "http://tracked:8000/v1/chat/completions")
            self.assertEqual(llm_env.endpoint(), "http://tracked:8000/v1/chat/completions")
            self.assertIn("llm.json", llm_env.describe())

    def test_default_when_neither_exists(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            llm_env.CONFIG = Path(d) / "llm.json"
            llm_env.LOCAL = Path(d) / "llm.local.json"
            self.assertEqual(llm_env.endpoints(), llm_env.DEFAULT["endpoints"])
            self.assertIn("built-in default", llm_env.describe())


if __name__ == "__main__":
    unittest.main()
