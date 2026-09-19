"""OpenCode is pointed at the local vLLM, and no external provider string runs.

Measured defect this replaces: `routine_opencode.sh` passed
`--model deepseek/deepseek-v4.1-flash`, an OpenCode provider string that
resolved to an external HTTPS service, because nothing in OpenCode read
`llm.local.json`. The model id is now derived from `llm_env` and namespaced under
a provider whose baseURL is the local endpoint.
"""

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

sys.path.insert(0, str(REPO_ROOT / "scripts" / "ops"))
from scripts.ops import opencode_config  # noqa: E402
import llm_env  # noqa: E402

ROUTINE = REPO_ROOT / "scripts" / "ops" / "routine_opencode.sh"


class TestBuild(unittest.TestCase):
    def test_config_points_at_local_base_url_and_model(self):
        cfg = opencode_config.build("deepseek-v4.1-flash",
                                    "http://192.168.100.100:8000/v1")
        prov = cfg["provider"][opencode_config.LOCAL_PROVIDER]
        self.assertEqual(prov["options"]["baseURL"],
                         "http://192.168.100.100:8000/v1")
        self.assertIn("deepseek-v4.1-flash", prov["models"])
        self.assertEqual(cfg["model"],
                         opencode_config.provider_model("deepseek-v4.1-flash"))

    def test_model_string_is_not_an_external_provider(self):
        model = opencode_config.provider_model("deepseek-v4.1-flash")
        self.assertFalse(model.startswith("deepseek/"))
        self.assertTrue(model.startswith(opencode_config.LOCAL_PROVIDER + "/"))

    def test_write_uses_llm_env_by_default(self):
        saved_model, saved_base = llm_env.model, llm_env.base_url
        try:
            llm_env.model = lambda: "m1"
            llm_env.base_url = lambda: "http://local:8000/v1"
            with TemporaryDirectory() as d:
                model = opencode_config.write(d)
                cfg = json.loads((Path(d) / "opencode.json").read_text(encoding="utf-8"))
        finally:
            llm_env.model, llm_env.base_url = saved_model, saved_base
        self.assertEqual(model, "icf-local/m1")
        self.assertEqual(cfg["provider"]["icf-local"]["options"]["baseURL"],
                         "http://local:8000/v1")


class TestScriptUsesLocalConfig(unittest.TestCase):
    def test_no_external_provider_string_remains(self):
        text = ROUTINE.read_text(encoding="utf-8")
        self.assertNotIn("deepseek/deepseek", text)
        # The model string is generated from the local config, not hard-coded.
        self.assertIn("opencode_config.py", text)
        self.assertIn("require_local_model", text)


if __name__ == "__main__":
    unittest.main()
