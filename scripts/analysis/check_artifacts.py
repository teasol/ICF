"""L1 attribution check for stored predictions (contamination-check redesign).

The value anchor in PROJECT.md SS3.4 was abolished by D-050: it could not tell a
configuration change from numeric drift, and the screen-only path it guarded is
dead (no code reads ICF_SHAPE_SCREEN_ONLY). What replaces it is not a value
check at all -- it checks ATTRIBUTION: does a stored prediction belong to the
declared configuration? Drift detection is LOST, not replaced; say so.

Spec: talks/reports/2026-09-19_contamination_check_spec.md
Scope: L1 only. L2 (same-machine re-run reproduces for the closed-form pipeline)
       is a run, not a file check, and lives with the runner.

  python scripts/analysis/check_artifacts.py \
      --artifacts predictions/ \
      --expect-config configs/baseline/v121_active.yaml \
      --expect-config configs/baseline/v121_7branch_active.yaml
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_pure import build_provenance  # noqa: E402
from src.models.config import TrainingFreeConfig  # noqa: E402

REQUIRED = ("branch_list", "weights", "config_sha256", "code_hash",
            "git_commit", "manifest_hash", "data_version")


def expected_for(config_path: Path) -> dict:
    """Declared branch_list and config hash for one expected configuration."""
    cfg = TrainingFreeConfig.from_yaml(config_path)
    prov = build_provenance(cfg, config_path, PROJECT_ROOT / "_none",
                            PROJECT_ROOT / "_none", torch.device("cpu"))
    return {"config_sha256": prov["config_sha256"],
            "branch_list": prov["branch_list"]}


def check_file(path: Path, expected: list[dict]) -> tuple[bool, str]:
    try:
        d = torch.load(path, weights_only=False, map_location="cpu")
    except Exception as exc:  # noqa: BLE001 - unreadable is a failure, not a crash
        return False, f"읽기 실패: {exc}"
    prov = d.get("provenance")
    if not isinstance(prov, dict):
        return False, "provenance 없음 (D-050 귀속 불가)"
    missing = [k for k in REQUIRED if not prov.get(k)]
    if missing:
        return False, f"provenance 필드 누락: {', '.join(missing)}"
    hashes = {e["config_sha256"] for e in expected}
    if prov["config_sha256"] not in hashes:
        return False, (f"config_sha256 {prov['config_sha256']} 가 기대 집합 "
                       f"{sorted(hashes)} 에 없다 (D-053 설정 치환 의심)")
    lists = {tuple(e["branch_list"]) for e in expected}
    if tuple(prov["branch_list"]) not in lists:
        return False, (f"branch_list {prov['branch_list']} 가 기대 "
                       f"{sorted(lists)} 와 다르다")
    return True, (f"ok config={prov['config_sha256']} "
                  f"branch_list={prov['branch_list']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifacts", required=True,
                    help="디렉토리 또는 glob (예: predictions/)")
    ap.add_argument("--expect-config", action="append", required=True,
                    help="기대 config (반복 가능)")
    args = ap.parse_args()

    expected = [expected_for(Path(c)) for c in args.expect_config]
    pattern = args.artifacts
    files = sorted(glob.glob(str(Path(pattern) / "*.pt")) if Path(pattern).is_dir()
                   else glob.glob(pattern))
    if not files:
        print(f"검사할 파일이 없다: {pattern}", file=sys.stderr)
        return 2

    bad = 0
    for f in files:
        ok, why = check_file(Path(f), expected)
        print(f"{'PASS' if ok else 'FAIL'}  {f}  {why}")
        bad += 0 if ok else 1
    print(f"\n{len(files)}개 검사 · 실패 {bad}개 (L1 귀속만; 드리프트는 검증하지 않는다)")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
