"""Task-level branch diagnosis from dumped per-branch margins (RU-101).

Reproduces the official trimmed_mean macro from stored per-branch margins and
reports, per task: the 7-branch and shape-dropped 5-branch fold-mean AUROC, and
each branch's standalone AUROC. Used to see whether SH/SJ carry signal exactly
where the 7-branch worsens.

  python scripts/analysis/task_branch_diagnosis.py \
      --margins predictions/margins_7b \
      --config configs/baseline/v121_7branch_active.yaml
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import statistics as st
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.config import TrainingFreeConfig  # noqa: E402
from src.models.aggregations.voting import trimmed_mean  # noqa: E402
from src.utils.metrics import auroc  # noqa: E402

SHAPE = ("sh", "sj")


def aggregate(cfg: TrainingFreeConfig, b: dict, with_shape: bool):
    return trimmed_mean(
        cfg, cv=b["cv"], m_cv=b["cv"], m_dd=None, m_ct=None, m_bm=b["bm"],
        m_bd=b["bd"], m_qa=b["qa"], m_ds=b["ds"], m_lr=None, m_de=None,
        m_sw=None,
        m_sj=b["sj"] if with_shape else None,
        m_sh=b["sh"] if with_shape else None,
    )


def diagnose(margins_dir: Path, config_path: Path) -> int:
    cfg7 = TrainingFreeConfig.from_yaml(config_path)
    cfg5 = dataclasses.replace(cfg7, weight_sh=0.0, weight_sj=0.0, weight_shj=0.0)
    files = sorted(glob.glob(str(margins_dir / "*.pt")))
    if not files:
        print(f"margin 파일이 없다: {margins_dir}", file=sys.stderr)
        return 2

    macro7, macro5, rows = [], [], []
    print(f"{'task':<34}{'7br':>8}{'5br':>8}{'Δ':>9} | "
          + " ".join(f"{k}:standalone" for k in SHAPE))
    for f in files:
        d = torch.load(f, weights_only=False, map_location="cpu")
        a7, a5 = [], []
        standalone = {k: [] for k in SHAPE}
        for pf in d["per_fold"]:
            y = pf["label"]
            b = {k: v.float() for k, v in pf["branch_margins"].items()}
            a7.append(float(auroc(aggregate(cfg7, b, True).float(), y)))
            a5.append(float(auroc(aggregate(cfg5, b, False).float(), y)))
            for k in SHAPE:
                standalone[k].append(float(auroc(b[k], y)))
        m7, m5 = st.mean(a7), st.mean(a5)
        macro7.append(m7)
        macro5.append(m5)
        rows.append((d["task"], m7, m5, {k: st.mean(v) for k, v in standalone.items()}))
        print(f"{d['task']:<34}{m7:>8.4f}{m5:>8.4f}{m7 - m5:>+9.4f} | "
              + " ".join(f"{k}:{rows[-1][3][k]:.3f}" for k in SHAPE))

    print(f"\nmacro 7br {st.mean(macro7):.4f}  5br {st.mean(macro5):.4f}")
    print("주의: branch standalone AUROC은 trimmed_mean 기여의 인과 분해가 아니다.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--margins", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    args = ap.parse_args()
    return diagnose(args.margins, args.config)


if __name__ == "__main__":
    raise SystemExit(main())
