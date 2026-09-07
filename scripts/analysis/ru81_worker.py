"""Run an official-fold shard with isolated RU-81 solver instrumentation."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import torch
import test_pathobench as evaluator
from scripts.analysis.ru81_probe import install

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--diagnostic-output", required=True)
args, remaining = parser.parse_known_args()
sys.argv = [sys.argv[0], *remaining]
rows = install(evaluator)
evaluator.main()
torch.save({"rows": rows}, args.diagnostic_output)
