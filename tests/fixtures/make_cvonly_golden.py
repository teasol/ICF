"""Record golden CV-only outputs BEFORE the dead-branch prune.

Why this exists: `TestCovarianceOnly` guarded the CV-only path by comparing it
against a full-branch model carrying the same weights. The prune deletes that
comparison target, so the guard has to become a numerical fixture instead --
otherwise the prune could silently change the model and no test would notice
(the SS62-7 duplicate-drift failure mode).

Run from the PRE-prune tree (git worktree at 8caa96c) to regenerate:

    python tests/fixtures/make_cvonly_golden.py

It writes `tests/fixtures/cvonly_golden.pt`, which
`tests/test_cvonly_golden.py` replays. Deleting and regenerating this file from
the post-prune tree would defeat its purpose -- it must come from the tree
whose behaviour is being preserved.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.datasets.synthetic_data import SyntheticManifoldGenerator  # noqa: E402
from src.utils.utils import merge_train_config  # noqa: E402

# One config per distinct sketch geometry / CV-2 setting in flight, so the
# fixture pins the knobs as well as the default path.
CONFIGS = (
    "configs/archive/v40_v45_cvonly/train_v41_cvonly_K128_1536.yaml",
    "configs/archive/v40_v45_cvonly/train_v41_cvonly_K64_1536.yaml",
    "configs/archive/v40_v45_cvonly/train_v42_rank2_1536.yaml",
    "configs/archive/v40_v45_cvonly/train_v44_lowT_1536.yaml",
)
EPISODES = 6
SEED = 42
# Keys whose values are compared exactly. Scale scalars are included on
# purpose: a prune that dropped a gate would otherwise pass unnoticed.
AUX_KEYS = (
    "covariance_logits",
    "covariance_ridge_logits",
    "covariance_ridge_scale",
    "covariance_residual_scale",
    "covariance_relation_logits",
    "covariance_relation_class_separation",
)


def build_model(config: dict):
    import importlib
    import inspect

    kwargs = {**config["model"], **config.get("model_kwargs", {})}
    for key, value in (config.get("model_overrides") or {}).items():
        kwargs[key] = value
    module_name, class_name = kwargs.pop("model_src").rsplit(".", 1)
    cls = getattr(importlib.import_module(module_name), class_name)
    accepted = inspect.signature(cls.__init__).parameters
    return cls(**{k: v for k, v in kwargs.items() if k in accepted})


def episodes_for(config: dict, count: int):
    import inspect

    accepted = inspect.signature(SyntheticManifoldGenerator.__init__).parameters
    # Small bags: the fixture pins numerics, not scale, and must stay fast
    # enough to run in the compact suite on CPU-sized GPUs.
    kwargs = {
        k: v
        for k, v in dict(config["data"].get("dataset_kwargs", {})).items()
        if k in accepted
    }
    kwargs["num_cells"] = [1, 512]
    # The recording predates per-bag cardinality; replay its exact dense input
    # contract even though the active training default is now ragged (§81).
    kwargs["per_bag_cardinality"] = False
    kwargs["num_bags"] = [12, 16]
    generator = SyntheticManifoldGenerator(**kwargs)
    rng = torch.Generator().manual_seed(SEED)
    return [generator.sample_episode(generator=rng) for _ in range(count)]


def record(config_path: str, device: str = "cuda") -> dict:
    config = merge_train_config(REPO_ROOT / config_path)
    torch.manual_seed(SEED)
    model = build_model(config).to(device).eval()

    entries = []
    for episode in episodes_for(config, EPISODES):
        x = episode.x.to(device)
        y = episode.y.to(device)
        query_index = torch.tensor([x.shape[0] - 1], device=device)
        with torch.no_grad():
            logits, auxiliary = model(x, y, query_index, return_auxiliary=True)
        entry = {"logits": logits.float().cpu()}
        for key in AUX_KEYS:
            value = auxiliary.get(key)
            if isinstance(value, torch.Tensor):
                entry[key] = value.float().cpu()
        entries.append(entry)

    # Only the weights the CV-only path can reach. Recording the whole
    # state_dict would put 691 MB of dead encoder weights in git for four
    # configs -- and every one of them is a tensor the forward never reads.
    # If the prune leaves a live parameter outside this filter, the replay test
    # fails loudly on the "absent pre-prune" assertion rather than silently
    # comparing against nothing.
    state = {
        k: v.float().cpu()
        for k, v in model.state_dict().items()
        if "covariance" in k
    }
    return {"episodes": entries, "state_dict": state}


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    golden = {
        "seed": SEED,
        "episodes": EPISODES,
        "aux_keys": list(AUX_KEYS),
        "configs": {path: record(path, device) for path in CONFIGS},
    }
    out = Path(__file__).with_name("cvonly_golden.pt")
    torch.save(golden, out)
    total = sum(len(v["episodes"]) for v in golden["configs"].values())
    print(f"wrote {out}  ({len(CONFIGS)} configs x {EPISODES} episodes = {total})")


if __name__ == "__main__":
    main()
