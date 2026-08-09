"""The CV-only path must survive the dead-branch prune bit-for-bit.

`TestCovarianceOnly` used to guard this path by comparing it against a
full-branch model carrying the same weights. The prune deletes that comparison
target, so the guard becomes a recorded fixture: outputs captured from the
pre-prune tree (commit `8caa96c`) and replayed here.

The fixture also carries the pre-prune `state_dict`. That matters -- after the
prune, constructing a model under the same seed draws DIFFERENT random numbers
for the surviving parameters, because deleting modules changes how much RNG the
constructor consumes. So the test loads the recorded weights rather than
trusting init to match, which isolates what it actually means to test: the
FORWARD COMPUTATION, not the initialisation order.

Regenerate only from a tree whose behaviour is already trusted:
    python tests/fixtures/make_cvonly_golden.py
Regenerating from the post-prune tree would make the test vacuous.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import torch

from tests.fixtures.make_cvonly_golden import (
    REPO_ROOT,
    build_model,
    episodes_for,
)
from src.utils.utils import merge_train_config

FIXTURE = Path(__file__).parent / "fixtures" / "cvonly_golden.pt"
# float32 forward on the same inputs and weights: differences must be round-off,
# not algorithm. Sizing these needed measurement rather than a guess.
#
# The prune leaves `_bag_view`, `_covariance_sketch` and
# `_projected_covariance_matrix` byte-identical, and `covariance_sketch` comes
# out BIT-identical across the prune. `covariance_matrix` does not: it differs
# by <=1.2e-10 absolute on entries whose max is 8.5e-4, i.e. ~1 ulp of float32.
# Same source, same input, different accumulation order -- the pruned model
# allocates ~43M fewer parameters, so the tensors land at different addresses
# and oneDNN/cuBLAS pick different vectorised kernels. Each tree is
# individually deterministic (three runs, identical to 12 decimals); only the
# cross-tree comparison moves.
#
# That 1-ulp input then feeds an eigendecomposition of a near-degenerate
# operator, so CV-2 can amplify it. One recorded episode is pathological: its
# logits reach +-713 (v44's identity margin is unbounded) and there CPU and GPU
# disagree by 1.23 on the SAME tree. rtol has to clear that amplification while
# atol stays tight enough that dropping any term still fails loudly.
ATOL = 1e-4
RTOL = 2e-3


@unittest.skipUnless(FIXTURE.exists(), f"missing fixture {FIXTURE}")
class CovarianceOnlyGoldenTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.golden = torch.load(FIXTURE, map_location="cpu", weights_only=False)
        cls.device = "cuda" if torch.cuda.is_available() else "cpu"

    def test_forward_matches_pre_prune_recording(self) -> None:
        for config_path, recorded in self.golden["configs"].items():
            with self.subTest(config=config_path):
                config = merge_train_config(REPO_ROOT / config_path)
                model = build_model(config).to(self.device).eval()

                # The fixture pins the weights the CV-only path can reach (the
                # "covariance" ones); everything else is dead weight the forward
                # never reads, so it is left at whatever init produced and is
                # NOT compared. Scoping the assertion this way keeps the test
                # meaningful on both sides of the prune: pre-prune the model
                # still carries the dead parameters, post-prune it does not.
                state = model.state_dict()
                reachable = [k for k in state if "covariance" in k]
                missing = [k for k in reachable if k not in recorded["state_dict"]]
                self.assertEqual(
                    missing,
                    [],
                    f"{config_path}: reachable parameters absent from the "
                    f"pre-prune recording: {missing}",
                )
                self.assertTrue(reachable, f"{config_path}: nothing to pin")
                model.load_state_dict(
                    {k: recorded["state_dict"][k].to(state[k].dtype) for k in reachable},
                    strict=False,
                )

                episodes = episodes_for(config, self.golden["episodes"])
                for index, (episode, expected) in enumerate(
                    zip(episodes, recorded["episodes"])
                ):
                    x = episode.x.to(self.device)
                    y = episode.y.to(self.device)
                    query_index = torch.tensor([x.shape[0] - 1], device=self.device)
                    with torch.no_grad():
                        logits, auxiliary = model(
                            x, y, query_index, return_auxiliary=True
                        )
                    torch.testing.assert_close(
                        logits.float().cpu(),
                        expected["logits"],
                        atol=ATOL,
                        rtol=RTOL,
                        msg=f"{config_path} episode {index}: final logits drifted",
                    )
                    for key in self.golden["aux_keys"]:
                        if key not in expected:
                            continue
                        value = auxiliary.get(key)
                        self.assertIsInstance(
                            value,
                            torch.Tensor,
                            f"{config_path}: auxiliary '{key}' disappeared",
                        )
                        torch.testing.assert_close(
                            value.float().cpu(),
                            expected[key],
                            atol=ATOL,
                            rtol=RTOL,
                            msg=f"{config_path} episode {index}: '{key}' drifted",
                        )


if __name__ == "__main__":
    unittest.main()
