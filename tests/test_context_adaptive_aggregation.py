"""Pin the v37 context-adaptive bag aggregation contract.

`bag_aggregation='context_adaptive'` replaces the fixed 40->1 linear map with a
convex combination whose weights `w` are produced from THIS episode's context
bags. The weights are built in D space and applied to the original token_dim
tokens, so the bag token keeps its dimension and no downstream module changes.

The properties that make this sound (and that a refactor could silently break):

  * `w` is a function of the CONTEXT bags only -- a query bag must never reach it.
  * `w` is permutation-invariant over bags.
  * `w` is label-blind, hence trivially label-permutation invariant.
  * zero-init weight head => `w` starts uniform => the bag token starts as the
    plain mean of the structured tokens.
  * the dense (training) and ragged (eval) paths implement the same rule -- v37
    touches three duplicated call sites in the 4D path, so this is the guard.
"""

from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.models.baseline import BaseModel  # noqa: E402
from src.utils.utils import merge_train_config  # noqa: E402

INPUT_DIM = 96


def _build(aggregation: str = "context_adaptive", seed: int = 0) -> BaseModel:
    config = merge_train_config(
        REPO_ROOT / "configs" / "train_v36_q1_baseline_1536.yaml"
    )
    kwargs = dict(config["model"])
    kwargs.update(config["model_kwargs"])
    kwargs.pop("model_src", None)
    kwargs["input_dim"] = INPUT_DIM
    kwargs["aggregator_covariance_sketch_dim"] = 16
    kwargs["aggregator_slot_affinity_dim"] = INPUT_DIM
    kwargs["meta_bag_aggregation"] = aggregation
    accepted = inspect.signature(BaseModel.__init__).parameters
    kwargs = {k: v for k, v in kwargs.items() if k in accepted}
    torch.manual_seed(seed)
    return BaseModel(**kwargs).eval()


def _episode(n_bags: int = 8, n_cells: int = 150, seed: int = 1):
    torch.manual_seed(seed)
    x = torch.randn(n_bags, n_cells, INPUT_DIM)
    y = torch.tensor([0, 1] * (n_bags // 2))
    return x, y


def _context_tokens(model: BaseModel, x, n_context: int) -> torch.Tensor:
    mask = torch.zeros(x.shape[0], dtype=torch.bool)
    mask[:n_context] = True
    with torch.no_grad():
        representation = model.aggregator(x, context_mask=mask)
    context = {name: tokens[mask] for name, tokens in representation.items()}
    return model.meta_classifier._all_structured_tokens(context)


class TestContextAdaptiveAggregation(unittest.TestCase):
    def test_weights_start_uniform_and_reduce_to_the_mean(self) -> None:
        """zero-init head => w uniform => bag token == mean of the tokens."""
        model = _build()
        x, _ = _episode()
        tokens = _context_tokens(model, x, 6)
        meta = model.meta_classifier
        with torch.no_grad():
            weights = meta._context_aggregation_weights(tokens)
            reduced = meta._projected_bag_tokens(tokens, weights)
        n_tokens = meta.structured_tokens_per_bag
        self.assertEqual(tuple(weights.shape), (n_tokens,))
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=5)
        self.assertLess(
            float((weights - 1.0 / n_tokens).abs().max()),
            1e-6,
            "weight head is zero-init, so w must start exactly uniform.",
        )
        self.assertLess(float((reduced - tokens.mean(dim=-2)).abs().max()), 1e-5)

    def test_weights_are_bag_order_invariant(self) -> None:
        """Bags are a set; the encoder must carry no positional information."""
        model = _build()
        x, _ = _episode()
        tokens = _context_tokens(model, x, 6)
        permutation = torch.tensor([4, 0, 5, 2, 1, 3])
        meta = model.meta_classifier
        with torch.no_grad():
            straight = meta._context_aggregation_weights(tokens)
            shuffled = meta._context_aggregation_weights(tokens[permutation])
        self.assertLess(float((straight - shuffled).abs().max()), 1e-5)

    def test_weights_see_only_context_bags(self) -> None:
        """The query must never reach the weight maker (leak guard)."""
        model = _build()
        x, y = _episode(n_bags=8)
        seen: list[int] = []
        original = model.meta_classifier._context_aggregation_weights

        def spy(tokens):
            seen.append(tokens.shape[-3])
            return original(tokens)

        model.meta_classifier._context_aggregation_weights = spy
        query_index = torch.tensor([2, 5])
        with torch.no_grad():
            model(x, y, query_index)
        self.assertTrue(seen, "the weight maker was never called")
        expected_context = x.shape[0] - query_index.numel()
        self.assertTrue(
            all(count == expected_context for count in seen),
            f"weight maker saw {seen} bags but the context has "
            f"{expected_context}; a query bag is leaking into w.",
        )

    def test_weights_are_label_blind(self) -> None:
        """w must not move when the context labels are permuted."""
        model = _build()
        x, _ = _episode()
        tokens = _context_tokens(model, x, 6)
        meta = model.meta_classifier
        with torch.no_grad():
            weights = meta._context_aggregation_weights(tokens)
        # labels never enter the call at all -- pin the signature so a future
        # label-conditioned variant has to change this test deliberately.
        self.assertNotIn(
            "label",
            inspect.signature(meta._context_aggregation_weights).parameters,
        )
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=5)

    def test_dense_and_ragged_paths_agree(self) -> None:
        """v37 touches three duplicated call sites in the 4D path."""
        for aggregation in ("projected", "context_adaptive"):
            with self.subTest(aggregation=aggregation):
                model = _build(aggregation)
                x, y = _episode()
                mask = torch.tensor([2, 5])
                with torch.no_grad():
                    ragged = model(list(x.unbind(0)), y, mask)
                    dense = model.forward_episode_batch(
                        x.unsqueeze(0), y.unsqueeze(0), mask.unsqueeze(0)
                    )
                dense_logits = dense[0] if isinstance(dense, tuple) else dense
                delta = (ragged.float() - dense_logits.float()).abs().max()
                self.assertLess(
                    float(delta),
                    1e-4,
                    f"dense/ragged disagree in {aggregation!r} "
                    f"(||delta||inf={float(delta):.3e}).",
                )

    def test_weights_stay_a_simplex_point_across_context_sizes(self) -> None:
        """Training sees 60-100 bags, eval 133-262; w must stay well formed.

        The encoder's softmax is set-size sensitive, so this records the drift
        rather than asserting a bound -- a real bound needs a trained model.
        """
        model = _build()
        meta = model.meta_classifier
        # break the zero-init so the weights actually vary with the input
        with torch.no_grad():
            for parameter in meta.bag_agg_weight_mlp[-1].parameters():
                parameter.normal_(0.0, 0.05)
        reference = None
        for n_bags in (8, 16, 32):
            x, _ = _episode(n_bags=n_bags, n_cells=90, seed=n_bags)
            tokens = _context_tokens(model, x, n_bags - 2)
            with torch.no_grad():
                weights = meta._context_aggregation_weights(tokens)
            self.assertTrue(bool(torch.isfinite(weights).all()))
            self.assertAlmostEqual(float(weights.sum()), 1.0, places=4)
            if reference is None:
                reference = weights
            else:
                drift = float((weights - reference).abs().max())
                self.assertLess(drift, 1.0)

    def test_token_type_ids_follow_the_structured_layout(self) -> None:
        """[global, slot_i x (center, spread, rare), tails]."""
        from src.models.baseline import StructuredPopulationMetaClassifier

        num_slots, num_density, num_tails = 12, 8, 3
        n_tokens = 1 + 3 * num_slots + num_tails
        types, slot_class, tails = (
            StructuredPopulationMetaClassifier._build_bag_token_type_ids(
                n_tokens, num_slots, num_density, num_tails
            )
        )
        self.assertEqual(int(types[0]), 0)                      # global
        self.assertEqual([int(v) for v in types[1:4]], [1, 2, 3])   # slot 0
        self.assertEqual([int(v) for v in types[4:7]], [1, 2, 3])   # slot 1
        self.assertTrue(all(int(v) == 4 for v in types[-num_tails:]))
        # density slots first, rare slots after
        self.assertEqual(int(slot_class[1]), 1)
        self.assertEqual(int(slot_class[1 + 3 * (num_density - 1)]), 1)
        self.assertEqual(int(slot_class[1 + 3 * num_density]), 2)
        self.assertEqual(int(slot_class[0]), 0)                 # global is not a slot
        self.assertEqual([int(v) for v in tails[-num_tails:]], [1, 2, 3])

    def test_projected_mode_is_untouched(self) -> None:
        """Default path must not gain parameters or change behaviour."""
        model = _build("projected")
        self.assertFalse(hasattr(model.meta_classifier, "bag_agg_encoder"))
        x, y = _episode()
        with torch.no_grad():
            logits = model(x, y, torch.tensor([2, 5]))
        self.assertEqual(tuple(logits.shape), (2, 2))

    def test_context_adaptive_requires_weights(self) -> None:
        model = _build()
        x, _ = _episode()
        tokens = _context_tokens(model, x, 6)
        with self.assertRaises(ValueError):
            model.meta_classifier._projected_bag_tokens(tokens)


if __name__ == "__main__":
    unittest.main()
