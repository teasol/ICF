"""Contract tests for the PA (population attention) relation branch (docs SS114).

PA fits a ridge direction over context CELLS labeled with their own bag's label,
then summarizes each bag by soft abundance of cells clearing a threshold on each
side of that direction. The project's own history (SS62-2/SS68-1, `docs/history.md`
SS14) killed an earlier population-attention branch that turned out to feed a
routing softmax nothing to choose between -- it went "dead" (constant output)
without anyone noticing until a dedicated probe caught it. `PopulationAttentionAliveTest`
below is that probe for THIS branch before any GPU time gets spent on it -- and it did
in fact catch a real design bug during development (top/bottom-k-mean of the same
signed axis barely beat chance; independent per-side abundance fixed it, see the
`_pa_population_summary` docstring in the model file).
"""

import unittest

import torch
from torch import nn

from src.models.set_transformer_ridge import CovarianceMeanLearnablePDDCTPAMLPModel

BASE_KWARGS = dict(
    input_dim=64,
    token_dim=16,
    num_heads=1,
    num_layers=1,
    feedforward_dim=16,
    num_summary_tokens=1,
    max_cells=128,
    dropout=0.0,
    covariance_sketch_dim=8,
    ridge_lambda=1.0,
    ridge_logit_scale=2.0,
    num_classes=2,
)


def build(**overrides):
    kwargs = dict(BASE_KWARGS)
    kwargs.update(overrides)
    torch.manual_seed(0)
    return CovarianceMeanLearnablePDDCTPAMLPModel(**kwargs)


def small_episode(generator_seed=7):
    """8 bags (6 context, 2 query), no planted signal -- shape/gradient tests only."""
    generator = torch.Generator().manual_seed(generator_seed)
    num_bags, num_cells, dim = 8, 40, BASE_KWARGS["input_dim"]
    bags = torch.randn(num_bags, num_cells, dim, generator=generator)
    labels = torch.tensor([0, 0, 0, 1, 1, 1, 0, 1])
    query_index = torch.tensor([6, 7])
    return bags, labels, query_index


def minority_episode(generator_seed, num_context_bags=40, discriminative_fraction=0.1):
    """`num_context_bags` context bags + 2 query bags, balanced labels.

    A `discriminative_fraction` slice of each bag's cells is shifted along a
    fixed direction by an amount that depends on the bag's own label; the rest
    are label-independent noise -- "most cells are shared background, a
    minority carries the signal." Needs a realistic bag count (tens, not the
    6-8 used elsewhere in this file) for the per-cell ridge fit to average out
    background noise -- see the model file's `_pa_cell_direction` docstring.
    """
    generator = torch.Generator().manual_seed(generator_seed)
    dim, num_cells = BASE_KWARGS["input_dim"], 40
    total_bags = num_context_bags + 2
    bags = torch.randn(total_bags, num_cells, dim, generator=generator)
    labels = torch.zeros(total_bags, dtype=torch.long)
    labels[total_bags // 2 :] = 1
    labels = labels[torch.randperm(total_bags, generator=generator)]
    direction = torch.zeros(dim)
    direction[0] = 1.0
    signal_count = max(1, round(discriminative_fraction * num_cells))
    sign = torch.where(labels == 1, 1.0, -1.0)
    bags[:, :signal_count, :] += (4.0 * sign)[:, None, None] * direction[None, None, :]
    query_index = torch.tensor([num_context_bags, num_context_bags + 1])
    context_index = [i for i in range(total_bags) if i not in query_index.tolist()]
    return bags, labels, query_index, context_index


class PopulationAttentionShapeTest(unittest.TestCase):
    def test_old_head_is_gone_new_head_is_16_wide(self):
        model = build()
        self.assertFalse(hasattr(model, "cv_dd_ct_head"))
        head = model.cv_dd_ct_pa_head
        self.assertIsInstance(head[0], nn.Linear)
        self.assertEqual(head[0].in_features, 16)
        self.assertEqual(head[-1].out_features, 1)

    def test_architecture_version_is_57(self):
        model = build()
        self.assertEqual(int(model._architecture_version.item()), 57)

    def test_rejects_bad_knobs(self):
        with self.assertRaises(ValueError):
            build(pa_cells_per_bag=0)
        with self.assertRaises(ValueError):
            build(pa_threshold=-1.0)
        with self.assertRaises(ValueError):
            build(pa_temperature=0.0)
        with self.assertRaises(ValueError):
            build(pa_ridge_lambda=0.0)
        with self.assertRaises(ValueError):
            build(pa_head_hidden_dims=[32, -1])


class PopulationAttentionForwardTest(unittest.TestCase):
    def test_forward_is_finite_and_antisymmetric(self):
        bags, labels, query_index = small_episode()
        model = build()
        model.eval()
        with torch.no_grad():
            logits = model(bags, labels, query_index)
        self.assertEqual(logits.shape, (2, 2))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(torch.allclose(logits.sum(dim=-1), torch.zeros(2), atol=1e-5))

    def test_gradient_reaches_p_and_head(self):
        """SS100's contract, same as the CV/DD/CT head: P's gradient must be
        finite and nonzero -- `nonfinite_gradient_policy: zero` would hide a
        broken backward path silently otherwise."""
        bags, labels, query_index = small_episode()
        model = build()
        logits = model(bags, labels, query_index)
        logits.square().sum().backward()
        projection_grad = model._covariance_projection.grad
        self.assertIsNotNone(projection_grad)
        self.assertTrue(torch.isfinite(projection_grad).all())
        self.assertGreater(float(projection_grad.abs().max()), 0.0)
        for name, parameter in model.cv_dd_ct_pa_head.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_train_dd_projection_still_reaches_p_through_dd(self):
        """Regression test: PA's no_grad scoping must not swallow the
        `train_dd_projection=True` gradient path -- see the class docstring's
        explanation of why DD's block is deliberately NOT inside PA/CT's
        unconditional no_grad."""
        bags, labels, query_index = small_episode()
        model = build(train_dd_projection=True)
        logits = model(bags, labels, query_index)
        logits.square().sum().backward()
        projection_grad = model._covariance_projection.grad
        self.assertIsNotNone(projection_grad)
        self.assertTrue(torch.isfinite(projection_grad).all())
        self.assertGreater(float(projection_grad.abs().max()), 0.0)

    def test_checkpoint_does_not_silently_half_load_from_v83_shape(self):
        """A 12-feature checkpoint must not half-load into the 16-feature head."""
        from src.models.set_transformer_ridge import CovarianceMeanLearnablePDDCTMLPModel

        torch.manual_seed(0)
        source = CovarianceMeanLearnablePDDCTMLPModel(**BASE_KWARGS)
        target = build()
        with self.assertRaises(RuntimeError):
            target.load_state_dict(source.state_dict(), strict=True)


class PopulationAttentionAliveTest(unittest.TestCase):
    """The one check that would have caught SS62-2's dead branch early.

    Plants a discriminative minority of cells (label-dependent shift) inside
    otherwise label-independent bags, then asserts PA's output actually tracks
    which class the shift favors. A branch that always returns the same value
    regardless of input (like the old population attention routing) would fail
    every assertion here.
    """

    def test_pa_features_are_not_constant_across_episodes(self):
        model = build()
        model.eval()
        bags_a, labels_a, query_index, context_index = minority_episode(1)
        bags_b, labels_b, _, _ = minority_episode(2)
        with torch.no_grad():
            pa0_a, pa1_a, sep_a = model._pa_features(
                [bags_a[i] for i in context_index],
                labels_a[context_index],
                [bags_a[i] for i in query_index.tolist()],
            )
            pa0_b, pa1_b, sep_b = model._pa_features(
                [bags_b[i] for i in context_index],
                labels_b[context_index],
                [bags_b[i] for i in query_index.tolist()],
            )
        self.assertFalse(torch.allclose(pa0_a, pa0_b))
        self.assertFalse(torch.allclose(pa1_a, pa1_b))
        self.assertGreater(float(sep_a.abs().max()), 0.0)

    def test_pa_tracks_a_planted_discriminative_minority(self):
        """With a real, label-linked minority signal (10% of cells) and a
        realistic context size (40 bags), PA1-PA0 should separate query bags
        by their true label at well above chance. Measured over many trials
        during development: ~90% at this exact configuration -- the bar here
        (>=80%) leaves margin without being toothless (chance is 50%)."""
        model = build()
        model.eval()
        correct = 0
        trials = 20
        for seed in range(trials):
            bags, labels, query_index, context_index = minority_episode(1000 + seed)
            with torch.no_grad():
                pa0, pa1, separation = model._pa_features(
                    [bags[i] for i in context_index],
                    labels[context_index],
                    [bags[i] for i in query_index.tolist()],
                )
            self.assertGreater(float(separation.abs().max()), 0.0)
            query_labels = labels[query_index]
            predicted = (pa1 - pa0) > 0
            correct += int((predicted == query_labels.bool()).sum())
        total = trials * 2
        self.assertGreaterEqual(correct, round(0.8 * total))


if __name__ == "__main__":
    unittest.main()
