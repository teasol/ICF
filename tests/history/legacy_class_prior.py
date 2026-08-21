"""Contract tests for the `class_prior` episode knob (docs SS116, v90).

`class_prior` is the first knob that lets a training episode be class-imbalanced.
Two things make it worth pinning with tests rather than trusting by inspection:

1. `ModelInterface._sample_query_index` RAISES on a single-class episode
   (`Every episode must contain all classes`). A prior near either end of its
   range can produce one, and at 1024 episodes/epoch x 50 epochs a "rare" crash
   is a certainty, not a risk. `_repair_missing_classes` is the guard and
   `ClassPriorGuardTest` is what proves it works at priors far more extreme than
   the arm actually uses.
2. The knob must be INERT when unset. v90 is judged seed-paired against v83, so
   if merely adding the parameter shifted the default label stream, the whole
   comparison would be measuring the refactor instead of the prior. That is what
   `test_default_is_bit_identical_to_the_old_stream` pins.
"""

import unittest

import torch

from src.datasets.synthetic_data import (
    SyntheticEpisodeDataset,
    SyntheticManifoldGenerator,
)

BASE_KWARGS = dict(
    num_bags=(60, 100),
    num_cells=(64, 256),
    latent_dim=8,
    output_dim=32,
    mlp_hidden_dim=16,
    mlp_num_layers=2,
    balanced=False,
)


def build(**overrides):
    kwargs = dict(BASE_KWARGS)
    kwargs.update(overrides)
    return SyntheticManifoldGenerator(**kwargs)


def draw_labels(dataset, num_bags, seed):
    generator = torch.Generator().manual_seed(seed)
    return dataset._sample_labels(num_bags, generator, torch.device("cpu"))


class ClassPriorValidationTest(unittest.TestCase):
    def test_rejects_out_of_range_priors(self):
        for bad in [0.0, 1.0, -0.1, 1.5, (0.5, 0.2), (0.0, 0.8), (0.2, 1.0)]:
            with self.assertRaises(ValueError, msg=f"accepted {bad}"):
                build(class_prior=bad)

    def test_rejects_class_prior_with_balanced(self):
        """`balanced=True` forces an exact 50/50 split, which would silently
        throw the prior away -- that must be loud, not silent."""
        with self.assertRaises(ValueError):
            build(balanced=True, class_prior=(0.2, 0.8))

    def test_accepts_scalar_and_range(self):
        self.assertEqual(build(class_prior=0.3).class_prior, (0.3, 0.3))
        self.assertEqual(build(class_prior=(0.15, 0.85)).class_prior, (0.15, 0.85))


class ClassPriorDefaultTest(unittest.TestCase):
    def test_default_is_none(self):
        self.assertIsNone(build().class_prior)

    def test_default_is_bit_identical_to_the_old_stream(self):
        """v90 is judged against v83, so the unset path must be byte-identical
        to the Bernoulli(0.5) `torch.randint` draw that v83 trained on."""
        dataset = build()
        for seed in range(8):
            expected = torch.randint(
                0, 2, (80,), dtype=torch.long, generator=torch.Generator().manual_seed(seed)
            )
            self.assertTrue(torch.equal(draw_labels(dataset, 80, seed), expected))


class ClassPriorDistributionTest(unittest.TestCase):
    def test_fixed_prior_is_matched_on_average(self):
        dataset = build(class_prior=0.2)
        fractions = [float(draw_labels(dataset, 300, s).float().mean()) for s in range(200)]
        self.assertAlmostEqual(sum(fractions) / len(fractions), 0.2, delta=0.02)

    def test_prior_is_drawn_once_per_episode_not_per_bag(self):
        """A per-BAG draw over [0.15,0.85] would concentrate every episode's
        positive fraction near 0.5; a per-EPISODE draw spreads them across the
        whole range. The spread is the entire point of the knob, so pin it."""
        dataset = build(class_prior=(0.15, 0.85))
        fractions = [float(draw_labels(dataset, 300, s).float().mean()) for s in range(300)]
        self.assertGreater(max(fractions) - min(fractions), 0.5)
        # Per-bag mixing would give sd ~ 0.03 at 300 bags; per-episode gives the
        # sd of U(0.15,0.85), which is 0.7/sqrt(12) = 0.202.
        mean = sum(fractions) / len(fractions)
        sd = (sum((f - mean) ** 2 for f in fractions) / (len(fractions) - 1)) ** 0.5
        self.assertGreater(sd, 0.15)

    def test_covers_the_measured_real_range(self):
        """Real tasks run 0.178 (LUAD STK11) to 0.780 (CCRCC VHL) -- docs SS115."""
        dataset = build(class_prior=(0.15, 0.85))
        fractions = [float(draw_labels(dataset, 300, s).float().mean()) for s in range(300)]
        self.assertLess(min(fractions), 0.178)
        self.assertGreater(max(fractions), 0.780)


class ClassPriorGuardTest(unittest.TestCase):
    """The guard that keeps `_sample_query_index` from raising mid-run."""

    def test_never_emits_a_single_class_episode_at_extreme_priors(self):
        for prior in (0.01, 0.99):
            dataset = build(class_prior=prior)
            for seed in range(400):
                labels = draw_labels(dataset, 8, seed)
                self.assertGreater(int((labels == 0).sum()), 0, f"prior={prior} seed={seed}")
                self.assertGreater(int((labels == 1).sum()), 0, f"prior={prior} seed={seed}")

    def test_guard_is_a_no_op_when_both_classes_are_present(self):
        labels = torch.tensor([0, 1, 1, 0, 1])
        repaired = SyntheticManifoldGenerator._repair_missing_classes(
            labels.clone(), torch.Generator().manual_seed(0), torch.device("cpu")
        )
        self.assertTrue(torch.equal(labels, repaired))


class ClassPriorEpisodeTest(unittest.TestCase):
    """The knob must survive the real paths, not just `_sample_labels`."""

    def test_full_episode_builds_and_stays_finite(self):
        generator_object = build(class_prior=(0.15, 0.85))
        episode = generator_object.sample_episode(
            generator=torch.Generator().manual_seed(3)
        )
        x, y = episode.x, episode.y
        self.assertTrue(torch.isfinite(x).all())
        self.assertEqual(int(y.shape[0]), int(x.shape[0]))
        self.assertGreater(int((y == 0).sum()), 0)
        self.assertGreater(int((y == 1).sum()), 0)

    def test_dataset_passes_the_knob_through_to_the_generator(self):
        """The config sets this under `dataset_kwargs`, so it reaches the
        generator via `**generator_kwargs` -- pin that wiring."""
        dataset = SyntheticEpisodeDataset(
            episodes_per_epoch=4, **BASE_KWARGS, class_prior=(0.15, 0.85)
        )
        self.assertEqual(dataset.episode_generator.class_prior, (0.15, 0.85))


if __name__ == "__main__":
    unittest.main()
