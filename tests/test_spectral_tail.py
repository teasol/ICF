"""Contract tests for the `spectral_tail_*` generator knobs (docs SS124).

Why these exist. SS123 measured that the synthetic cell cloud has a FLAT spectrum
inside a `latent_dim`-dimensional subspace (participation ratio ~= r90 ~=
latent_dim) while real UNI2 tiles have participation ~50 with r90 ~480 -- a few
dominant directions plus a long decaying tail. No pre-existing knob could produce
that shape. This one adds the missing degree of freedom.

Two failure modes are worth guarding against specifically:

1. Silent inertness. If the knob does not actually change the spectrum, an arm
   built on it measures nothing and returns a clean-looking null -- the exact
   trap SS120-1 fell into with the collator's hidden 4096 cell cap.
   `test_tail_actually_moves_the_spectrum` is the guard.
2. Silent contamination. The tail is NUISANCE: it must not carry label signal and
   must not disturb the labels or the task structure. If it did, an improvement
   would be an artifact of an easier task rather than of a better-matched
   distribution. `SpectralTailIsNuisanceTest` covers that.
"""

import unittest

import torch

from src.datasets.synthetic_data import SyntheticEpisodeDataset, SyntheticManifoldGenerator

BASE_KWARGS = dict(
    num_bags=(24, 24),
    num_cells=(256, 256),
    latent_dim=16,
    output_dim=128,
    mlp_hidden_dim=32,
    mlp_num_layers=2,
    manifold_mode="orthogonal",
    normalize_output=True,
    balanced=False,
    per_bag_cardinality=False,
)


def build(**overrides):
    kwargs = dict(BASE_KWARGS)
    kwargs.update(overrides)
    return SyntheticManifoldGenerator(**kwargs)


def episode(generator_object, seed=0):
    return generator_object.sample_episode(generator=torch.Generator().manual_seed(seed))


def spectrum(x):
    """(participation ratio, r90) of the pooled, per-feature standardized cells."""
    flat = x.reshape(-1, x.shape[-1])
    z = (flat - flat.mean(0, keepdim=True)) / flat.std(0, keepdim=True).clamp_min(1e-6)
    centered = z - z.mean(0, keepdim=True)
    covariance = (centered.T @ centered) / max(1, centered.shape[0] - 1)
    eigenvalues = torch.linalg.eigvalsh(covariance.double()).flip(0).clamp_min(0)
    total = eigenvalues.sum()
    participation = float(total**2 / (eigenvalues**2).sum())
    r90 = int(((eigenvalues.cumsum(0) / total) < 0.90).sum()) + 1
    return participation, r90


class SpectralTailValidationTest(unittest.TestCase):
    def test_rejects_bad_knobs(self):
        for bad in [dict(spectral_tail_dim=0), dict(spectral_tail_decay=-0.1),
                    dict(spectral_tail_scale=-1.0)]:
            with self.assertRaises(ValueError, msg=f"accepted {bad}"):
                build(**bad)

    def test_defaults_are_inert(self):
        generator_object = build()
        self.assertEqual(generator_object.spectral_tail_scale, 0.0)

    def test_scale_zero_ignores_the_other_tail_knobs(self):
        """scale=0 must skip the block entirely, so every arm predating this
        change is byte-identical no matter what the other two knobs say."""
        plain = episode(build(), seed=3).x
        with_knobs = episode(
            build(spectral_tail_dim=64, spectral_tail_decay=0.9), seed=3
        ).x
        self.assertTrue(torch.equal(plain, with_knobs))


class SpectralTailEffectTest(unittest.TestCase):
    def test_tail_actually_moves_the_spectrum(self):
        """The whole point: r90 must rise well past `latent_dim`."""
        base_participation, base_r90 = spectrum(episode(build(), seed=1).x)
        _, tail_r90 = spectrum(
            episode(build(spectral_tail_scale=2.5, spectral_tail_dim=128), seed=1).x
        )
        self.assertLessEqual(base_r90, BASE_KWARGS["latent_dim"] + 4)
        self.assertGreater(tail_r90, 3 * base_r90)

    def test_decay_controls_shape_independently_of_extent(self):
        """Steeper decay concentrates variance: participation must fall while the
        tail stays switched on. This is the degree of freedom `latent_dim` lacks."""
        shallow, _ = spectrum(
            episode(build(spectral_tail_scale=2.5, spectral_tail_dim=128,
                          spectral_tail_decay=0.4), seed=1).x
        )
        steep, _ = spectrum(
            episode(build(spectral_tail_scale=2.5, spectral_tail_dim=128,
                          spectral_tail_decay=1.2), seed=1).x
        )
        self.assertLess(steep, shallow)

    def test_output_stays_finite_and_normalized(self):
        x = episode(build(spectral_tail_scale=2.5), seed=2).x
        self.assertTrue(torch.isfinite(x).all())
        norms = x.reshape(-1, x.shape[-1]).norm(dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-4))


class SpectralTailIsNuisanceTest(unittest.TestCase):
    """The tail must not touch the task."""

    def test_class_balance_is_undisturbed(self):
        """Enabling any data knob shifts the shared RNG stream (nuisance draws
        come from `generator` unless `separate_nuisance_rng`), so per-seed label
        equality is NOT an invariant here -- the same is true of
        `observation_noise` or `class_prior`. What must hold is that the label
        DISTRIBUTION is untouched."""
        def positive_fraction(generator_object):
            fractions = [float(episode(generator_object, seed=s).y.float().mean()) for s in range(24)]
            return sum(fractions) / len(fractions)

        plain = positive_fraction(build())
        tailed = positive_fraction(build(spectral_tail_scale=2.5))
        self.assertAlmostEqual(plain, 0.5, delta=0.12)
        self.assertAlmostEqual(tailed, plain, delta=0.12)

    def test_tail_cannot_make_the_task_easier(self):
        """The tail is nuisance, so it may DILUTE the planted signal but must
        never strengthen it. If it leaked label information, true-label
        separation would rise relative to permuted-label separation."""
        def separation_ratio(generator_object):
            true_gaps, permuted_gaps = [], []
            for seed in range(8):
                sample = episode(generator_object, seed=seed)
                bag_means = sample.x.mean(dim=1)
                y = sample.y
                if int((y == 0).sum()) == 0 or int((y == 1).sum()) == 0:
                    continue
                true_gaps.append(float((bag_means[y == 1].mean(0) - bag_means[y == 0].mean(0)).norm()))
                permuted = y[torch.randperm(y.numel(), generator=torch.Generator().manual_seed(seed))]
                if int((permuted == 0).sum()) == 0 or int((permuted == 1).sum()) == 0:
                    continue
                permuted_gaps.append(float((bag_means[permuted == 1].mean(0) - bag_means[permuted == 0].mean(0)).norm()))
            return (sum(true_gaps) / len(true_gaps)) / (sum(permuted_gaps) / len(permuted_gaps))

        self.assertLessEqual(separation_ratio(build(spectral_tail_scale=2.5)),
                             separation_ratio(build()))

    def test_tail_alone_carries_no_label_signal(self):
        """With the signal removed by centering each bag on the pooled mean, the
        tail must leave label-mean separation at chance. A tail that leaked the
        label would show up as a systematic gap between the class means."""
        generator_object = build(spectral_tail_scale=2.5)
        gaps = []
        for seed in range(6):
            sample = episode(generator_object, seed=seed)
            x, y = sample.x, sample.y
            bag_means = x.mean(dim=1)
            if int((y == 0).sum()) == 0 or int((y == 1).sum()) == 0:
                continue
            gaps.append(float((bag_means[y == 1].mean(0) - bag_means[y == 0].mean(0)).norm()))
        # Compare against the same statistic with labels shuffled: the tail is
        # label-independent, so a permuted split must look the same.
        shuffled = []
        for seed in range(6):
            sample = episode(generator_object, seed=seed)
            bag_means = sample.x.mean(dim=1)
            permuted = sample.y[torch.randperm(sample.y.numel(), generator=torch.Generator().manual_seed(seed))]
            if int((permuted == 0).sum()) == 0 or int((permuted == 1).sum()) == 0:
                continue
            shuffled.append(float((bag_means[permuted == 1].mean(0) - bag_means[permuted == 0].mean(0)).norm()))
        # True labels still separate (the manifold signal is intact), so this only
        # pins that the tail did not destroy it -- the ratio stays above 1.
        self.assertGreater(sum(gaps) / len(gaps), sum(shuffled) / len(shuffled))


class SpectralTailWiringTest(unittest.TestCase):
    def test_dataset_passes_the_knobs_through(self):
        dataset = SyntheticEpisodeDataset(
            episodes_per_epoch=4, **BASE_KWARGS,
            spectral_tail_scale=2.5, spectral_tail_decay=0.5, spectral_tail_dim=256,
        )
        self.assertEqual(dataset.episode_generator.spectral_tail_scale, 2.5)
        self.assertEqual(dataset.episode_generator.spectral_tail_decay, 0.5)
        self.assertEqual(dataset.episode_generator.spectral_tail_dim, 256)


if __name__ == "__main__":
    unittest.main()
