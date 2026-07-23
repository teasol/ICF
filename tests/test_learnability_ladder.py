import unittest
from pathlib import Path

import torch

from src.datasets.synthetic_data import SyntheticEpisodeDataset
from src.modules.model_interface import ModelInterface
from src.utils.utils import merge_train_config


class FixedEpisodeBankTest(unittest.TestCase):
    def _dataset(self, **overrides):
        kwargs = {
            "episodes_per_epoch": 6,
            "seed": 101,
            "fixed_episode_count": 2,
            "generation_device": "cpu",
            "num_bags": 6,
            "num_cells": 10,
            "latent_dim": 4,
            "output_dim": 8,
            "mlp_hidden_dim": 8,
            "mlp_num_layers": 2,
        }
        kwargs.update(overrides)
        return SyntheticEpisodeDataset(**kwargs)

    def test_indices_repeat_within_fixed_bank(self):
        dataset = self._dataset()
        first = dataset[0]
        repeated = dataset[2]
        other = dataset[1]
        torch.testing.assert_close(first[0], repeated[0])
        torch.testing.assert_close(first[1], repeated[1])
        self.assertFalse(torch.equal(first[0], other[0]))

    def test_fixed_bank_requires_seed(self):
        with self.assertRaisesRegex(ValueError, "requires a fixed dataset seed"):
            self._dataset(seed=None)

    def test_fixed_bank_size_must_fit_epoch(self):
        with self.assertRaisesRegex(ValueError, "fixed_episode_count"):
            self._dataset(fixed_episode_count=7)


class FixedTrainingQueryTest(unittest.TestCase):
    def test_split_is_deterministic_and_keeps_every_class_in_context(self):
        interface = ModelInterface(
            model_src="src.models.baseline.BaseModel",
            input_dim=8,
            aggregator_num_slots=4,
            aggregator_num_density_slots=3,
            meta_hidden_dim=16,
            meta_num_heads=4,
            meta_num_set_layers=1,
            meta_relation_hidden_dim=16,
            meta_ridge_dim=4,
            training_targets_per_episode=4,
            fixed_training_queries=True,
        )
        labels = torch.tensor([1, 0, 1, 0, 1, 0, 1, 0])
        first = interface._sample_training_queries(labels)
        second = interface._sample_training_queries(labels)
        torch.testing.assert_close(first, second)
        context = torch.ones_like(labels, dtype=torch.bool)
        context[first] = False
        self.assertEqual(torch.unique(labels[context]).tolist(), [0, 1])


class ManifoldLadderTest(unittest.TestCase):
    def test_shared_nonlinear_uses_one_mapping_across_episode_rngs(self):
        dataset = FixedEpisodeBankTest()._dataset(
            manifold_mode="shared_nonlinear", manifold_seed=17
        )
        generator = dataset.episode_generator
        z = torch.randn(2, 3, 4)
        first = generator._map_episode_manifold(
            z, torch.Generator().manual_seed(1), torch.device("cpu")
        )
        second = generator._map_episode_manifold(
            z, torch.Generator().manual_seed(2), torch.device("cpu")
        )
        torch.testing.assert_close(first, second)

    def test_orthogonal_manifold_preserves_pairwise_distances(self):
        dataset = FixedEpisodeBankTest()._dataset(manifold_mode="orthogonal")
        generator = dataset.episode_generator
        z = torch.randn(1, 5, 4)
        mapped = generator._map_episode_manifold(
            z, torch.Generator().manual_seed(3), torch.device("cpu")
        )
        torch.testing.assert_close(torch.cdist(z[0], z[0]), torch.cdist(mapped[0], mapped[0]))

    def test_binary_diagnostics_match_known_predictions(self):
        logits = torch.tensor([[2.0, 0.0], [0.0, 2.0], [0.0, 2.0], [2.0, 0.0]])
        targets = torch.tensor([0, 1, 0, 1])
        terms = ModelInterface._binary_query_diagnostics(logits, targets)
        self.assertAlmostEqual(terms["query_positive_fraction"].item(), 0.5)
        self.assertAlmostEqual(terms["majority_accuracy"].item(), 0.5)
        self.assertAlmostEqual(terms["empirical_prior_ce"].item(), 0.693147, places=5)
        self.assertAlmostEqual(terms["balanced_accuracy"].item(), 0.5)
        self.assertAlmostEqual(terms["auroc"].item(), 0.5)


class OracleAbundanceTest(unittest.TestCase):
    def test_query_labels_are_not_used_to_fit_oracle(self):
        abundance = torch.tensor([0.1, 0.8, 0.2, 0.7, 0.3, 0.6])
        labels = torch.tensor([0, 1, 0, 1, 0, 1])
        query = torch.tensor([4, 5])
        expected = ModelInterface._fit_oracle_abundance_logits(
            abundance, labels, query
        )
        changed_query_labels = labels.clone()
        changed_query_labels[query] = 1 - changed_query_labels[query]
        actual = ModelInterface._fit_oracle_abundance_logits(
            abundance, changed_query_labels, query
        )
        torch.testing.assert_close(expected, actual)
        self.assertFalse(actual.requires_grad)

    def test_oracle_metrics_are_finite_and_separable(self):
        abundance = torch.tensor([0.1, 0.8, 0.2, 0.7, 0.15, 0.75])
        labels = torch.tensor([0, 1, 0, 1, 0, 1])
        query = torch.tensor([4, 5])
        metrics = ModelInterface._oracle_abundance_diagnostics(
            abundance, labels, query, torch.tensor(0.5)
        )
        self.assertTrue(all(torch.isfinite(value) for value in metrics.values()))
        self.assertEqual(metrics["oracle_abundance_accuracy"].item(), 1.0)
        self.assertEqual(metrics["oracle_abundance_auroc"].item(), 1.0)
        self.assertEqual(metrics["oracle_model_auroc_gap"].item(), 0.5)


class NuisanceResolvedConfigTest(unittest.TestCase):
    nuisance = {
        "d0": ("donor_shift_scale", 0.35),
        "d1": ("donor_component_shift_scale", 0.12),
        "d2": ("donor_mixture_logit_scale", 0.65),
        "d3": ("shared_component_base_logit_scale", 0.70),
        "d4": ("donor_shared_component_logit_scale", 0.70),
    }

    def test_d_stages_enable_exactly_one_selected_nuisance(self):
        root = Path(__file__).resolve().parents[1]
        configs = {
            stage: merge_train_config(
                root / "configs" / f"train_learnability_{stage}.yaml"
            )
            for stage in self.nuisance
        }
        base = merge_train_config(
            root / "configs" / "train_learnability_d_base.yaml"
        )
        nuisance_keys = {
            key for key, _ in self.nuisance.values() if key is not None
        }
        for stage, (enabled_key, enabled_value) in self.nuisance.items():
            kwargs = configs[stage]["data"]["dataset_kwargs"]
            self.assertTrue(kwargs["return_oracle_diagnostics"])
            for key in nuisance_keys:
                expected = enabled_value if key == enabled_key else 0.0
                self.assertEqual(kwargs[key], expected)
            comparable = {
                **configs[stage],
                "data": {
                    **configs[stage]["data"],
                    "dataset_kwargs": {
                        **kwargs,
                        **{
                            key: base["data"]["dataset_kwargs"][key]
                            for key in nuisance_keys
                        },
                    },
                },
            }
            self.assertEqual(comparable, base)


class A6000DDPResolvedConfigTest(unittest.TestCase):
    stages = (
        "a", "b", "c", "c0", "c1", "c2", "c3", "c4", "c5", "c4_n", "c4_d",
        "d0", "d1", "d2", "d3", "d4",
    )

    def test_requested_stages_use_four_rank_effective_batch_eight(self):
        root = Path(__file__).resolve().parents[1]
        for stage in self.stages:
            with self.subTest(stage=stage):
                config = merge_train_config(
                    root / "configs" / f"train_learnability_{stage}.yaml"
                )
                self.assertEqual(config["trainer"]["devices"], 4)
                self.assertEqual(config["trainer"]["strategy"], "ddp")
                self.assertEqual(config["data"]["episode_batch_size"], 2)
                self.assertEqual(
                    config["data"]["dataset_kwargs"]["shape_group_size"], 2
                )
                checkpoint = config["callbacks"]["checkpoint"]
                self.assertEqual(checkpoint["save_top_k"], 1)
                self.assertFalse(checkpoint["save_last"])
                self.assertEqual(checkpoint["selection_start_epoch"], 5)

    def test_frozen_protocol_matches_resolved_configs(self):
        root = Path(__file__).resolve().parents[1]
        import yaml

        protocol = yaml.safe_load(
            (root / "experiments" / "v18_learnability_protocol.yaml").read_text()
        )
        self.assertEqual(protocol["training"]["seeds"], [42, 43, 44])
        self.assertEqual(protocol["training"]["effective_episode_batch"], 8)
        self.assertEqual(protocol["evaluation"]["frozen_evaluation_episodes"], 8192)
        for display_stage, stage_spec in protocol["stages"].items():
            with self.subTest(stage=display_stage):
                config = merge_train_config(root / stage_spec["config"])
                self.assertEqual(config["trainer"]["max_epochs"], stage_spec["max_epochs"])
                self.assertEqual(
                    config["callbacks"]["checkpoint"]["monitor"],
                    stage_spec.get("checkpoint_metric", "val_ce_loss"),
                )
                bank = protocol["banks"][stage_spec["evaluation_bank"]]
                if bank.get("independent_evaluation") is False:
                    continue
                self.assertEqual(
                    config["data"]["test_dataset_kwargs"]["episodes_per_epoch"],
                    bank["evaluation_episodes"],
                )
                self.assertEqual(
                    config["data"]["test_dataset_kwargs"]["seed"],
                    bank["evaluation_seed"],
                )


if __name__ == "__main__":
    unittest.main()
