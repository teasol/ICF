"""Exhaustive contract tests for YAML configuration parsing and TrainingFreeConfig.

Guards against 11 distinct YAML & Python configuration pitfalls:
1. Scientific notation string trap (e.g. "1e-3", 2e-5 coerced to float)
2. Default-value masking trap (anti-coincidence non-default override test)
3. YAML 1.1 Norway problem (bool vs string coercion safety)
4. Explicit null / None preservation (not reverting to default)
5. Silent typo and unknown key rejection with typo suggestion
6. Int / float coercion and fractional integer rejection
7. Mutable list to immutable hashable tuple conversion
8. Polymorphic field type preservation (int, float, str, None)
9. Enum and domain range validation (out-of-bounds, invalid choices)
10. Immutability & hashability contract
11. Lossless round-trip serialization and v120 baseline fidelity
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
from pathlib import Path
import unittest

from src.models.config import (
    TrainingFreeConfig,
    from_dict,
    from_yaml,
    to_dict,
    to_yaml,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


class TestYAMLConfigContract(unittest.TestCase):
    """Exhaustive contract tests for configuration loading and serialization."""

    # --------------------------------------------------------------------------
    # Pitfall 1: Scientific Notation String Trap
    # --------------------------------------------------------------------------
    def test_scientific_notation_float_coercion(self) -> None:
        """YAML scientific notation strings (e.g. '1e-3', '2e-5') must parse as float."""
        yaml_content = """
        basis:
          ridge_lambda: 1e-3
          ridge_scale: 2.5e-1
        branches:
          bd:
            eps: 1e-6
            lambda: 1e-4
          ct:
            eps: 1.0e-5
            temperature: 5e-1
          dd:
            eps: 2e-6
            shrinkage: 1.5e-1
        """
        cfg = from_yaml(yaml_content)

        self.assertIsInstance(cfg.ridge_lambda, float)
        self.assertEqual(cfg.ridge_lambda, 0.001)

        self.assertIsInstance(cfg.ridge_scale, float)
        self.assertEqual(cfg.ridge_scale, 0.25)

        self.assertIsInstance(cfg.bd_eps, float)
        self.assertEqual(cfg.bd_eps, 1e-6)

        self.assertIsInstance(cfg.bd_lambda, float)
        self.assertEqual(cfg.bd_lambda, 1e-4)

        self.assertIsInstance(cfg.ct_eps, float)
        self.assertEqual(cfg.ct_eps, 1e-5)

        self.assertIsInstance(cfg.ct_temperature, float)
        self.assertEqual(cfg.ct_temperature, 0.5)

        self.assertIsInstance(cfg.dd_eps, float)
        self.assertEqual(cfg.dd_eps, 2e-6)

        self.assertIsInstance(cfg.dd_shrinkage, float)
        self.assertEqual(cfg.dd_shrinkage, 0.15)

    # --------------------------------------------------------------------------
    # Pitfall 2: Default-Value Masking Trap (Anti-Coincidence Override Test)
    # --------------------------------------------------------------------------
    def test_anti_coincidence_all_non_default_values(self) -> None:
        """Every field must be tested with a strictly non-default value to verify it applies."""
        default_cfg = TrainingFreeConfig()

        non_defaults = {
            "sketch_dim": 127,
            "ridge_lambda": 3.1415,
            "ridge_scale": 0.5,
            "cv_blocks": "cov+mean",
            "weight_cv": 0.777,
            "weight_ct": 0.888,
            "ct_readout": "prototype",
            "ct_pca_dim": 16,
            "ct_num_tokens": 128,
            "ct_cells_per_bag": 32,
            "ct_cells_fraction": 0.25,
            "ct_cells_min": 8,
            "ct_cells_scale": "median",
            "ct_sampling": "random",
            "ct_sampling_seed": 42,
            "ct_distance_kernel": "cosine",
            "ct_tokenizer": "kmeans_plusplus",
            "ct_bisect_iterations": 4,
            "ct_bisect_power_iterations": 5,
            "ct_tree_reduction": "mean",
            "ct_hdbscan_min_cluster_size": 128,
            "ct_hdbscan_min_cluster_fraction": 0.05,
            "ct_hdbscan_min_samples": 16,
            "ct_hdbscan_cluster_selection_method": "eom",
            "ct_hdbscan_build_algo": "kdtree",
            "ct_hdbscan_allow_single_cluster": True,
            "ct_dbscan_eps": 0.35,
            "ct_dbscan_min_samples": 8,
            "ct_temperature": 0.25,
            "ct_eps": 1e-4,
            "ct_kmeans_iterations": 2,
            "ct_kmeans_max_iterations": 12,
            "ct_kmeans_tolerance": 1e-3,
            "ct_kmeans_seed": 7,
            "weight_bm": 0.555,
            "bm_dim": 16,
            "bm_lambda": 2.5,
            "weight_bd": 0.444,
            "bd_dim": 64,
            "bd_metric": "trace",
            "bd_lambda": 0.5,
            "bd_separation_floor": 2.0,
            "bd_eps": 1e-5,
            "bd_readout": "ridge",
            "weight_qa": 0.999,
            "qa_dim": 16,
            "qa_quantiles": (0.10, 0.20, 0.80, 0.90),
            "qa_lambda": 0.25,
            "weight_ds": 0.666,
            "ds_dim": 16,
            "ds_lambda": 0.75,
            "ds_temperature": 2.0,
            "ds_tokens": 128,
            "weight_dd": 0.333,
            "dd_shrinkage": 0.5,
            "dd_eps": 1e-5,
            "dd_readout": "distance",
            "dd_separation_floor": 2.0,
            "weight_de": 0.222,
            "de_dim": 16,
            "de_topk_fraction": 0.10,
            "de_topk_min": 2,
            "de_topk_max": 32,
            "de_lambda": 0.5,
            "weight_sw": 0.111,
            "sw_dim": 16,
            "sw_num_slices": 16,
            "sw_num_quantiles": 16,
            "sw_lambda": 0.5,
            "weight_lr": 0.123,
            "lr_dim": 16,
            "lr_lambda": 0.5,
            "lr_tau": 2.5,
            "lr_topk_fraction": 0.10,
            "lr_topk_min": 2,
            "lr_topk_max": 32,
            "lr_patches_per_ctx": 32,
            "krr_kernel": "rbf",
            "krr_gamma": 0.05,
            "krr_degree": 3,
            "krr_coef0": 2.0,
            "aggregation": "linear",
            "loo_gamma": 3.0,
            "loo_floor": 0.60,
        }

        # 1. Verify that our test values are genuinely different from defaults
        for k, non_def_val in non_defaults.items():
            def_val = getattr(default_cfg, k)
            self.assertNotEqual(
                non_def_val,
                def_val,
                f"Test setup error: non-default value for '{k}' ({non_def_val!r}) matches default ({def_val!r})",
            )

        # 2. Parse dictionary into config
        parsed_cfg = from_dict(non_defaults)

        # 3. Assert EVERY single field was genuinely updated to the non-default value
        for k, expected_val in non_defaults.items():
            actual_val = getattr(parsed_cfg, k)
            self.assertEqual(
                actual_val,
                expected_val,
                f"Field '{k}' did not apply non-default value! Got {actual_val!r}, expected {expected_val!r}",
            )
            self.assertNotEqual(
                actual_val,
                getattr(default_cfg, k),
                f"Field '{k}' coincidentally retained its default value!",
            )

    # --------------------------------------------------------------------------
    # Pitfall 3: YAML 1.1 Norway Problem (Bool vs String)
    # --------------------------------------------------------------------------
    def test_norway_problem_strings(self) -> None:
        """String fields must reject booleans and preserve string identity."""
        # Unquoted 'off' or 'yes' in YAML 1.1 becomes boolean False/True
        bad_yaml = """
        branches:
          cv:
            blocks: off
        """
        with self.assertRaises(TypeError) as ctx:
            from_yaml(bad_yaml)
        self.assertIn("expects string, but received boolean", str(ctx.exception))

        # Properly quoted string must succeed
        good_yaml = """
        branches:
          cv:
            blocks: "offdiag"
        """
        cfg = from_yaml(good_yaml)
        self.assertEqual(cfg.cv_blocks, "offdiag")

    # --------------------------------------------------------------------------
    # Pitfall 4: Explicit null / None Preservation
    # --------------------------------------------------------------------------
    def test_explicit_null_preservation(self) -> None:
        """Explicitly writing null must NOT fall back to non-None default."""
        yaml_content = """
        branches:
          ct:
            pca_dim: null
            cells_fraction: null
            dbscan_eps: null
        """
        cfg = from_yaml(yaml_content)
        self.assertIsNone(cfg.ct_pca_dim, "Explicit null ct_pca_dim was overwritten by default 32!")
        self.assertIsNone(cfg.ct_cells_fraction)
        self.assertIsNone(cfg.ct_dbscan_eps)

    # --------------------------------------------------------------------------
    # Pitfall 5: Silent Typo and Unknown Key Rejection
    # --------------------------------------------------------------------------
    def test_silent_typo_and_unknown_key_rejection(self) -> None:
        """Typos in key names must raise ValueError with suggestion rather than being silently ignored."""
        typo_yaml = """
        basis:
          sketch_dim: 256
          ridge_lamda: 0.1
        """
        with self.assertRaises(ValueError) as ctx:
            from_yaml(typo_yaml, strict=True)
        err = str(ctx.exception)
        self.assertIn("Unknown config key 'ridge_lamda'", err)
        self.assertIn("Did you mean 'ridge_lambda'?", err)

    # --------------------------------------------------------------------------
    # Pitfall 6: Int / Float Coercion & Fractional Integer Rejection
    # --------------------------------------------------------------------------
    def test_int_float_coercion_and_rejection(self) -> None:
        """Float fields accept integers; integer fields reject fractional numbers."""
        # Float field given integer
        cfg = from_dict({"ridge_lambda": 2, "weight_cv": 1})
        self.assertIsInstance(cfg.ridge_lambda, float)
        self.assertEqual(cfg.ridge_lambda, 2.0)
        self.assertIsInstance(cfg.weight_cv, float)
        self.assertEqual(cfg.weight_cv, 1.0)

        # Integer field given integer float (256.0 -> 256)
        cfg_int = from_dict({"sketch_dim": 256.0})
        self.assertIsInstance(cfg_int.sketch_dim, int)
        self.assertEqual(cfg_int.sketch_dim, 256)

        # Integer field given fractional float (256.5 -> error)
        with self.assertRaises(ValueError):
            from_dict({"sketch_dim": 256.5})

        # Integer field given boolean True -> error
        with self.assertRaises(TypeError):
            from_dict({"sketch_dim": True})

    # --------------------------------------------------------------------------
    # Pitfall 7: Mutable List to Immutable Tuple Conversion
    # --------------------------------------------------------------------------
    def test_mutable_sequence_to_frozen_tuple(self) -> None:
        """YAML lists for tuple fields must be converted to hashable tuples."""
        yaml_content = """
        branches:
          qa:
            quantiles: [0.10, 0.20, 0.80, 0.90]
        """
        cfg = from_yaml(yaml_content)
        self.assertIsInstance(cfg.qa_quantiles, tuple)
        self.assertEqual(cfg.qa_quantiles, (0.10, 0.20, 0.80, 0.90))
        for q in cfg.qa_quantiles:
            self.assertIsInstance(q, float)

        # Immutability & hashability check
        h = hash(cfg)
        self.assertIsInstance(h, int)

    # --------------------------------------------------------------------------
    # Pitfall 8: Polymorphic Field Handling
    # --------------------------------------------------------------------------
    def test_polymorphic_field_handling(self) -> None:
        """Polymorphic fields (e.g. ct_abundance_cells_per_bag) preserve exact types."""
        # Int
        cfg1 = from_dict({"ct_abundance_cells_per_bag": 64})
        self.assertEqual(cfg1.ct_abundance_cells_per_bag, 64)
        self.assertIsInstance(cfg1.ct_abundance_cells_per_bag, int)

        # Float
        cfg2 = from_dict({"ct_abundance_cells_per_bag": 0.125})
        self.assertEqual(cfg2.ct_abundance_cells_per_bag, 0.125)
        self.assertIsInstance(cfg2.ct_abundance_cells_per_bag, float)

        # String
        cfg3 = from_dict({"ct_abundance_cells_per_bag": "match"})
        self.assertEqual(cfg3.ct_abundance_cells_per_bag, "match")
        self.assertIsInstance(cfg3.ct_abundance_cells_per_bag, str)

        # None
        cfg4 = from_dict({"ct_abundance_cells_per_bag": None})
        self.assertIsNone(cfg4.ct_abundance_cells_per_bag)

    # --------------------------------------------------------------------------
    # Pitfall 9: Enum and Domain Range Validation
    # --------------------------------------------------------------------------
    def test_enum_and_range_validation(self) -> None:
        """Invalid enum choices, negative dimensions, or unsorted quantiles must be rejected."""
        # Invalid aggregation
        with self.assertRaises(ValueError) as ctx:
            from_dict({"aggregation": "softmax"})
        self.assertIn("Invalid aggregation", str(ctx.exception))

        # Invalid cv_blocks
        with self.assertRaises(ValueError) as ctx:
            from_dict({"cv_blocks": "diag_only"})
        self.assertIn("Invalid cv_blocks", str(ctx.exception))

        # Negative sketch_dim
        with self.assertRaises(ValueError) as ctx:
            from_dict({"sketch_dim": -128})
        self.assertIn("sketch_dim must be positive", str(ctx.exception))

        # Negative weight
        with self.assertRaises(ValueError) as ctx:
            from_dict({"weight_cv": -1.0})
        self.assertIn("weight_cv must be non-negative", str(ctx.exception))

        # Unsorted quantiles
        with self.assertRaises(ValueError) as ctx:
            from_dict({"qa_quantiles": (0.90, 0.10)})
        self.assertIn("qa_quantiles must be strictly ascending", str(ctx.exception))

        # Out-of-bound quantiles
        with self.assertRaises(ValueError) as ctx:
            from_dict({"qa_quantiles": (0.10, 1.50)})
        self.assertIn("qa_quantiles elements must be in (0, 1)", str(ctx.exception))

    # --------------------------------------------------------------------------
    # Pitfall 10: Immutability Contract
    # --------------------------------------------------------------------------
    def test_immutability_contract(self) -> None:
        """TrainingFreeConfig must be frozen against accidental in-place mutation."""
        cfg = TrainingFreeConfig()
        with self.assertRaises(FrozenInstanceError):
            cfg.weight_cv = 2.0  # type: ignore[misc]

    # --------------------------------------------------------------------------
    # Pitfall 11: Lossless Round-Trip & v120 Baseline Fidelity
    # --------------------------------------------------------------------------
    def test_lossless_round_trip(self) -> None:
        """to_yaml -> from_yaml must reproduce the exact same object."""
        cfg = TrainingFreeConfig(
            sketch_dim=384,
            weight_qa=1.0,
            qa_dim=32,
            aggregation="trimmed_mean",
        )
        serialized = to_yaml(cfg)
        deserialized = from_yaml(serialized)
        self.assertEqual(cfg, deserialized)

    def test_v120_baseline_fidelity(self) -> None:
        """v120_active.yaml must accurately reproduce the active v120 baseline."""
        v120_path = REPO_ROOT / "configs" / "baseline" / "v120_active.yaml"
        self.assertTrue(v120_path.exists(), f"Missing {v120_path}")

        cfg = from_yaml(v120_path)
        self.assertEqual(cfg.sketch_dim, 256)
        self.assertEqual(cfg.weight_cv, 1.0)
        self.assertEqual(cfg.cv_blocks, "offdiag")
        self.assertEqual(cfg.weight_ct, 1.0)
        self.assertEqual(cfg.ct_pca_dim, 32)
        self.assertEqual(cfg.ct_num_tokens, 256)
        self.assertEqual(cfg.weight_bm, 1.0)
        self.assertEqual(cfg.bm_dim, 32)
        self.assertEqual(cfg.weight_bd, 1.0)
        self.assertEqual(cfg.bd_dim, 256)
        self.assertEqual(cfg.bd_metric, "entropy")
        self.assertEqual(cfg.weight_qa, 1.0)
        self.assertEqual(cfg.qa_dim, 32)
        self.assertEqual(cfg.weight_ds, 1.0)
        self.assertEqual(cfg.ds_dim, 32)
        self.assertEqual(cfg.weight_dd, 0.0)
        self.assertEqual(cfg.aggregation, "trimmed_mean")


if __name__ == "__main__":
    unittest.main()
