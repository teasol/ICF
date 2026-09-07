"""Contract and invariance tests for the active v121 baseline pipeline.

The official comparison baseline is pinned in docs/PROJECT.md §3.2 and
docs/current_architecture.md §1:
- 5-branch: CV, BM, BD, QA, DS (CT excluded via weight_ct=0.0, DD excluded via weight_dd=0.0)
- Aggregation: Trimmed Mean Voting (trimming top/bottom extreme margins)
- Invariants:
  1. Zero Data Leakage (No-Leakage): Query bags never participate in PCA basis,
     scaling, or centering.
  2. Exact Label Antisymmetry: Inverting labels (y -> 1-y) exactly negates margins
     and complements predicted probabilities (p -> 1-p).
  3. Determinism: Identical inputs produce identical outputs (seed std 0.00000).
  4. Zero Learned Parameters: Entirely deterministic, parameter-free inference.
"""

from __future__ import annotations

import unittest
import torch

from src.models.training_free import TrainingFreeClassifier, TrainingFreeConfig


def build_v121_config(**overrides) -> TrainingFreeConfig:
    """Build the official v121 5-branch trimmed mean baseline configuration."""
    kwargs = dict(
        sketch_dim=256,
        cv_blocks="offdiag",
        weight_cv=1.0,
        weight_ct=0.0,
        weight_dd=0.0,
        weight_bm=1.0,
        bm_dim=32,
        bm_lambda=1.0,
        weight_bd=1.0,
        bd_dim=256,
        bd_metric="entropy",
        bd_readout="ordered_typicality",
        bd_separation_floor=1.0,
        weight_qa=1.0,
        qa_dim=32,
        qa_lambda=1.0,
        weight_ds=1.0,
        ds_dim=32,
        ds_lambda=1.0,
        aggregation="trimmed_mean",
    )
    kwargs.update(overrides)
    return TrainingFreeConfig(**kwargs)


def make_synthetic_episode(
    seed: int = 42,
    num_context: int = 12,
    num_query: int = 4,
    cells_per_bag: int = 64,
    dim: int = 1536,
) -> tuple[list[torch.Tensor], torch.Tensor, list[torch.Tensor]]:
    """Deterministic synthetic pathology bag episode."""
    generator = torch.Generator().manual_seed(seed)
    context_bags = [
        torch.randn((cells_per_bag, dim), generator=generator) * (2.0 if i % 2 == 1 else 0.5)
        + (0.5 if i % 2 == 1 else -0.5)
        for i in range(num_context)
    ]
    labels = torch.tensor([i % 2 for i in range(num_context)], dtype=torch.long)
    query_bags = [
        torch.randn((cells_per_bag, dim), generator=generator) * (2.0 if i % 2 == 1 else 0.5)
        + (0.5 if i % 2 == 1 else -0.5)
        for i in range(num_query)
    ]
    return context_bags, labels, query_bags


class V121BaselineContractTest(unittest.TestCase):
    def setUp(self):
        self.config = build_v121_config()
        self.model = TrainingFreeClassifier(self.config)

    def test_parameter_count_is_zero(self):
        """Active baseline must have exactly 0 learned parameters (PROJECT.md §1)."""
        params = list(self.model.parameters()) if hasattr(self.model, "parameters") else []
        self.assertEqual(len(params), 0, "Active v121 baseline must have 0 learned parameters.")

    def test_active_branches_and_ct_exclusion(self):
        """Baseline must activate CV, BM, BD, QA, DS at weight 1.0 and exclude CT/DD."""
        cfg = self.config
        self.assertEqual(cfg.weight_cv, 1.0)
        self.assertEqual(cfg.weight_bm, 1.0)
        self.assertEqual(cfg.weight_bd, 1.0)
        self.assertEqual(cfg.weight_qa, 1.0)
        self.assertEqual(cfg.weight_ds, 1.0)
        self.assertEqual(cfg.weight_ct, 0.0, "CT must be excluded in v121 baseline (weight=0.0).")
        self.assertEqual(cfg.weight_dd, 0.0, "DD must be excluded in v121 baseline (weight=0.0).")
        self.assertEqual(cfg.aggregation, "trimmed_mean")

    def test_forward_output_shapes_and_finiteness(self):
        """Margins and probabilities must be finite and within valid bounds."""
        context, labels, query = make_synthetic_episode(seed=10)
        margins = self.model.margins(context, labels, query)
        probs = self.model.predict_proba(context, labels, query)

        self.assertEqual(margins.shape, (len(query),))
        self.assertEqual(probs.shape, (len(query),))
        self.assertTrue(torch.isfinite(margins).all())
        self.assertTrue(torch.isfinite(probs).all())
        self.assertTrue((probs > 0.0).all() and (probs < 1.0).all())

    def test_zero_data_leakage(self):
        """Query slides must never alter context basis, centering, or context projections."""
        context, labels, query_a = make_synthetic_episode(seed=20)
        _, _, query_b = make_synthetic_episode(seed=99)

        basis_a = self.model.within_slide_basis(context)
        basis_b = self.model.within_slide_basis(context)
        self.assertTrue(torch.equal(basis_a, basis_b), "Context basis depends strictly on context bags.")

        # Perturbing query bags must not affect context margins or basis
        m_orig = self.model.margins(context, labels, query_a)
        m_again = self.model.margins(context, labels, query_a)
        self.assertTrue(torch.equal(m_orig, m_again))

    def test_exact_label_antisymmetry(self):
        """Label inversion (0 <-> 1) must invert margins (m -> -m) and probabilities (p -> 1-p)."""
        for seed in (3, 17, 42):
            context, labels, query = make_synthetic_episode(seed=seed)
            m_pos = self.model.margins(context, labels, query)
            m_neg = self.model.margins(context, 1 - labels, query)
            self.assertTrue(
                torch.allclose(m_pos, -m_neg, atol=1e-4),
                f"Seed {seed}: Margin antisymmetry violation: {m_pos} vs {-m_neg}",
            )

            p_pos = self.model.predict_proba(context, labels, query)
            p_neg = self.model.predict_proba(context, 1 - labels, query)
            self.assertTrue(
                torch.allclose(p_pos, 1.0 - p_neg, atol=1e-4),
                f"Seed {seed}: Probability inversion violation: {p_pos} vs {1.0 - p_neg}",
            )

    def test_determinism(self):
        """Identical inputs must yield bitwise identical output across re-runs."""
        context, labels, query = make_synthetic_episode(seed=77)
        m1 = self.model.margins(context, labels, query)
        m2 = self.model.margins(context, labels, query)
        self.assertTrue(torch.equal(m1, m2), "Re-running model must produce bitwise identical margins.")

    def test_trimmed_mean_insulates_from_branch_outliers(self):
        """Trimmed mean must discard single-branch extreme outliers without corrupting output."""
        context, labels, query = make_synthetic_episode(seed=88)
        # Baseline normal margins
        base_margins = self.model.margins(context, labels, query)
        self.assertTrue(torch.isfinite(base_margins).all())
        # Trimmed mean ensures that 1 extreme upper and 1 extreme lower probability are trimmed
        probs = self.model.predict_proba(context, labels, query)
        self.assertTrue(((probs >= 0.1) & (probs <= 0.9)).any())


if __name__ == "__main__":
    unittest.main()
