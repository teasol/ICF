import unittest

import torch

from src.models.baseline import BaseModel, StructuredPopulationMetaClassifier


class DRCCERTest(unittest.TestCase):
    def _model(self, **overrides) -> BaseModel:
        torch.manual_seed(41)
        kwargs = dict(
            input_dim=8,
            aggregator_num_slots=4,
            aggregator_num_density_slots=3,
            aggregator_context_samples_per_bag=16,
            meta_hidden_dim=16,
            meta_num_heads=4,
            meta_num_set_layers=1,
            meta_relation_hidden_dim=16,
            meta_dr_ccer_enabled=True,
            meta_dr_ccer_abs_topks=(1, 4),
            meta_dr_ccer_frac_topks=(0.1,),
            meta_dr_ccer_expert_hidden=16,
            meta_dr_ccer_max_donors=32,
            bag_representation="poolz_l2",
            num_classes=2,
        )
        kwargs.update(overrides)
        return BaseModel(**kwargs)

    @staticmethod
    def _episode() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        torch.manual_seed(43)
        x = torch.randn(2, 6, 7, 8)
        y = torch.tensor([[0, 1, 0, 1, 0, 1]] * 2)
        mask_index = torch.tensor([[0], [1]])
        return x, y, mask_index

    def test_architecture_version_and_zero_init_preserves_v30_ranking(self) -> None:
        model = self._model().eval()
        x, y, mask_index = self._episode()
        with torch.no_grad():
            logits, aux = model.forward_episode_batch(
                x, y, mask_index, return_auxiliary=True
            )
        self.assertEqual(model.architecture_version, 32)
        # Gate starts closed.
        self.assertTrue((aux["dr_ccer_gate"] >= 0.0).all())
        self.assertTrue((aux["dr_ccer_gate"] <= 1.0).all())
        self.assertLess(aux["dr_ccer_gate"].mean().item(), 0.05)
        # Expert output head is zero-init -> expert margin is exactly zero.
        torch.testing.assert_close(
            aux["dr_ccer_expert_margin"],
            torch.zeros_like(aux["dr_ccer_expert_margin"]),
        )
        # Final margin must preserve v30 ranking (scale+shift are monotonic).
        plain = self._model(meta_dr_ccer_enabled=False).eval()
        with torch.no_grad():
            v30_logits = plain.forward_episode_batch(x, y, mask_index)
        v30_margin = v30_logits[..., 1] - v30_logits[..., 0]
        final_margin = logits[..., 1] - logits[..., 0]
        corr = torch.corrcoef(
            torch.stack((v30_margin.flatten(), final_margin.flatten()))
        )[0, 1].item()
        self.assertGreater(corr, 0.999)

    def test_dense_and_single_episode_paths_match(self) -> None:
        model = self._model().eval()
        x, y, mask_index = self._episode()
        with torch.no_grad():
            batched = model.forward_episode_batch(x[:1], y[:1], mask_index[:1])
            single = model(x[0], y[0], mask_index[0])
        torch.testing.assert_close(
            batched.reshape(-1), single.reshape(-1), atol=1e-4, rtol=0
        )

    def test_donor_permutation_invariance(self) -> None:
        classifier = StructuredPopulationMetaClassifier(
            token_dim=8,
            hidden_dim=16,
            num_heads=4,
            relation_hidden_dim=16,
            dr_ccer_enabled=True,
            dr_ccer_abs_topks=(1, 4),
            dr_ccer_frac_topks=(0.1,),
            dr_ccer_expert_hidden=16,
            num_classes=2,
        ).eval()
        torch.manual_seed(47)
        slots = torch.randn(9, 4, 3, 8)
        labels = torch.randint(0, 2, (9,))
        query = torch.randn(3, 7, 8)
        base = torch.randn(3, 2)
        perm = torch.randperm(9)
        with torch.no_grad():
            out1, _ = classifier._dr_ccer_logits(
                {"slots": slots}, labels, query, base
            )
            out2, _ = classifier._dr_ccer_logits(
                {"slots": slots[perm]}, labels[perm], query, base
            )
        torch.testing.assert_close(out1, out2, atol=1e-5, rtol=1e-4)

    def test_label_equivariance_flips_margin(self) -> None:
        classifier = StructuredPopulationMetaClassifier(
            token_dim=8,
            hidden_dim=16,
            num_heads=4,
            relation_hidden_dim=16,
            dr_ccer_enabled=True,
            dr_ccer_abs_topks=(1, 4),
            dr_ccer_frac_topks=(0.1,),
            dr_ccer_expert_hidden=16,
            num_classes=2,
        ).eval()
        torch.manual_seed(53)
        slots = torch.randn(9, 4, 3, 8)
        labels = torch.randint(0, 2, (9,))
        query = torch.randn(3, 7, 8)
        base = torch.randn(3, 2)
        flipped = 1 - labels
        with torch.no_grad():
            out1, aux1 = classifier._dr_ccer_logits(
                {"slots": slots}, labels, query, base
            )
            out2, aux2 = classifier._dr_ccer_logits(
                {"slots": slots}, flipped, query, base
            )
        # The expert margin (class1 - class0) must flip sign, because the
        # per-class evidence roles are exchanged by the label flip.
        torch.testing.assert_close(
            aux1["dr_ccer_expert_margin"],
            -aux2["dr_ccer_expert_margin"],
            atol=1e-5,
            rtol=1e-4,
        )

    def test_duplicate_route_masking(self) -> None:
        classifier = StructuredPopulationMetaClassifier(
            token_dim=8,
            hidden_dim=16,
            num_heads=4,
            relation_hidden_dim=16,
            dr_ccer_enabled=True,
            dr_ccer_abs_topks=(1, 4),
            dr_ccer_frac_topks=(0.1,),
            dr_ccer_use_bottom_tail=False,
            dr_ccer_expert_hidden=16,
            num_classes=2,
        ).eval()
        # n=4: abs top-1 -> count 1, abs top-4 -> count 4, frac 0.1 -> count 1
        # (duplicate of top-1), dense mean -> count 4 (duplicate of top-4),
        # agreement mean -> count 4 (duplicate). Only the first occurrence of
        # each count should keep router mass.
        slots = torch.randn(6, 4, 3, 8)
        labels = torch.tensor([0, 1, 0, 1, 0, 1])
        query = torch.randn(2, 4, 8)
        base = torch.randn(2, 2)
        _, aux = classifier._dr_ccer_logits(
            {"slots": slots}, labels, query, base
        )
        weights = aux["dr_ccer_route_weights"]
        # 5 routes: abs(1), abs(4), frac(0.1), dense mean, agreement mean.
        self.assertEqual(weights.shape, (2, 5))
        # Duplicate-count routes must carry zero mass.
        self.assertEqual(weights[0, 2].item(), 0.0)  # frac duplicates top-1
        self.assertEqual(weights[0, 3].item(), 0.0)  # dense duplicates top-4
        self.assertEqual(weights[0, 4].item(), 0.0)  # agreement duplicates top-4
        self.assertGreater(weights[0, 0].item(), 0.0)
        self.assertGreater(weights[0, 1].item(), 0.0)

    def test_expert_margin_learns_and_gate_stays_bounded(self) -> None:
        model = self._model().train()
        x, y, mask_index = self._episode()
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=1e-2
        )
        for _ in range(8):
            optimizer.zero_grad(set_to_none=True)
            logits, aux = model.forward_episode_batch(
                x, y, mask_index, return_auxiliary=True
            )
            loss = logits.square().mean() + aux["dr_ccer_expert_margin"].square().mean()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            logits, aux = model.forward_episode_batch(
                x, y, mask_index, return_auxiliary=True
            )
        self.assertGreater(aux["dr_ccer_expert_margin"].std().item(), 1e-3)
        self.assertTrue((aux["dr_ccer_gate"] >= 0.0).all())
        self.assertTrue((aux["dr_ccer_gate"] <= 1.0).all())


if __name__ == "__main__":
    unittest.main()
