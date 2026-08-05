import unittest

import torch

from src.models.baseline import BaseModel, StructuredPopulationMetaClassifier


class CCERV2Test(unittest.TestCase):
    def _model(self) -> BaseModel:
        torch.manual_seed(41)
        return BaseModel(
            input_dim=8,
            aggregator_num_slots=4,
            aggregator_num_density_slots=3,
            aggregator_context_samples_per_bag=16,
            meta_hidden_dim=16,
            meta_num_heads=4,
            meta_num_set_layers=1,
            meta_relation_hidden_dim=16,
            meta_ccer_v2_topks=(1, 4),
            meta_ccer_v2_route_floor=0.30,
            meta_ccer_v2_residual_scale=0.10,
            bag_representation="poolz_l2",
            num_classes=2,
        )

    @staticmethod
    def _episode() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        torch.manual_seed(43)
        x = torch.randn(2, 6, 7, 8)
        y = torch.tensor([[0, 1, 0, 1, 0, 1]] * 2)
        mask_index = torch.tensor([[0], [1]])
        return x, y, mask_index

    def test_zero_output_preserves_base_and_stages_gradients(self) -> None:
        model = self._model().train()
        x, y, mask_index = self._episode()
        logits, auxiliary = model.forward_episode_batch(
            x, y, mask_index, return_auxiliary=True
        )
        self.assertEqual(model.architecture_version, 31)
        self.assertEqual(auxiliary["ccer_v2_route_scores"].shape, (2, 1, 2, 3))
        self.assertEqual(auxiliary["ccer_v2_route_weights"].shape, (2, 1, 3))
        torch.testing.assert_close(
            auxiliary["ccer_v2_logits"],
            torch.zeros_like(auxiliary["ccer_v2_logits"]),
        )

        logits.square().mean().backward()
        head = model.meta_classifier.ccer_v2_output_head.weight
        self.assertIsNotNone(head.grad)
        self.assertTrue(torch.isfinite(head.grad).all())
        encoder = model.meta_classifier.ccer_v2_instance_encoder[1].weight
        self.assertTrue(encoder.grad is None or torch.count_nonzero(encoder.grad) == 0)

        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            head.fill_(1.0)
        model.forward_episode_batch(x, y, mask_index).square().mean().backward()
        self.assertIsNotNone(encoder.grad)
        self.assertGreater(torch.count_nonzero(encoder.grad).item(), 0)

    def test_dense_and_single_episode_paths_match(self) -> None:
        model = self._model().eval()
        x, y, mask_index = self._episode()
        with torch.no_grad():
            batched = model.forward_episode_batch(x[:1], y[:1], mask_index[:1])
            single = model(x[0], y[0], mask_index[0])
        torch.testing.assert_close(
            batched.reshape(-1), single.reshape(-1), atol=1e-4, rtol=0
        )

    def test_top1_survives_background_and_every_route_has_mass(self) -> None:
        classifier = StructuredPopulationMetaClassifier(
            token_dim=8,
            hidden_dim=16,
            num_heads=4,
            relation_hidden_dim=16,
            ccer_v2_topks=(1, 4),
            ccer_v2_route_floor=0.30,
            num_classes=2,
        ).eval()
        torch.manual_seed(47)
        prototypes = torch.randn(2, 4, 16)
        evidence = torch.tensor([[[3.0, 0.0], [-3.0, 0.0]]])
        appended = torch.cat((evidence, torch.zeros(1, 2, 5)), dim=-1)
        original = classifier._ccer_v2_pool_evidence(evidence, prototypes)
        extended = classifier._ccer_v2_pool_evidence(appended, prototypes)

        torch.testing.assert_close(original[1][..., 0], extended[1][..., 0])
        self.assertFalse(torch.allclose(original[1][..., -1], extended[1][..., -1]))
        minimum_weight = classifier.ccer_v2_route_floor / 3
        self.assertGreaterEqual(extended[2].min().item(), minimum_weight - 1e-6)


if __name__ == "__main__":
    unittest.main()
