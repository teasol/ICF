import unittest

import torch

from src.models.baseline import BaseModel, StructuredPopulationMetaClassifier


class CCERLiteTest(unittest.TestCase):
    def _model(self, *, project: bool = True) -> BaseModel:
        torch.manual_seed(19)
        return BaseModel(
            input_dim=8,
            meta_hidden_dim=16,
            meta_num_heads=4,
            meta_num_set_layers=1,
            meta_relation_hidden_dim=16,
            meta_ccer_temperatures=(0.25, 1.0, 4.0),
            meta_ccer_residual_scale=0.10,
            meta_ccer_presence_temperature=0.5,
            project_structured_tokens=project,
            projection_bottleneck_dim=4 if project else None,
            projection_residual_mean=project,
            bag_representation="poolz_l2",
            num_classes=2,
        )

    @staticmethod
    def _episode() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        torch.manual_seed(23)
        x = torch.randn(2, 6, 9, 8)
        y = torch.tensor([[0, 1, 0, 1, 0, 1]] * 2)
        mask_index = torch.tensor([[0], [1]])
        return x, y, mask_index

    def test_forward_backward_and_ccer_parameter_gradients(self) -> None:
        model = self._model().train()
        x, y, mask_index = self._episode()
        logits, auxiliary = model.forward_episode_batch(
            x, y, mask_index, return_auxiliary=True
        )
        self.assertEqual(logits.shape, (2, 1, 2))
        self.assertEqual(auxiliary["ccer_route_scores"].shape, (2, 1, 2, 3))
        self.assertEqual(auxiliary["ccer_presence"].shape, (2, 1))
        self.assertEqual(auxiliary["ccer_route_weights"].shape, (2, 3))
        self.assertTrue(torch.isfinite(logits).all())

        logits.square().mean().backward()
        for name in (
            "ccer_route_logits",
            "ccer_support_router.weight",
            "ccer_null_threshold",
            "ccer_residual_logit",
        ):
            parameter = dict(model.meta_classifier.named_parameters())[name]
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_dense_and_list_paths_match(self) -> None:
        model = self._model(project=False).eval()
        x, y, mask_index = self._episode()
        with torch.no_grad():
            dense = model.forward_episode_batch(x[:1], y[:1], mask_index[:1])
            listed = model(x[0], y[0], mask_index[0])
        torch.testing.assert_close(
            dense.reshape(-1), listed.reshape(-1), atol=1e-4, rtol=0
        )

    def test_logmeanexp_is_invariant_to_uniform_duplication(self) -> None:
        classifier = StructuredPopulationMetaClassifier(
            token_dim=8,
            hidden_dim=16,
            num_heads=4,
            relation_hidden_dim=16,
            ccer_temperatures=(0.25, 1.0, 4.0),
            num_classes=2,
        ).eval()
        torch.manual_seed(29)
        evidence = torch.randn(3, 2, 7)
        memories = torch.randn(2, 8, 16)
        original = classifier._ccer_pool_evidence(evidence, memories)
        duplicated = classifier._ccer_pool_evidence(
            evidence.repeat_interleave(3, dim=-1), memories
        )
        torch.testing.assert_close(original[0], duplicated[0], atol=1e-5, rtol=0)
        torch.testing.assert_close(original[1], duplicated[1], atol=1e-5, rtol=0)
        torch.testing.assert_close(original[2], duplicated[2], atol=1e-6, rtol=0)


if __name__ == "__main__":
    unittest.main()
