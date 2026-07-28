"""Unit tests for the v22 batched multi-episode forward path.

Replaces the v21 large-context retrieval suite. `forward_episode_batch` and
the 4D branch of `forward` survived the v22 retrieval removal because they are
what lets one optimizer step average gradients over several episodes; only the
retrieval layer that used to shrink each episode's candidate pool is gone.

Shapes are deliberately small so this runs in seconds on CPU. The old suite
built 96 bags x 100 cells x 512 dims and became effectively un-runnable
outside a GPU, which is why its regressions went unnoticed.
"""

import unittest

import torch

from src.models.baseline import BaseModel


class TestBatchedEpisodeForward(unittest.TestCase):
    def setUp(self) -> None:
        self.input_dim = 32
        self.model = BaseModel(
            input_dim=self.input_dim,
            meta_hidden_dim=16,
            meta_num_heads=4,
            meta_num_set_layers=1,
            meta_relation_hidden_dim=16,
            meta_ridge_dim=4,
            num_classes=2,
        )
        self.model.eval()

    def _episode_batch(
        self, episodes: int, bags: int, cells: int, queries: int = 1
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        generator = torch.Generator().manual_seed(0)
        x = torch.randn(episodes, bags, cells, self.input_dim, generator=generator)
        # Guarantee both classes are present as context in every episode so the
        # class-memory branch always has something to attend to.
        y = torch.randint(0, 2, (episodes, bags), generator=generator)
        y[:, 0] = 0
        y[:, 1] = 1
        mask_index = (
            torch.arange(bags - queries, bags, dtype=torch.long)
            .unsqueeze(0)
            .expand(episodes, queries)
            .contiguous()
        )
        return x, y, mask_index

    def test_forward_episode_batch_shape(self) -> None:
        episodes, bags, cells = 3, 10, 12
        x, y, mask_index = self._episode_batch(episodes, bags, cells)
        with torch.no_grad():
            logits = self.model.forward_episode_batch(x, y, mask_index)
        self.assertEqual(logits.shape, (episodes, 1, 2))

    def test_forward_episode_batch_multi_query(self) -> None:
        episodes, bags, cells, queries = 2, 12, 10, 4
        x, y, mask_index = self._episode_batch(episodes, bags, cells, queries)
        with torch.no_grad():
            logits = self.model.forward_episode_batch(x, y, mask_index)
        self.assertEqual(logits.shape, (episodes, queries, 2))

    def test_forward_4d_matches_per_episode_forward(self) -> None:
        """The 4D branch must be a pure loop over independent episodes."""
        episodes, bags, cells = 3, 10, 12
        x, y, mask_index = self._episode_batch(episodes, bags, cells)
        with torch.no_grad():
            batched = self.model.forward(x, y, mask_index=mask_index)
            separate = torch.cat(
                [
                    self.model.forward(x[e], y[e], mask_index=mask_index[e])
                    for e in range(episodes)
                ],
                dim=0,
            )
        self.assertEqual(batched.shape, (episodes, 2))
        self.assertTrue(torch.allclose(batched, separate, atol=1e-5, rtol=1e-5))

    def test_out_of_range_mask_index_is_rejected(self) -> None:
        x, y, mask_index = self._episode_batch(2, 8, 10)
        mask_index = mask_index.clone()
        mask_index[0, 0] = 99
        with self.assertRaises(IndexError):
            self.model.forward_episode_batch(x, y, mask_index)


if __name__ == "__main__":
    unittest.main()
