import unittest
import torch
import time
from src.models.training_free import TrainingFreeClassifier, TrainingFreeConfig

class InEpisodeLOOTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.dim = 64
        self.n_ctx = 40
        self.n_qry = 20
        self.ctx_bags = [torch.randn(torch.randint(50, 120, (1,)).item(), self.dim) for _ in range(self.n_ctx)]
        self.ctx_labels = torch.tensor([i % 2 for i in range(self.n_ctx)], dtype=torch.long)
        self.qry_bags = [torch.randn(torch.randint(50, 120, (1,)).item(), self.dim) for _ in range(self.n_qry)]

    def test_in_episode_loo_forward_and_antisymmetry(self):
        config = TrainingFreeConfig(
            sketch_dim=32,
            weight_cv=1.0, weight_ct=1.0, weight_bm=1.0, weight_bd=1.0,
            weight_qa=1.0, weight_ds=1.0, weight_de=1.0, weight_sw=1.0,
            aggregation="context_loo_power",
            loo_gamma=2.0,
            loo_floor=0.50,
        )
        clf = TrainingFreeClassifier(config)

        t0 = time.perf_counter()
        margins_orig = clf.margins(self.ctx_bags, self.ctx_labels, self.qry_bags)
        t_elapsed = (time.perf_counter() - t0) * 1000.0
        print(f"\n[Test] Forward pass with 8 branches + In-Episode Context LOO completed in: {t_elapsed:.2f} ms")

        # Inverted labels for exact antisymmetry check
        inverted_labels = 1 - self.ctx_labels
        margins_inv = clf.margins(self.ctx_bags, inverted_labels, self.qry_bags)

        sym_error = (margins_orig + margins_inv).abs().max().item()
        print(f"[Test] Exact Label Antisymmetry Error: {sym_error:.8e}")
        self.assertLess(sym_error, 1e-4, f"Antisymmetry broken: {sym_error}")
        print("[Test] In-Episode Context LOO Unit Test PASSED perfectly!")

if __name__ == "__main__":
    unittest.main()
