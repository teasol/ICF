"""Controlled replay, label symmetry, and solver-state restoration for RU-81."""
import unittest
from types import SimpleNamespace
import torch
from scripts.analysis.ru81_probe import geometry, ensemble, install
from src.models.common.solvers import solve_kernel_ridge, standardise
from src.models.set_transformer_ridge import SetTransformerRidgeModel


class TestRU81Probe(unittest.TestCase):
    def test_geometry_is_label_flip_invariant(self):
        torch.manual_seed(81)
        x = torch.randn(9, 4)
        y = torch.tensor([0] * 3 + [1] * 6)
        a, b = geometry(x, y), geometry(x, 1-y)
        self.assertEqual(a['penalties'], b['penalties'])
        self.assertTrue(all(a['df_probe'][i] >= a['df_probe'][i+1] for i in range(2)))

    def test_no_query_information_in_geometry(self):
        x = torch.tensor([[0., 1.], [1., 0.], [2., 3.], [3., 2.]])
        y = torch.tensor([0, 0, 1, 1])
        self.assertTrue(all(p > 0 for p in geometry(x, y)['penalties']))
        with self.assertRaises(ValueError):
            geometry(torch.zeros_like(x), y)

    def test_hooks_preserve_baseline_and_restore_penalty(self):
        torch.manual_seed(3)
        x, q = torch.randn(10, 4), torch.randn(5, 4)
        y = torch.tensor([0, 1] * 5)
        obj = SimpleNamespace(num_classes=2, ridge_log_lambda=torch.tensor(0.),
                              ridge_log_scale=torch.tensor(0.), _normalize_descriptors=standardise)
        original_cv = SetTransformerRidgeModel._ridge_logits
        def trial(**kwargs):
            cv = SetTransformerRidgeModel._ridge_logits(obj, x, y, q)
            m = {'cv': cv[:, 1]-cv[:, 0], 'bd': q[:, 0]}
            for b in ('bm', 'qa', 'ds'):
                m[b] = ev._solve_kernel_ridge(x, y, q, kernel='linear', reg_lambda=1.)
            return {'probability': ensemble(m), 'queried_ids': list(range(5)),
                    'target': torch.tensor([0, 1, 0, 1, 0]), **{'m_'+b:v for b,v in m.items()}}
        ev = SimpleNamespace(_solve_kernel_ridge=solve_kernel_ridge, evaluate_trial=trial)
        with torch.no_grad():
            baseline = trial()['probability'].clone()
            try:
                rows = install(ev)
                result = ev.evaluate_trial()
                torch.testing.assert_close(result['probability'], baseline, rtol=0, atol=0)
                self.assertEqual(obj.ridge_log_lambda.item(), 0.)
                self.assertEqual(len(rows), 1)
                self.assertEqual(set(rows[0]['probe']), {'cv', 'bm', 'qa', 'ds'})
            finally:
                SetTransformerRidgeModel._ridge_logits = original_cv


if __name__ == '__main__':
    unittest.main()
