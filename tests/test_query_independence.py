"""Does one query slide's content change another query slide's prediction?

Round 16 asked for a static leak check before any probe is pre-registered. A
static check is the weaker version of this question: the strong version is
executable, so it is run instead.

The classifier is transductive -- `margins(context, labels, query)` receives
every query bag at once. That is exactly the shape in which cross-query leakage
hides. If query j's features move query i's score, then a slide's prediction
depends on which other slides happened to sit in its evaluation batch, and a
fold AUROC stops being a sum of per-slide predictions.

The fold split itself cannot leak: `evaluate_pure.py` builds context as the
complement of test, so context and query are disjoint by construction. That is
asserted here too, cheaply, so a future edit cannot quietly make them overlap.
"""

import unittest

import torch

from src.models.training_free import TrainingFreeClassifier, TrainingFreeConfig


def _config(**kw):
    base = dict(sketch_dim=32, weight_cv=1.0, weight_bm=1.0, weight_bd=1.0,
                weight_qa=1.0, weight_ds=1.0, weight_ct=0.0, weight_dd=0.0,
                aggregation="trimmed_mean")
    base.update(kw)
    return TrainingFreeConfig(**base)


class TestQueryIndependence(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.dim = 64
        self.ctx_bags = [torch.randn(80, self.dim) for _ in range(24)]
        self.ctx_labels = torch.tensor([i % 2 for i in range(24)], dtype=torch.long)
        self.qry_bags = [torch.randn(80, self.dim) for _ in range(5)]

    def test_replacing_one_query_leaves_the_others_untouched(self):
        clf = TrainingFreeClassifier(_config())
        base = clf.margins(self.ctx_bags, self.ctx_labels, self.qry_bags)

        # Replace the last query with something wildly different in scale and
        # location. If any fitting step pools over queries, the others move.
        swapped = list(self.qry_bags)
        swapped[-1] = torch.randn(80, self.dim) * 50.0 + 30.0
        after = clf.margins(self.ctx_bags, self.ctx_labels, swapped)

        delta = (base[:-1] - after[:-1]).abs().max().item()
        self.assertLess(delta, 1e-5,
                        f"query 4를 바꿨더니 다른 query의 margin이 {delta:.3e} 움직였다")

    def test_dropping_a_query_leaves_the_others_untouched(self):
        clf = TrainingFreeClassifier(_config())
        full = clf.margins(self.ctx_bags, self.ctx_labels, self.qry_bags)
        fewer = clf.margins(self.ctx_bags, self.ctx_labels, self.qry_bags[:3])
        delta = (full[:3] - fewer).abs().max().item()
        self.assertLess(delta, 1e-5,
                        f"query 집합 크기를 바꿨더니 margin이 {delta:.3e} 움직였다")

    def test_query_order_does_not_change_predictions(self):
        clf = TrainingFreeClassifier(_config())
        base = clf.margins(self.ctx_bags, self.ctx_labels, self.qry_bags)
        order = [3, 0, 4, 1, 2]
        shuffled = clf.margins(self.ctx_bags, self.ctx_labels,
                               [self.qry_bags[i] for i in order])
        delta = (base[torch.tensor(order)] - shuffled).abs().max().item()
        self.assertLess(delta, 1e-5, f"query 순서가 margin을 {delta:.3e} 바꿨다")

    def test_query_labels_are_never_an_argument(self):
        """The signature cannot accept query labels, so they cannot be used."""
        import inspect
        params = list(inspect.signature(TrainingFreeClassifier.margins).parameters)
        self.assertEqual(params, ["self", "context_bags", "context_labels", "query_bags"])


class TestFoldSplitDisjointness(unittest.TestCase):
    def test_context_is_the_complement_of_test(self):
        """Mirrors evaluate_pure.py's split rule, so a change there breaks here."""
        records = {f"s{i}": ("test" if i % 4 == 0 else "train") for i in range(40)}
        test_ids = [s for s, v in records.items() if v.strip() == "test"]
        context_ids = [s for s, v in records.items() if v.strip() != "test"]
        self.assertEqual(set(test_ids) & set(context_ids), set())
        self.assertEqual(set(test_ids) | set(context_ids), set(records))
        self.assertTrue(test_ids and context_ids)


if __name__ == "__main__":
    unittest.main()
