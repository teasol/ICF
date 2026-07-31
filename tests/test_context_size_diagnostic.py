import unittest

import torch

from scripts.diagnose_context_size import parse_sizes, split_indices


class TestContextSizeDiagnostic(unittest.TestCase):
    def test_split_is_balanced_disjoint_and_nested(self) -> None:
        labels = torch.tensor([0, 1] * 10)
        chosen = split_indices(labels, queries_per_class=2, max_context_per_class=6)
        self.assertIsNotNone(chosen)
        query, streams = chosen
        self.assertEqual(torch.bincount(labels[query], minlength=2).tolist(), [2, 2])
        self.assertEqual(len(set(query.tolist()) & set(torch.cat(streams).tolist())), 0)
        for per_class in (2, 4, 6):
            context = torch.cat((streams[0][:per_class], streams[1][:per_class]))
            self.assertEqual(
                torch.bincount(labels[context], minlength=2).tolist(),
                [per_class, per_class],
            )
        self.assertTrue(torch.equal(streams[0][:2], streams[0][:4][:2]))
        self.assertTrue(torch.equal(streams[1][:4], streams[1][:6][:4]))

    def test_split_rejects_insufficient_class_members(self) -> None:
        labels = torch.tensor([0] * 10 + [1] * 3)
        self.assertIsNone(
            split_indices(labels, queries_per_class=2, max_context_per_class=2)
        )

    def test_context_sizes_must_be_increasing_even_values(self) -> None:
        self.assertEqual(parse_sizes("10,20,40"), (10, 20, 40))
        with self.assertRaises(Exception):
            parse_sizes("10,15")
        with self.assertRaises(Exception):
            parse_sizes("20,10")


if __name__ == "__main__":
    unittest.main()
