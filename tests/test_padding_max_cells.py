"""Contract tests for `data.padding_max_cells` (docs SS120).

The dense training collator has always subsampled every bag down to a hard-coded
4,096 cells. That is a THIRD, independent ceiling -- the dataset's
`per_bag_max_cells` and the model's `max_cells` are the other two -- and it went
unnoticed for the whole project because no arm had ever asked for a bag bigger
than 4,096. v92 does, and without these tests the arm would have trained on
silently truncated bags and "measured" a data change that never reached the
model.

The failure mode is the dangerous kind: no error, no warning, plausible loss
curves, and a result that looks like a clean null. `test_bags_are_truncated_at_the_default`
is what pins that the cap is real, and `test_default_is_unchanged` is what pins
that making it configurable did not move any existing arm.
"""

import unittest

import torch

from src.modules.data_interface import (
    SYNTHETIC_PADDING_MAX_CELLS,
    _collate_ragged_batch,
    collate_synthetic_training_episode,
)

DIM = 4


def episode(bag_sizes, label_values=None):
    """One ragged episode: a list of [n_i, DIM] bags plus its labels."""
    bags = [torch.randn(n, DIM) for n in bag_sizes]
    if label_values is None:
        label_values = [i % 2 for i in range(len(bag_sizes))]
    return bags, torch.tensor(label_values, dtype=torch.long)


class PaddingCapDefaultTest(unittest.TestCase):
    def test_default_is_unchanged(self):
        """Every arm before v92 relied on 4,096; the refactor must not move it."""
        self.assertEqual(SYNTHETIC_PADDING_MAX_CELLS, 4096)

    def test_bags_are_truncated_at_the_default(self):
        x, y, cell_mask, bag_mask = _collate_ragged_batch([episode([10, 5000])])
        self.assertEqual(x.shape[2], 4096)
        self.assertEqual(int(cell_mask[0, 1].sum()), 4096)
        self.assertEqual(int(cell_mask[0, 0].sum()), 10)

    def test_width_follows_the_largest_bag_when_under_the_cap(self):
        """The cap is a ceiling, not a target -- small episodes stay small."""
        x, *_ = _collate_ragged_batch([episode([10, 300])])
        self.assertEqual(x.shape[2], 300)


class PaddingCapOverrideTest(unittest.TestCase):
    def test_raising_the_cap_actually_widens_bags(self):
        """The whole point of v92: per_bag_max_cells above 4,096 must reach the
        model instead of being silently subsampled back down."""
        x, y, cell_mask, bag_mask = _collate_ragged_batch(
            [episode([10, 12288])], max_cells=12288
        )
        self.assertEqual(x.shape[2], 12288)
        self.assertEqual(int(cell_mask[0, 1].sum()), 12288)

    def test_raised_cap_still_truncates_beyond_itself(self):
        x, _, cell_mask, _ = _collate_ragged_batch(
            [episode([20000])], max_cells=12288
        )
        self.assertEqual(x.shape[2], 12288)
        self.assertEqual(int(cell_mask[0, 0].sum()), 12288)

    def test_lowering_the_cap_works_too(self):
        x, _, cell_mask, _ = _collate_ragged_batch([episode([3000])], max_cells=512)
        self.assertEqual(x.shape[2], 512)
        self.assertEqual(int(cell_mask[0, 0].sum()), 512)

    def test_rejects_nonpositive_cap(self):
        with self.assertRaises(ValueError):
            _collate_ragged_batch([episode([10])], max_cells=0)

    def test_public_collate_forwards_the_cap(self):
        """`collate_synthetic_training_episode` is what the DataLoader binds, so
        the parameter has to survive that hop."""
        x, *_ = collate_synthetic_training_episode(
            [episode([10, 8000])], max_cells=8192
        )
        self.assertEqual(x.shape[2], 8000)
        x, *_ = collate_synthetic_training_episode([episode([10, 8000])])
        self.assertEqual(x.shape[2], 4096)


class PaddingCapMaskTest(unittest.TestCase):
    def test_padding_stays_masked_and_labelled_minus_one(self):
        """Truncation must not disturb the padded-bag bookkeeping: padded bags
        carry label -1 so they are never sampled as queries or used as context."""
        x, y, cell_mask, bag_mask = _collate_ragged_batch(
            [episode([6000, 10]), episode([50])], max_cells=6000
        )
        self.assertEqual(x.shape[:3], (2, 2, 6000))
        self.assertTrue(bool(bag_mask[0, 0]) and bool(bag_mask[0, 1]))
        self.assertTrue(bool(bag_mask[1, 0]))
        self.assertFalse(bool(bag_mask[1, 1]))
        self.assertEqual(int(y[1, 1]), -1)
        self.assertEqual(int(cell_mask[1, 1].sum()), 0)
        self.assertTrue(torch.equal(x[1, 1], torch.zeros(6000, DIM)))


if __name__ == "__main__":
    unittest.main()
