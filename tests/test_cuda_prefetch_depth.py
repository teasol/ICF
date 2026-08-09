"""The prefetch queue must deepen without dropping or reordering batches.

Depth is a memory-for-latency trade: each queued episode sits on the GPU, and
synthetic episodes reach ~5.9 GB. What must never change is the epoch itself --
same batches, same order, same count. The drain-on-exhaustion path is the easy
one to get wrong: a queued future that hits the end of the source must not
discard the batches already produced behind it.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.modules.data_interface import _CudaPrefetchIterator  # noqa: E402


class _FakeStream:
    def synchronize(self) -> None:
        return None

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        return None


class PrefetchDepthTest(unittest.TestCase):
    def _drain(self, items, depth):
        """Run the iterator without needing CUDA."""
        import src.modules.data_interface as module

        original_stream = module.torch.cuda.Stream
        original_ctx = module.torch.cuda.stream
        module.torch.cuda.Stream = _FakeStream
        module.torch.cuda.stream = lambda _stream: _FakeStream()
        try:
            iterator = _CudaPrefetchIterator(iter(items), depth=depth)
            return list(iterator)
        finally:
            module.torch.cuda.Stream = original_stream
            module.torch.cuda.stream = original_ctx

    def test_every_batch_survives_at_each_depth(self) -> None:
        items = list(range(17))
        for depth in (1, 2, 4, 32):
            with self.subTest(depth=depth):
                self.assertEqual(
                    self._drain(items, depth),
                    items,
                    f"depth={depth} changed the epoch's batches or their order",
                )

    def test_depth_deeper_than_the_epoch_still_drains(self) -> None:
        """The queue exhausts the source before the consumer asks for anything."""
        items = [1, 2, 3]
        self.assertEqual(self._drain(items, depth=10), items)

    def test_empty_source_stops_cleanly(self) -> None:
        self.assertEqual(self._drain([], depth=3), [])


if __name__ == "__main__":
    unittest.main()
