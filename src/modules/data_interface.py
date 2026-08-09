from __future__ import annotations
from importlib import import_module
from concurrent.futures import ThreadPoolExecutor
from typing import Any
import torch
from collections import deque

from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate
from lightning import LightningDataModule


class EvaluationEpisodeCollator:
    """Prepend the full training set and mark evaluation positions for masking."""

    def __init__(self, train_dataset: Dataset[Any]) -> None:
        self.train_dataset = train_dataset
        self._train_batch: tuple[list[torch.Tensor], torch.Tensor] | None = None

    def _get_train_batch(self) -> tuple[list[torch.Tensor], torch.Tensor]:
        if self._train_batch is None:
            train_samples = [
                self.train_dataset[index]
                for index in range(len(self.train_dataset))
            ]
            train_x = [sample[0] for sample in train_samples]
            train_y = default_collate([sample[1] for sample in train_samples])
            self._train_batch = (train_x, train_y)
        return self._train_batch

    def __call__(self, evaluation_samples: list[Any]):
        if not evaluation_samples:
            raise ValueError("An evaluation episode must contain at least one target sample.")

        train_x, train_y = self._get_train_batch()
        evaluation_x = [sample[0] for sample in evaluation_samples]
        evaluation_y = default_collate([sample[1] for sample in evaluation_samples])

        x = train_x + evaluation_x
        y = torch.cat((train_y, evaluation_y), dim=0)
        mask_index = torch.arange(
            len(train_x),
            len(train_x) + len(evaluation_samples),
            dtype=torch.long,
        )
        return x, y, mask_index


def _collate_ragged_batch(samples: list[Any]):
    """Pad ragged (B2b, per-bag-cardinality) episodes into a dense batch.

    Returns ``(x, y, cell_mask, bag_mask)`` with shapes
    [episodes, bags, instances, dim], [episodes, bags],
    [episodes, bags, instances], [episodes, bags]. ``cell_mask`` marks real
    cells, ``bag_mask`` marks real bags, and padded bags get label -1 so they
    are never sampled as queries or treated as context.
    """
    first_bag = samples[0][0][0]
    device = first_bag.device
    dtype = first_bag.dtype
    label_dtype = samples[0][1].dtype
    episodes = len(samples)
    num_bags = max(len(sample[0]) for sample in samples)
    num_instances = max(
        bag.shape[0] for sample in samples for bag in sample[0]
    )
    dim = first_bag.shape[-1]
    x = torch.zeros(
        (episodes, num_bags, num_instances, dim),
        dtype=dtype,
        device=device,
    )
    cell_mask = torch.zeros(
        (episodes, num_bags, num_instances), dtype=torch.bool, device=device
    )
    bag_mask = torch.zeros((episodes, num_bags), dtype=torch.bool, device=device)
    y = torch.full((episodes, num_bags), -1, dtype=label_dtype, device=device)
    for episode_index, sample in enumerate(samples):
        bags, labels = sample[0], sample[1]
        if len(bags) != labels.numel():
            raise ValueError("Ragged episode labels must match bag count.")
        n_bags = len(bags)
        bag_mask[episode_index, :n_bags] = True
        y[episode_index, :n_bags] = labels
        for bag_index, bag in enumerate(bags):
            if bag.ndim != 2 or bag.shape[-1] != dim:
                raise ValueError("Ragged bag must be [instances, dim].")
            count = bag.shape[0]
            cell_mask[episode_index, bag_index, :count] = True
            x[episode_index, bag_index, :count] = bag
    return x, y, cell_mask, bag_mask


def collate_synthetic_training_episode(samples: list[Any]):
    """Stack equal-shape episodes, or pad ragged (B2b) episodes.

    Equal-shape (B2) episodes stack into a dense [episodes, bags, cells, dim]
    batch. Ragged per-bag-cardinality (B2b) episodes are padded to the batch's
    max shape and returned as ``(x, y, cell_mask, bag_mask)``, which enables
    ``episode_batch_size > 1`` and unlocks batched vectorized training.
    """
    if not samples:
        raise ValueError("A synthetic training batch cannot be empty.")
    if len(samples) == 1:
        return samples[0]
    if not isinstance(samples[0][0], torch.Tensor):
        return _collate_ragged_batch(samples)
    x = torch.stack([sample[0] for sample in samples])
    y = torch.stack([sample[1] for sample in samples])
    field_count = len(samples[0])
    if any(len(sample) != field_count for sample in samples):
        raise ValueError("Synthetic episode samples must have matching fields.")
    if field_count == 2:
        return x, y
    extras = tuple(
        torch.stack([sample[field] for sample in samples])
        for field in range(2, field_count)
    )
    return x, y, *extras


class _CudaPrefetchIterator:
    """Generate upcoming CUDA batches while the current batch trains.

    `depth` batches are kept in flight on one background CUDA stream. Depth 1
    is enough to hide generation when generation is the SHORTER leg; it buys
    nothing in steady state once generation is the longer one, because the
    single producer is then saturated. What deeper queues actually buy here is
    variance absorption: synthetic episodes span 20k to 950k cells, so a run of
    large episodes stalls a depth-1 queue even when the average would keep up.

    The cost is GPU memory -- a queued episode is up to ~5.9 GB at the top of
    that range -- which is why this is a knob and not simply raised.
    """

    def __init__(self, source: Any, depth: int = 1) -> None:
        self.source = source
        self.depth = max(1, int(depth))
        # One worker, so generation stays sequential on one stream: several
        # concurrent generators would interleave their kernels and multiply
        # peak memory rather than overlap usefully.
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.stream = torch.cuda.Stream()
        self.futures: deque[Any] = deque()
        self.exhausted = False
        self._fill()

    def _next_batch(self) -> Any:
        with torch.cuda.stream(self.stream):
            batch = next(self.source)
        self.stream.synchronize()
        return batch

    def _fill(self) -> None:
        while not self.exhausted and len(self.futures) < self.depth:
            self.futures.append(self.executor.submit(self._next_batch))

    def __iter__(self) -> "_CudaPrefetchIterator":
        return self

    def __next__(self) -> Any:
        if not self.futures:
            raise StopIteration
        try:
            batch = self.futures.popleft().result()
        except StopIteration:
            # A queued future hit the end of the source. Mark exhausted so no
            # further work is submitted, then drain what was already produced --
            # dropping it would silently shorten the epoch.
            self.exhausted = True
            return self.__next__()
        except BaseException:
            self.executor.shutdown(wait=False)
            raise
        self._fill()
        return batch

    def __del__(self) -> None:
        executor = getattr(self, "executor", None)
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)


class CudaPrefetchDataLoader(DataLoader[Any]):
    """DataLoader whose upcoming batches are produced on a background stream."""

    def __init__(self, *args: Any, prefetch_depth: int = 1, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.prefetch_depth = max(1, int(prefetch_depth))

    def __iter__(self) -> _CudaPrefetchIterator:
        return _CudaPrefetchIterator(super().__iter__(), depth=self.prefetch_depth)


def collate_synthetic_evaluation_episode(samples: list[Any]):
    """Create a deterministic context/query split for a synthetic episode.

    Evaluation must not expose one query's label as context for another query.
    Twenty percent of bags (at most 20) are queried together, while one bag
    from every observed class is protected as labelled context.
    """
    episode = collate_synthetic_training_episode(samples)
    x, y = episode[:2]
    num_bags = len(x) if not isinstance(x, torch.Tensor) else x.shape[0]
    observed_classes = torch.unique(y, sorted=True)
    protected = []
    for class_index in observed_classes:
        class_members = torch.nonzero(y == class_index, as_tuple=False).flatten()
        protected.append(class_members[0])
    can_query = torch.ones(num_bags, dtype=torch.bool)
    can_query[torch.stack(protected)] = False
    candidates = torch.nonzero(can_query, as_tuple=False).flatten()
    requested_queries = max(1, min(20, (num_bags + 4) // 5))
    num_queries = min(requested_queries, candidates.numel())
    if num_queries == 0:
        raise ValueError(
            "A synthetic evaluation episode needs context bags and a query."
        )
    mask_index = candidates[:num_queries]
    return x, y, mask_index, *episode[2:]


class DataInterface(LightningDataModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters()
    
    def setup(self, stage: str | None = None) -> None:
        dataset_src: str | None = self.hparams.get("dataset_src")
        if dataset_src is None:
            return

        dataset_cls = self._dataset_class(dataset_src)
        dataset_kwargs: dict[str, Any] = self.hparams.get("dataset_kwargs") or {}

        if self.hparams.get("episode_dataset", False):
            if stage in (None, "fit"):
                self.train_dataset = self._build_episode_dataset(
                    dataset_cls, "train", dataset_kwargs
                )
                self.val_dataset = self._build_episode_dataset(
                    dataset_cls, "val", dataset_kwargs
                )
            if stage in (None, "test"):
                self.test_dataset = self._build_episode_dataset(
                    dataset_cls, "test", dataset_kwargs
                )
            return

        if stage in (None, "fit"):
            self.train_dataset = self._build_dataset(dataset_cls, "train", dataset_kwargs)
            self.val_dataset = self._build_dataset(dataset_cls, "val", dataset_kwargs)
        if stage in (None, "test"):
            if not hasattr(self, "train_dataset"):
                self.train_dataset = self._build_dataset(dataset_cls, "train", dataset_kwargs)
            self.test_dataset = self._build_dataset(
                dataset_cls,
                "test",
                dataset_kwargs,
                state="external",
            )

    def train_dataloader(self) -> DataLoader[Any]:
        if self.hparams.get("episode_dataset", False):
            return self._episode_dataloader(
                self.train_dataset,
                "train",
                collate_synthetic_training_episode,
            )
        shuffle: bool = self.hparams.get("train_shuffle", self.hparams.get("shuffle", True))
        return self._dataloader(
            self.train_dataset,
            shuffle=shuffle,
            batch_size=len(self.train_dataset),
        )

    def val_dataloader(self) -> DataLoader[Any]:
        if self.hparams.get("episode_dataset", False):
            return self._episode_dataloader(
                self.val_dataset,
                "val",
                collate_synthetic_evaluation_episode,
            )
        shuffle: bool = self.hparams.get("val_shuffle", False)
        if shuffle:
            raise ValueError("val_shuffle must be false when building validation episodes.")
        val_collator = EvaluationEpisodeCollator(self.train_dataset)

        return DataLoader(
            self.val_dataset,
            batch_size=len(self.val_dataset),
            shuffle=False,
            num_workers=0,
            pin_memory=self.hparams.get("pin_memory", False),
            drop_last=False,
            persistent_workers=False,
            collate_fn=val_collator,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        if self.hparams.get("episode_dataset", False):
            return self._episode_dataloader(
                self.test_dataset,
                "test",
                collate_synthetic_evaluation_episode,
            )
        shuffle: bool = self.hparams.get("test_shuffle", False)
        if shuffle:
            raise ValueError("test_shuffle must be false when building test episodes.")
        test_collator = EvaluationEpisodeCollator(self.train_dataset)
        return DataLoader(
            self.test_dataset,
            batch_size=len(self.test_dataset),
            shuffle=False,
            num_workers=0,
            pin_memory=self.hparams.get("pin_memory", False),
            drop_last=False,
            persistent_workers=False,
            collate_fn=test_collator,
        )

    def _dataloader(
        self,
        dataset: Dataset[Any],
        shuffle: bool,
        batch_size: int | None = None,
    ) -> DataLoader[Any]:
        if batch_size is None:
            batch_size = self.hparams.get("batch_size", 1)
        num_workers: int = self.hparams.get("num_workers", 0)
        pin_memory: bool = self.hparams.get("pin_memory", False)
        drop_last: bool = self.hparams.get("drop_last", False)
        persistent_workers: bool = self.hparams.get("persistent_workers", False) and num_workers > 0

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            persistent_workers=persistent_workers,
        )

    def _episode_dataloader(
        self,
        dataset: Dataset[Any],
        split: str,
        collate_fn: Any,
    ) -> DataLoader[Any]:
        # A dataset that generates directly on CUDA (dataset_kwargs
        # generation_device="cuda") gains nothing from multiprocessing
        # workers -- there is still only one physical GPU -- while each
        # worker adds its own CUDA context and its own prefetched batches,
        # which multiplies GPU memory usage (observed: 4 workers alone held
        # 100+ GiB) and starves the main process's forward/backward pass.
        # The existing CudaPrefetchDataLoader already overlaps generation
        # with training via an in-process background CUDA stream, so
        # multi-process workers are only used for CPU-side generation.
        generates_on_cuda = torch.device(
            getattr(dataset, "generation_device", "cpu")
        ).type == "cuda"
        num_workers: int = 0 if generates_on_cuda else self.hparams.get("num_workers", 0)
        persistent_workers = (
            self.hparams.get("persistent_workers", False) and num_workers > 0
        )
        batch_size = self.hparams.get("episode_batch_size", 1) if split == "train" else 1
        use_prefetch = self.hparams.get("cuda_prefetch", True) and torch.cuda.is_available()
        loader_cls = (
            CudaPrefetchDataLoader
            if split == "train" and use_prefetch
            else DataLoader
        )
        # Worker processes that generate data on CUDA cannot fork from a
        # process that has already initialized a CUDA context --
        # torch.Generator(device="cuda") raises "CUDA error: initialization
        # error" inside a forked worker. "spawn" starts fresh interpreters
        # instead, avoiding the conflict. Moot when generates_on_cuda forces
        # num_workers=0 above, but kept for CPU-generating multi-worker runs.
        multiprocessing_context = "spawn" if num_workers > 0 else None
        # pin_memory only applies to dense CPU tensors; a CUDA-generating
        # dataset hands the loader CUDA tensors already, which pin_memory()
        # cannot pin.
        pin_memory = self.hparams.get("pin_memory", True) and not generates_on_cuda
        extra: dict[str, Any] = {}
        if loader_cls is CudaPrefetchDataLoader:
            # How many generated episodes may sit on the GPU waiting to train.
            # Depth 1 already hides generation behind the step; more only
            # absorbs the size variance of synthetic episodes, and each queued
            # episode costs its own GPU memory.
            extra["prefetch_depth"] = self.hparams.get("cuda_prefetch_depth", 1)
        return loader_cls(
            dataset,
            batch_size=batch_size,
            shuffle=self.hparams.get(f"{split}_shuffle", split == "train"),
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
            persistent_workers=persistent_workers,
            collate_fn=collate_fn,
            multiprocessing_context=multiprocessing_context,
            **extra,
        )

    def _dataset_class(self, dataset_src: str) -> type[Dataset[Any]]:
        import_path = dataset_src
        if not import_path.startswith("src.datasets."):
            import_path = f"src.datasets.{import_path}"

        module_name, class_name = import_path.rsplit(".", maxsplit=1)
        module = import_module(module_name)
        dataset_cls: type[Dataset[Any]] = getattr(module, class_name)
        return dataset_cls

    def _build_dataset(
        self,
        dataset_cls: type[Dataset[Any]],
        split: str,
        dataset_kwargs: dict[str, Any],
        state: str | None = None,
    ) -> Dataset[Any]:
        split_kwargs: dict[str, Any] = self.hparams.get(f"{split}_dataset_kwargs") or {}

        kwargs = dict(dataset_kwargs)
        kwargs.update(split_kwargs)
        if state is not None:
            kwargs["state"] = state
        else:
            kwargs.setdefault("state", split)
        
        return dataset_cls(**kwargs)

    def _build_episode_dataset(
        self,
        dataset_cls: type[Dataset[Any]],
        split: str,
        dataset_kwargs: dict[str, Any],
    ) -> Dataset[Any]:
        kwargs = dict(dataset_kwargs)
        kwargs.update(self.hparams.get(f"{split}_dataset_kwargs") or {})
        return dataset_cls(**kwargs)
