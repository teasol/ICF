from __future__ import annotations
from importlib import import_module
from concurrent.futures import ThreadPoolExecutor
from typing import Any
import torch
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


def collate_synthetic_training_episode(samples: list[Any]):
    """Stack equal-shape synthetic episodes for single-device DDP emulation."""
    if not samples:
        raise ValueError("A synthetic training batch cannot be empty.")
    if len(samples) == 1:
        return samples[0]
    x = torch.stack([sample[0] for sample in samples])
    y = torch.stack([sample[1] for sample in samples])
    if len(samples[0]) == 2:
        return x, y
    if len(samples[0]) != 3 or any(len(sample) != 3 for sample in samples):
        raise ValueError("Synthetic episode samples must have two or three fields.")
    oracle_abundance = torch.stack([sample[2] for sample in samples])
    return x, y, oracle_abundance


class _CudaPrefetchIterator:
    """Generate the next CUDA batch while the current batch trains."""

    def __init__(self, source: Any) -> None:
        self.source = source
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.stream = torch.cuda.Stream()
        self.future = self.executor.submit(self._next_batch)

    def _next_batch(self) -> Any:
        with torch.cuda.stream(self.stream):
            batch = next(self.source)
        self.stream.synchronize()
        return batch

    def __iter__(self) -> "_CudaPrefetchIterator":
        return self

    def __next__(self) -> Any:
        try:
            batch = self.future.result()
        except BaseException:
            self.executor.shutdown(wait=False)
            raise
        self.future = self.executor.submit(self._next_batch)
        return batch

    def __del__(self) -> None:
        executor = getattr(self, "executor", None)
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)


class CudaPrefetchDataLoader(DataLoader[Any]):
    """DataLoader whose next batch is produced on a background CUDA stream."""

    def __iter__(self) -> _CudaPrefetchIterator:
        return _CudaPrefetchIterator(super().__iter__())


def collate_synthetic_evaluation_episode(samples: list[Any]):
    """Create a deterministic context/query split for a synthetic episode.

    Evaluation must not expose one query's label as context for another query.
    Twenty percent of bags (at most 20) are queried together, while one bag
    from every observed class is protected as labelled context.
    """
    episode = collate_synthetic_training_episode(samples)
    x, y = episode[:2]
    observed_classes = torch.unique(y, sorted=True)
    protected = []
    for class_index in observed_classes:
        class_members = torch.nonzero(y == class_index, as_tuple=False).flatten()
        protected.append(class_members[0])
    can_query = torch.ones(x.shape[0], dtype=torch.bool)
    can_query[torch.stack(protected)] = False
    candidates = torch.nonzero(can_query, as_tuple=False).flatten()
    requested_queries = max(1, min(20, (x.shape[0] + 4) // 5))
    num_queries = min(requested_queries, candidates.numel())
    if num_queries == 0:
        raise ValueError(
            "A synthetic evaluation episode needs context bags and a query."
        )
    mask_index = candidates[:num_queries]
    if len(episode) == 2:
        return x, y, mask_index
    return x, y, mask_index, episode[2]


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
        return DataLoader(
            self.val_dataset,
            batch_size=len(self.val_dataset),
            shuffle=False,
            num_workers=0,
            pin_memory=self.hparams.get("pin_memory", False),
            drop_last=False,
            persistent_workers=False,
            collate_fn=EvaluationEpisodeCollator(self.train_dataset),
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
        return DataLoader(
            self.test_dataset,
            batch_size=len(self.test_dataset),
            shuffle=False,
            num_workers=0,
            pin_memory=self.hparams.get("pin_memory", False),
            drop_last=False,
            persistent_workers=False,
            collate_fn=EvaluationEpisodeCollator(self.train_dataset),
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
        num_workers: int = self.hparams.get("num_workers", 0)
        persistent_workers = (
            self.hparams.get("persistent_workers", False) and num_workers > 0
        )
        batch_size = self.hparams.get("episode_batch_size", 1) if split == "train" else 1
        loader_cls = (
            CudaPrefetchDataLoader
            if split == "train" and self.hparams.get("cuda_prefetch", False)
            else DataLoader
        )
        return loader_cls(
            dataset,
            batch_size=batch_size,
            shuffle=self.hparams.get(f"{split}_shuffle", split == "train"),
            num_workers=num_workers,
            pin_memory=self.hparams.get("pin_memory", False),
            drop_last=False,
            persistent_workers=persistent_workers,
            collate_fn=collate_fn,
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
