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


class RetrievalEvaluationEpisodeCollator:
    """Select class-balanced top-K similar context donors for each evaluation sample."""

    def __init__(self, train_dataset: Dataset[Any], retrieval_k: int = 24) -> None:
        self.train_dataset = train_dataset
        self.retrieval_k = retrieval_k
        self.k_per_class = max(1, retrieval_k // 2)
        self._train_samples: list[tuple[torch.Tensor, torch.Tensor]] | None = None
        self._train_summaries: torch.Tensor | None = None

    @staticmethod
    def _compute_bag_summary(bag: torch.Tensor) -> torch.Tensor:
        bag_float = bag.float()
        mean = torch.nn.functional.normalize(bag_float.mean(dim=0, keepdim=True), dim=-1)
        centered = bag_float - bag_float.mean(dim=0, keepdim=True)
        spread = torch.nn.functional.normalize(
            torch.sqrt(centered.square().mean(dim=0, keepdim=True) + 1e-6), dim=-1
        )
        return torch.cat([mean, spread], dim=-1).squeeze(0)

    def _get_train_cache(self) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        if self._train_samples is None or self._train_summaries is None:
            samples = [self.train_dataset[i] for i in range(len(self.train_dataset))]
            self._train_samples = [(s[0], s[1]) for s in samples]
            summaries = torch.stack([self._compute_bag_summary(s[0]) for s in samples])
            self._train_summaries = summaries
        return self._train_samples, self._train_summaries

    def __call__(self, evaluation_samples: list[Any]):
        if not evaluation_samples:
            raise ValueError("An evaluation episode must contain at least one target sample.")

        train_samples, train_summaries = self._get_train_cache()
        train_y = default_collate([s[1] for s in train_samples])

        evaluation_x = [sample[0] for sample in evaluation_samples]
        evaluation_y = default_collate([sample[1] for sample in evaluation_samples])

        eval_summary = self._compute_bag_summary(evaluation_x[0])
        sims = torch.nn.functional.cosine_similarity(
            eval_summary.unsqueeze(0), train_summaries, dim=-1
        )

        selected_indices: list[int] = []
        observed_classes = torch.unique(train_y, sorted=True)
        for class_idx in observed_classes:
            class_mask = (train_y == class_idx)
            class_indices = torch.nonzero(class_mask, as_tuple=False).flatten()
            class_sims = sims[class_indices]
            k_for_this_class = min(self.k_per_class, class_indices.numel())
            top_k_in_class = torch.topk(class_sims, k=k_for_this_class).indices
            selected_indices.extend(class_indices[top_k_in_class].tolist())

        selected_train_x = [train_samples[i][0] for i in selected_indices]
        selected_train_y = train_y[selected_indices]

        x = selected_train_x + evaluation_x
        y = torch.cat((selected_train_y, evaluation_y), dim=0)
        mask_index = torch.arange(
            len(selected_train_x),
            len(selected_train_x) + len(evaluation_samples),
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


class RetrievalSyntheticTrainingEpisodeCollator:
    """Subsample synthetic episode bags using Class-Balanced Top-K Retrieval."""

    def __init__(self, retrieval_k: int = 24) -> None:
        self.retrieval_k = retrieval_k
        self.k_per_class = max(1, retrieval_k // 2)

    @staticmethod
    def _compute_bag_summary(bag: torch.Tensor) -> torch.Tensor:
        bag_float = bag.float()
        mean = torch.nn.functional.normalize(bag_float.mean(dim=0, keepdim=True), dim=-1)
        centered = bag_float - bag_float.mean(dim=0, keepdim=True)
        spread = torch.nn.functional.normalize(
            torch.sqrt(centered.square().mean(dim=0, keepdim=True) + 1e-6), dim=-1
        )
        return torch.cat([mean, spread], dim=-1).squeeze(0)

    def __call__(self, samples: list[Any]):
        episode = collate_synthetic_training_episode(samples)
        x_full, y_full = episode[0], episode[1]

        if x_full.dim() == 4:
            res_x, res_y, res_masks = [], [], []
            for b in range(x_full.shape[0]):
                x_b, y_b = x_full[b], y_full[b]
                x_ret, y_ret, mask_idx = self._retrieve_single(x_b, y_b)
                res_x.append(x_ret)
                res_y.append(y_ret)
                res_masks.append(mask_idx)
            return torch.stack(res_x), torch.stack(res_y), torch.stack(res_masks), *episode[2:]
        else:
            x_ret, y_ret, mask_idx = self._retrieve_single(x_full, y_full)
            return x_ret, y_ret, mask_idx, *episode[2:]

    def _retrieve_single(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = x.device
        num_bags = x.shape[0]
        summaries = torch.stack([self._compute_bag_summary(x[i]) for i in range(num_bags)])

        observed_classes = torch.unique(y, sorted=True)
        protected = []
        for class_index in observed_classes:
            class_members = torch.nonzero(y == class_index, as_tuple=False).flatten()
            protected.append(class_members[0])

        can_query = torch.ones(num_bags, dtype=torch.bool, device=device)
        can_query[torch.stack(protected)] = False
        candidates = torch.nonzero(can_query, as_tuple=False).flatten()
        
        # 1 Query donor per 24 Context donors (24 Context + 1 Query = 25 Bags Total)
        query_indices = candidates[:1] if candidates.numel() > 0 else torch.tensor([0], device=device)

        is_context = torch.ones(num_bags, dtype=torch.bool, device=device)
        is_context[query_indices] = False
        context_indices = torch.nonzero(is_context, as_tuple=False).flatten()

        query_summary = summaries[query_indices].mean(dim=0)
        context_summaries = summaries[context_indices]
        context_y = y[context_indices]

        sims = torch.nn.functional.cosine_similarity(
            query_summary.unsqueeze(0), context_summaries, dim=-1
        )

        selected_context_idx: list[int] = []
        for class_idx in observed_classes:
            class_mask = (context_y == class_idx)
            class_idxs_in_context = torch.nonzero(class_mask, as_tuple=False).flatten()
            if class_idxs_in_context.numel() == 0:
                continue
            class_sims = sims[class_idxs_in_context]
            k_for_class = min(self.k_per_class, class_idxs_in_context.numel())
            top_k_local = torch.topk(class_sims, k=k_for_class).indices
            selected_context_idx.extend(context_indices[class_idxs_in_context[top_k_local]].cpu().tolist())

        if len(selected_context_idx) < self.retrieval_k and context_indices.numel() > len(selected_context_idx):
            remaining = [idx.item() for idx in context_indices if idx.item() not in selected_context_idx]
            needed = self.retrieval_k - len(selected_context_idx)
            selected_context_idx.extend(remaining[:needed])

        selected_context_tensor = torch.tensor(selected_context_idx, dtype=torch.long, device=device)

        final_x = torch.cat([x[selected_context_tensor], x[query_indices]], dim=0)
        final_y = torch.cat([y[selected_context_tensor], y[query_indices]], dim=0)
        mask_index = torch.tensor([len(selected_context_idx)], dtype=torch.long, device=device)
        return final_x, final_y, mask_index


class SignalAwarePretrainEpisodeCollator:
    """Pass large candidate pool (N=60~120+) bags to model for internal 40-dim Signal-Aware Retrieval."""

    def __init__(self, retrieval_k: int = 24) -> None:
        self.retrieval_k = retrieval_k

    def __call__(self, samples: list[Any]):
        episode = collate_synthetic_training_episode(samples)
        x_full, y_full = episode[0], episode[1]

        if x_full.dim() == 4:
            res_masks = []
            for b in range(x_full.shape[0]):
                num_bags = x_full.shape[1]
                # Default query is the last bag
                mask_idx = torch.tensor([num_bags - 1], dtype=torch.long, device=x_full.device)
                res_masks.append(mask_idx)
            return x_full, y_full, torch.stack(res_masks), *episode[2:]
        else:
            num_bags = x_full.shape[0]
            mask_idx = torch.tensor([num_bags - 1], dtype=torch.long, device=x_full.device)
            return x_full, y_full, mask_idx, *episode[2:]



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
            retrieval_k = self.hparams.get("retrieval_k", 0)
            use_signal_aware = self.hparams.get("use_signal_aware_retrieval", False)
            if use_signal_aware and retrieval_k > 0:
                collate_fn = SignalAwarePretrainEpisodeCollator(retrieval_k=retrieval_k)
            elif retrieval_k > 0:
                collate_fn = RetrievalSyntheticTrainingEpisodeCollator(retrieval_k=retrieval_k)
            else:
                collate_fn = collate_synthetic_training_episode
            return self._episode_dataloader(
                self.train_dataset,
                "train",
                collate_fn,
            )
        shuffle: bool = self.hparams.get("train_shuffle", self.hparams.get("shuffle", True))
        return self._dataloader(
            self.train_dataset,
            shuffle=shuffle,
            batch_size=len(self.train_dataset),
        )

    def val_dataloader(self) -> DataLoader[Any]:
        if self.hparams.get("episode_dataset", False):
            retrieval_k = self.hparams.get("retrieval_k", 0)
            use_signal_aware = self.hparams.get("use_signal_aware_retrieval", False)
            if use_signal_aware and retrieval_k > 0:
                collate_fn = SignalAwarePretrainEpisodeCollator(retrieval_k=retrieval_k)
            elif retrieval_k > 0:
                collate_fn = RetrievalSyntheticTrainingEpisodeCollator(retrieval_k=retrieval_k)
            else:
                collate_fn = collate_synthetic_evaluation_episode
            return self._episode_dataloader(
                self.val_dataset,
                "val",
                collate_fn,
            )
        shuffle: bool = self.hparams.get("val_shuffle", False)
        if shuffle:
            raise ValueError("val_shuffle must be false when building validation episodes.")
        retrieval_k = self.hparams.get("retrieval_k", 0)
        val_collator = (
            RetrievalEvaluationEpisodeCollator(self.train_dataset, retrieval_k=retrieval_k)
            if retrieval_k > 0
            else EvaluationEpisodeCollator(self.train_dataset)
        )
        test_collator = (
            RetrievalEvaluationEpisodeCollator(self.train_dataset, retrieval_k=retrieval_k)
            if retrieval_k > 0
            else EvaluationEpisodeCollator(self.train_dataset)
        )

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
        retrieval_k = self.hparams.get("retrieval_k", 0)
        test_collator = (
            RetrievalEvaluationEpisodeCollator(self.train_dataset, retrieval_k=retrieval_k)
            if retrieval_k > 0
            else EvaluationEpisodeCollator(self.train_dataset)
        )
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
        num_workers: int = self.hparams.get("num_workers", 0)
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
        return loader_cls(
            dataset,
            batch_size=batch_size,
            shuffle=self.hparams.get(f"{split}_shuffle", split == "train"),
            num_workers=num_workers,
            pin_memory=self.hparams.get("pin_memory", True),
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
