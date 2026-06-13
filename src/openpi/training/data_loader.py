from collections.abc import Iterator, Sequence
import dataclasses
import json
import logging
import multiprocessing
import os
import pathlib
import typing
from typing import Literal, Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import pyarrow.parquet as pq
import torch

import openpi.models.model as _model
import openpi.training.config as _config
from openpi.training.droid_rlds_dataset import DroidRldsDataset
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)
TrainingBatch = tuple[_model.Observation, _model.Actions] | tuple[_model.Observation, _model.Actions, jax.Array]


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


class PreserveKeysDataset(Dataset[T_co]):
    def __init__(
        self,
        dataset: Dataset,
        transforms: Sequence[_transforms.DataTransformFn],
        preserved_keys: Sequence[str],
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._preserved_keys = tuple(preserved_keys)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        item = self._dataset[index]
        preserved = {key: item[key] for key in self._preserved_keys if key in item}
        transformed = self._transform(item)
        return {**transformed, **preserved}

    def __len__(self) -> int:
        return len(self._dataset)


class IterableTransformedDataset(IterableDataset[T_co]):
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:
                # Transforms are designed to be applied to individual samples. So we need to split the batch into
                # individual samples and apply the transform to each sample individually.
                batch_size = next(v.shape[0] for v in sample.values())

                # Split batch into individual samples using tree_map
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)]  # noqa: B023

                # Transform each sample
                transformed = [self._transform(s) for s in individual_samples]

                # Recombine batch with tree_map
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


_DEMO_LABEL = "demo"
_ROLLOUT_LABEL = "rollout"
_INTERVENTION_LABEL = "intervention"
_PRE_INTERVENTION_LABEL = "pre_intervention"


def compute_hil_labels(
    commander_states: Sequence[str], *, pre_intervention_frames: int
) -> np.ndarray:
    labels = np.full(len(commander_states), _ROLLOUT_LABEL, dtype=object)
    states = np.asarray(commander_states)
    intervention_mask = states == "teleop"
    labels[intervention_mask] = _INTERVENTION_LABEL

    intervention_starts = np.flatnonzero(intervention_mask & np.concatenate([[True], ~intervention_mask[:-1]]))
    for start in intervention_starts:
        pre_start = max(0, start - pre_intervention_frames)
        pre_indices = np.arange(pre_start, start)
        pre_indices = pre_indices[states[pre_indices] == "inference"]
        labels[pre_indices] = _PRE_INTERVENTION_LABEL

    return labels


def labels_to_weights(labels: np.ndarray, weighted_bc: _config.WeightedBCConfig) -> np.ndarray:
    weight_map = {
        _DEMO_LABEL: weighted_bc.demo_weight,
        _ROLLOUT_LABEL: weighted_bc.rollout_weight,
        _INTERVENTION_LABEL: weighted_bc.intervention_weight,
        _PRE_INTERVENTION_LABEL: weighted_bc.pre_intervention_weight,
    }
    return np.asarray([weight_map.get(label, weighted_bc.rollout_weight) for label in labels], dtype=np.float32)


def has_intervention_transition(labels: np.ndarray) -> bool:
    return bool(np.any((labels[:-1] != _INTERVENTION_LABEL) & (labels[1:] == _INTERVENTION_LABEL)))


@dataclasses.dataclass(frozen=True)
class _WeightedSource:
    dataset: Dataset
    valid_indices: np.ndarray
    loss_weights: np.ndarray


class ChallengeWeightedBCDataset(Dataset):
    def __init__(
        self,
        sources: Sequence[_config.WeightedBCSourceConfig],
        weighted_bc: _config.WeightedBCConfig,
        action_horizon: int,
        action_sequence_keys: Sequence[str],
        *,
        prompt_from_task: bool,
        video_backend: str | None,
    ):
        self._sources = [
            self._load_source(
                source,
                weighted_bc,
                action_horizon,
                action_sequence_keys,
                prompt_from_task=prompt_from_task,
                video_backend=video_backend,
            )
            for source in sources
        ]
        self._cumulative_sizes = np.cumsum([len(source.valid_indices) for source in self._sources])
        if len(self) == 0:
            raise ValueError("Weighted BC dataset has no valid samples.")

    def _load_source(
        self,
        source: _config.WeightedBCSourceConfig,
        weighted_bc: _config.WeightedBCConfig,
        action_horizon: int,
        action_sequence_keys: Sequence[str],
        *,
        prompt_from_task: bool,
        video_backend: str | None,
    ) -> _WeightedSource:
        dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(source.repo_id, root=source.local_files_path)
        dataset = lerobot_dataset.LeRobotDataset(
            source.repo_id,
            delta_timestamps={
                key: [t / dataset_meta.fps for t in range(action_horizon)] for key in action_sequence_keys
            },
            root=source.local_files_path,
            video_backend=video_backend,
        )
        if prompt_from_task:
            dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])

        labels_by_episode = self._load_labels_by_episode(source, weighted_bc)
        weights = []
        valid_indices = []
        global_index = 0
        for episode in _read_episodes(pathlib.Path(source.local_files_path)):
            episode_index = int(episode["episode_index"])
            episode_length = int(episode["length"])
            episode_labels = labels_by_episode[episode_index]
            for frame_index in range(episode_length):
                labels = episode_labels[frame_index : frame_index + action_horizon]
                if len(labels) < action_horizon:
                    labels = np.pad(labels, (0, action_horizon - len(labels)), mode="edge")
                if weighted_bc.drop_transition_chunks and has_intervention_transition(labels):
                    continue
                valid_indices.append(global_index + frame_index)
                weights.append(labels_to_weights(labels, weighted_bc))
            global_index += episode_length

        return _WeightedSource(
            dataset=dataset,
            valid_indices=np.asarray(valid_indices, dtype=np.int64),
            loss_weights=np.asarray(weights, dtype=np.float32),
        )

    def _load_labels_by_episode(
        self, source: _config.WeightedBCSourceConfig, weighted_bc: _config.WeightedBCConfig
    ) -> dict[int, np.ndarray]:
        root = pathlib.Path(source.local_files_path)
        labels_by_episode = {}
        for episode in _read_episodes(root):
            episode_index = int(episode["episode_index"])
            if source.source_type == "demo":
                labels_by_episode[episode_index] = np.full(int(episode["length"]), _DEMO_LABEL, dtype=object)
                continue

            parquet_path = episode_parquet_path(root, episode_index)
            table = pq.read_table(parquet_path, columns=[weighted_bc.mode_key])
            commander_states = table[weighted_bc.mode_key].to_pylist()
            labels_by_episode[episode_index] = compute_hil_labels(
                commander_states, pre_intervention_frames=weighted_bc.pre_intervention_frames
            )

        return labels_by_episode

    def __getitem__(self, index: SupportsIndex) -> dict:
        index = index.__index__()
        source_index = int(np.searchsorted(self._cumulative_sizes, index, side="right"))
        source_start = 0 if source_index == 0 else self._cumulative_sizes[source_index - 1]
        index_in_source = index - source_start
        source = self._sources[source_index]

        item = dict(source.dataset[int(source.valid_indices[index_in_source])])
        item["loss_weights"] = source.loss_weights[index_in_source]
        return item

    def __len__(self) -> int:
        return int(self._cumulative_sizes[-1])


def _read_episodes(root: pathlib.Path) -> list[dict]:
    episodes_path = root / "meta" / "episodes.jsonl"
    with episodes_path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def episode_parquet_path(root: str | pathlib.Path, episode_index: int) -> pathlib.Path:
    root = pathlib.Path(root)
    info_path = root / "meta" / "info.json"
    if info_path.is_file():
        with info_path.open() as f:
            info = json.load(f)
        data_path = info.get("data_path")
        if data_path is not None:
            chunks_size = int(info.get("chunks_size", 1000))
            episode_chunk = episode_index // chunks_size
            parquet_path = root / data_path.format(episode_chunk=episode_chunk, episode_index=episode_index)
            if parquet_path.is_file():
                return parquet_path

    parquet_name = f"episode_{episode_index:06d}.parquet"
    matches = sorted((root / "data").glob(f"chunk-*/{parquet_name}"))
    if matches:
        return matches[0]

    return root / "data" / "chunk-000" / parquet_name


def create_torch_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    model_config: _model.BaseModelConfig,
    *,
    weighted_bc: _config.WeightedBCConfig | None = None,
) -> Dataset:
    """Create a dataset for training."""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)
    if weighted_bc is not None and weighted_bc.enabled:
        if not data_config.weighted_bc_sources:
            raise ValueError("Weighted BC is enabled but data_config.weighted_bc_sources is empty.")
        return ChallengeWeightedBCDataset(
            data_config.weighted_bc_sources,
            weighted_bc,
            action_horizon,
            data_config.action_sequence_keys,
            prompt_from_task=data_config.prompt_from_task,
            video_backend=data_config.video_backend,
        )

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id, root=data_config.local_files_path)
    dataset = lerobot_dataset.LeRobotDataset(
        data_config.repo_id,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
        },
        root=data_config.local_files_path,
        video_backend=data_config.video_backend,
    )

    if data_config.prompt_from_task:
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])

    return dataset


def create_rlds_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    shuffle: bool = False,
) -> Dataset:
    # At the moment, we only support DROID for RLDS datasets.
    return DroidRldsDataset(
        data_dir=data_config.rlds_data_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        action_chunk_size=action_horizon,
        action_space=data_config.action_space,
        filter_dict_path=data_config.filter_dict_path,
    )


def transform_dataset(
    dataset: Dataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    preserve_loss_weights: bool = False,
) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    transforms = [
        *data_config.repack_transforms.inputs,
        *data_config.data_transforms.inputs,
        _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.model_transforms.inputs,
    ]
    if preserve_loss_weights:
        return PreserveKeysDataset(dataset, transforms, preserved_keys=("loss_weights",))
    return TransformedDataset(dataset, transforms)


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        is_batched=is_batched,
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
) -> DataLoader[TrainingBatch]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader (JAX only).
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return.
        skip_norm_stats: Whether to skip data normalization.
        framework: The framework to use ("jax" or "pytorch").
    """
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"data_config: {data_config}")

    if data_config.rlds_data_dir is not None:
        return create_rlds_data_loader(
            data_config,
            action_horizon=config.model.action_horizon,
            batch_size=config.batch_size,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
            framework=framework,
        )
    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
        weighted_bc=config.weighted_bc,
    )


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    framework: str = "jax",
    weighted_bc: _config.WeightedBCConfig | None = None,
) -> DataLoader[TrainingBatch]:
    """Create a data loader for training.

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
        seed: The seed to use for shuffling the data.
    """
    weighted_bc = weighted_bc or _config.WeightedBCConfig()
    dataset = create_torch_dataset(data_config, action_horizon, model_config, weighted_bc=weighted_bc)
    dataset = transform_dataset(
        dataset,
        data_config,
        skip_norm_stats=skip_norm_stats,
        preserve_loss_weights=weighted_bc.enabled,
    )

    # Use TorchDataLoader for both frameworks
    # For PyTorch DDP, create DistributedSampler and divide batch size by world size
    # For JAX, divide by process count
    sampler = None
    if framework == "pytorch":
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=shuffle,
                drop_last=True,
            )
            local_batch_size = batch_size // torch.distributed.get_world_size()
        else:
            local_batch_size = batch_size
    else:
        local_batch_size = batch_size // jax.process_count()

    logging.info(f"local_batch_size: {local_batch_size}")
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
    )

    return DataLoaderImpl(data_config, data_loader)


def create_rlds_data_loader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    framework: str = "jax",
) -> DataLoader[TrainingBatch]:
    """Create an RLDS data loader for training.

    Note: This data loader requires some extra dependencies -- see examples/droid/README_train.md

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
    """
    if framework == "pytorch":
        raise NotImplementedError("PyTorch RLDS data loader is not supported yet")
    dataset = create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=shuffle)
    dataset = transform_iterable_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats, is_batched=True)

    data_loader = RLDSDataLoader(
        dataset,
        sharding=sharding,
        num_batches=num_batches,
    )

    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:
    """Torch data loader implementation."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        framework: str = "jax",
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        # Store sharding - None for PyTorch, JAX sharding for JAX
        self._sharding = sharding
        if sharding is None and framework == "jax":
            # Use data parallel sharding by default for JAX only.
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
            sampler=sampler,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                # For JAX, convert to sharded arrays; for PyTorch, return torch tensors
                if self._sharding is not None:
                    yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
                else:
                    yield jax.tree.map(torch.as_tensor, batch)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


class RLDSDataLoader:
    """Shallow wrapper around the DROID data loader to make it compatible with openpi.

    All batching already happens in the DROID dataset, so we don't need to do anything here.
    """

    def __init__(
        self,
        dataset: DroidRldsDataset,
        *,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
    ):
        self._dataset = dataset
        self._num_batches = num_batches

        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._dataset)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader | RLDSDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            batch = dict(batch)
            actions = batch["actions"]
            loss_weights = batch.pop("loss_weights", None)
            observation = _model.Observation.from_dict(batch)
            if loss_weights is None:
                yield observation, actions
            else:
                yield observation, actions, loss_weights
