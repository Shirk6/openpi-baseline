"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

import copy
import pathlib

import numpy as np
import pyarrow.parquet as pq
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


class SkipVideoDecodingDataset(_data_loader.Dataset):
    """Avoid decoding video frames when computing stats for state/actions only."""

    _DUMMY_IMAGE = np.zeros((3, 1, 1), dtype=np.float32)

    def __init__(self, dataset: _data_loader.Dataset):
        self._dataset = dataset
        self._disable_video_decoding(dataset)

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index):
        item = dict(self._dataset[index])
        if "images" in item:
            item["images"] = {key: self._DUMMY_IMAGE for key in item["images"]}
        return item

    def _disable_video_decoding(self, dataset):
        if isinstance(dataset, _data_loader.TransformedDataset):
            self._disable_video_decoding(dataset._dataset)
            return

        sources = getattr(dataset, "_sources", None)
        if sources is not None:
            for source in sources:
                self._disable_video_decoding(source.dataset)
            return

        meta = getattr(dataset, "meta", None)
        if meta is None:
            return

        info = copy.deepcopy(meta.info)
        for key, feature in info.get("features", {}).items():
            if feature.get("dtype") == "video":
                feature["dtype"] = "image"
        meta.info = info



def _make_stats_item(state: np.ndarray, actions: np.ndarray, prompt: str = "") -> dict:
    dummy_image = SkipVideoDecodingDataset._DUMMY_IMAGE
    return {
        "observation.state": state,
        "action": actions,
        "task": prompt,
        "observation.images.cam_high": dummy_image,
        "observation.images.cam_left_wrist": dummy_image,
        "observation.images.cam_right_wrist": dummy_image,
    }


def _valid_frame_indices_for_episode(
    source: _config.WeightedBCSourceConfig,
    episode: dict,
    weighted_bc: _config.WeightedBCConfig,
    action_horizon: int,
) -> np.ndarray:
    episode_length = int(episode["length"])
    frame_indices = np.arange(episode_length, dtype=np.int64)
    if source.source_type == "demo":
        return frame_indices

    parquet_path = _data_loader.episode_parquet_path(source.local_files_path, int(episode["episode_index"]))
    table = pq.read_table(parquet_path, columns=[weighted_bc.mode_key])
    labels = _data_loader.compute_hil_labels(
        table[weighted_bc.mode_key].to_pylist(), pre_intervention_frames=weighted_bc.pre_intervention_frames
    )
    keep = np.ones(episode_length, dtype=bool)
    if weighted_bc.drop_transition_chunks:
        for frame_index in range(episode_length):
            chunk_labels = labels[frame_index : frame_index + action_horizon]
            if len(chunk_labels) < action_horizon:
                chunk_labels = np.pad(chunk_labels, (0, action_horizon - len(chunk_labels)), mode="edge")
            keep[frame_index] = not _data_loader.has_intervention_transition(chunk_labels)
    return frame_indices[keep]


def compute_weighted_bc_norm_stats_fast(
    config: _config.TrainConfig,
    data_config: _config.DataConfig,
    max_frames: int | None = None,
) -> dict[str, normalize.NormStats]:
    """Compute weighted-BC stats from parquet columns without decoding videos or building DataLoaders."""
    transforms_fn = transforms.compose(
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            RemoveStrings(),
        ]
    )
    stats = {key: normalize.RunningStats() for key in ("state", "actions")}
    total_seen = 0

    episode_refs = []
    for source in data_config.weighted_bc_sources:
        for episode in _data_loader._read_episodes(pathlib.Path(source.local_files_path)):
            episode_refs.append((source, episode))

    for source, episode in tqdm.tqdm(episode_refs, desc="Computing stats", unit="episode"):
        if max_frames is not None and total_seen >= max_frames:
            break

        episode_index = int(episode["episode_index"])
        parquet_path = _data_loader.episode_parquet_path(source.local_files_path, episode_index)
        table = pq.read_table(parquet_path, columns=["observation.state", "action"])
        states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        valid_indices = _valid_frame_indices_for_episode(source, episode, config.weighted_bc, config.model.action_horizon)

        if max_frames is not None:
            valid_indices = valid_indices[: max_frames - total_seen]
        if len(valid_indices) == 0:
            continue

        horizon_offsets = np.arange(config.model.action_horizon, dtype=np.int64)
        action_indices = np.minimum(valid_indices[:, None] + horizon_offsets[None, :], len(actions) - 1)
        batch_actions = actions[action_indices]
        batch_states = states[valid_indices]

        # Keep the same state/action transforms used during training, but avoid video and DataLoader overhead.
        transformed = [
            transforms_fn(_make_stats_item(state, action_seq)) for state, action_seq in zip(batch_states, batch_actions)
        ]
        stats["state"].update(np.stack([item["state"] for item in transformed], axis=0))
        stats["actions"].update(np.stack([item["actions"] for item in transformed], axis=0))
        total_seen += len(valid_indices)

    if total_seen == 0:
        raise ValueError("No frames were available for computing normalization stats.")
    return {key: value.get_statistics() for key, value in stats.items()}

def create_torch_dataloader(
    config: _config.TrainConfig,
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_torch_dataset(
        data_config, action_horizon, model_config, weighted_bc=config.weighted_bc
    )
    dataset = _data_loader.TransformedDataset(
        SkipVideoDecodingDataset(dataset),
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        # NOTE: this length is currently hard-coded for DROID.
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def main(config_name: str, max_frames: int | None = None):
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)

    if config.weighted_bc.enabled and data_config.weighted_bc_sources:
        norm_stats = compute_weighted_bc_norm_stats_fast(config, data_config, max_frames)
        output_path = config.assets_dirs / data_config.repo_id
        print(f"Writing stats to: {output_path}")
        normalize.save(output_path, norm_stats)
        return

    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size, max_frames
        )
    else:
        data_loader, num_batches = create_torch_dataloader(
            config,
            data_config,
            config.model.action_horizon,
            config.batch_size,
            config.model,
            config.num_workers,
            max_frames,
        )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    output_path = config.assets_dirs / data_config.repo_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
