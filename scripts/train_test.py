import dataclasses
import os
import pathlib

import jax.numpy as jnp
import numpy as np
import pytest

os.environ["JAX_PLATFORMS"] = "cpu"

from openpi.training import config as _config

from . import train


def test_reduce_chunked_loss_unweighted_and_weighted():
    chunked_loss = jnp.asarray([[1.0, 3.0], [5.0, 7.0]])
    loss_weights = jnp.asarray([[1.0, 1.0], [0.0, 2.0]])

    np.testing.assert_allclose(train._reduce_chunked_loss(chunked_loss, None), 4.0)  # noqa: SLF001
    np.testing.assert_allclose(
        train._reduce_chunked_loss(chunked_loss, loss_weights, batch_normalize_weights=False), 4.5  # noqa: SLF001
    )
    np.testing.assert_allclose(
        train._reduce_chunked_loss(chunked_loss, loss_weights, batch_normalize_weights=True), 4.5  # noqa: SLF001
    )


def test_reduce_chunked_loss_with_weights():
    chunked_loss = jnp.asarray([[1.0, 3.0], [5.0, 7.0]])
    weights = jnp.asarray([[2.0, 2.0], [0.0, 4.0]])

    unweighted = train._reduce_chunked_loss(chunked_loss, None)  # noqa: SLF001
    weighted = train._reduce_chunked_loss(chunked_loss, weights, batch_normalize_weights=False)  # noqa: SLF001
    normalized_weighted = train._reduce_chunked_loss(
        chunked_loss, weights, batch_normalize_weights=True
    )  # noqa: SLF001

    assert float(unweighted) == 4.0
    assert float(weighted) == 9.0
    assert float(normalized_weighted) == 4.5


@pytest.mark.parametrize("config_name", ["debug"])
def test_train(tmp_path: pathlib.Path, config_name: str):
    config = dataclasses.replace(
        _config._CONFIGS_DICT[config_name],  # noqa: SLF001
        batch_size=2,
        checkpoint_base_dir=str(tmp_path / "checkpoint"),
        exp_name="test",
        overwrite=False,
        resume=False,
        num_train_steps=2,
        log_interval=1,
    )
    train.main(config)

    # test resuming
    config = dataclasses.replace(config, resume=True, num_train_steps=4)
    train.main(config)
