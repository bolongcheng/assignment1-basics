import os
from typing import IO, Any, BinaryIO, TypedDict

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn


def load_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    data_len = len(dataset)
    rng = np.random.default_rng()
    starts = rng.integers(0, data_len - context_length, size=batch_size)
    x_np = np.stack([dataset[i : i + context_length] for i in starts]).astype(np.int64, copy=False)
    y_np = np.stack([dataset[i + 1 : i + 1 + context_length] for i in starts]).astype(np.int64, copy=False)
    x = torch.from_numpy(x_np)
    y = torch.from_numpy(y_np)
    if device == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)
    return x, y


class Checkpoint(TypedDict):
    iteration: int
    model: dict[str, Any]
    optimizer: dict[str, Any]


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    obj = Checkpoint(
        iteration=iteration,
        model=model.state_dict(),
        optimizer=optimizer.state_dict(),
    )
    torch.save(obj, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:

    obj: Checkpoint = torch.load(src, map_location="cpu")
    state = {k.replace("_orig_mod.", ""): v for k, v in obj["model"].items()}
    model.load_state_dict(state)
    optimizer.load_state_dict(obj["optimizer"])

    return obj["iteration"]


def load_dataset(
    path: str | os.PathLike,
    dtype: np.dtype = np.uint32,
) -> npt.NDArray:
    return np.memmap(path, dtype=dtype, mode="r")
