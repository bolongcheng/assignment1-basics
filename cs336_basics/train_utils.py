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
    x = np.stack([dataset[i : i + context_length] for i in starts])
    y = np.stack([dataset[i + 1 : i + 1 + context_length] for i in starts])
    return (torch.tensor(x, dtype=torch.long, device=device), torch.tensor(y, dtype=torch.long, device=device))


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

    obj: Checkpoint = torch.load(src)
    model.load_state_dict(obj["model"])
    optimizer.load_state_dict(obj["optimizer"])

    return obj["iteration"]


def load_dataset(
    path: str | os.PathLike,
    dtype: np.dtype = np.uint32,
) -> npt.NDArray:
    return np.memmap(path, dtype=dtype, mode="r")
