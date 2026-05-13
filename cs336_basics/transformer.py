import math

import torch
import torch.nn as nn
from einops import einsum


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.W = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        sigma = math.sqrt(2 / (in_features + out_features))
        nn.init.trunc_normal_(self.W, mean=0, std=sigma, a=-3 * sigma, b=3 * sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # return torch.einsum("...i,oi->...o", x, self.W)
        # return einsum(x, self.W, "... in_features, out_features in_features -> ... out_features")
        return x @ self.W.T


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,  # vocab_size
        embedding_dim: int,  # d_model
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.E = nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))
        nn.init.trunc_normal_(self.E, mean=0, std=1, a=-3, b=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # E[x] -- where x is [B, T]
        return self.E[x]
