import math

import torch
import torch.nn as nn
from einops import einsum, reduce


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
        # NOTE(einsum): torch.einsum("...i,oi->...o", x, self.W)
        # NOTE(einops): einsum(x, self.W, "... in_features, out_features in_features -> ... out_features")
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


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.gain = nn.Parameter(torch.ones((d_model,), device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        RMSNorm is layer normalization

        Let x in R^d_model
        RMSNorm_x[i] = gain[i] * x[i] / (rms(x) + eps)
        where rms(x) = sqrt(sum(x**2) / d_model)

        x.shape = (batch_size, seq_length, d_model)
        out.shape = (batch_size, seq_length, d_model)
        """

        in_dtype = x.dtype
        x = x.to(torch.float32)

        # NOTE(einops): reduce(x**2, "batch_size seq_length d_model -> batch_size seq_length 1", "sum")
        # NOTE(einsum): torch.einsum("bsd -> bs", x**2).unsqueeze(-1)

        rms = torch.sqrt(torch.sum(x**2, dim=2, keepdim=True) / self.d_model + self.eps)
        result = x / rms * self.gain

        return result.to(in_dtype)


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        d_ff_approx = round((8 * d_model // 3) / 64) * 64
        self.d_ff = d_ff or d_ff_approx
        self.W_up = nn.Parameter(torch.empty(self.d_ff, self.d_model, device=device, dtype=dtype))
        self.W_down = nn.Parameter(torch.empty(self.d_model, self.d_ff, device=device, dtype=dtype))
        self.W_gate = nn.Parameter(torch.empty(self.d_ff, self.d_model, device=device, dtype=dtype))
        sigma = math.sqrt(2 / (self.d_model + self.d_ff))
        nn.init.trunc_normal_(
            self.W_up,
            mean=0,
            std=sigma,
            a=-3 * sigma,
            b=3 * sigma,
        )
        nn.init.trunc_normal_(
            self.W_down,
            mean=0,
            std=sigma,
            a=-3 * sigma,
            b=3 * sigma,
        )
        nn.init.trunc_normal_(
            self.W_gate,
            mean=0,
            std=sigma,
            a=-3 * sigma,
            b=3 * sigma,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        up = x @ self.W_up.T
        gate = silu(x @ self.W_gate.T)
        return (up * gate) @ self.W_down.T
