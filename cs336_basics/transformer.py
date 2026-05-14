from typing import Any
import math

import torch
import torch.nn as nn
from einops import einsum, reduce, rearrange


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.W = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        self._init_params()

    def _init_params(self) -> None:
        out_features, in_features = self.W.shape
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
    ) -> None:
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
    ) -> None:
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
    ) -> None:
        super().__init__()
        self.d_model = d_model
        # round up to nearest multiple of 64
        hidden_exp = int(8 * d_model / 3)
        d_ff_approx = ((hidden_exp + 63) // 64) * 64
        self.d_ff = d_ff or d_ff_approx
        self.W_up = Linear(self.d_model, self.d_ff, device=device, dtype=dtype)
        self.W_down = Linear(self.d_ff, self.d_model, device=device, dtype=dtype)
        self.W_gate = Linear(self.d_model, self.d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        up = self.W_up(x)
        gate = silu(self.W_gate(x))
        return self.W_down(up * gate)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.theta = theta
        if d_k % 2 != 0:
            raise ValueError(f"d_k needs to be divisible by 2, d_k={d_k}")
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        positions = torch.arange(max_seq_len, device=device)
        inv_freqs = 1 / (theta ** (torch.arange(0, d_k, step=2, device=device) / d_k))
        # thetas: (max_seq_len,  d_k / 2)
        # NOTE(einsum): thetas = torch.einsum("s,d->sd", positions, freqs)
        # NOTE(einops): thetas = einsum(positions, freqs, "max_seq_len, d_k_half -> max_seq_len d_k_half")
        thetas = positions[:, None] * inv_freqs[None, :]

        # (max_seq_len, d_k/2)
        self.register_buffer("sines", torch.sin(thetas), persistent=False)
        self.register_buffer("cosines", torch.cos(thetas), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None) -> torch.Tensor:
        # x: (..., seq_len, d_k)
        # token_positions: (..., seq_len)
        # out: (..., seq_len, d_k)
        x_even = x[..., ::2]  # (..., seq_len, d_k/2)
        x_odd = x[..., 1::2]
        if token_positions is None:
            token_positions = torch.arange(self.max_seq_len)
        out = torch.stack(
            [
                x_even * self.cosines[token_positions, :] - x_odd * self.sines[token_positions, :],
                x_even * self.sines[token_positions, :] + x_odd * self.cosines[token_positions, :],
            ],
            dim=-1,
        )

        # NOTE(einsum): rearrange(out, "... seq_len d_k_half pair-> ... seq_len (d_k_half pair)")
        return out.flatten(start_dim=-2)


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    max_val = x.max(dim=dim, keepdim=True).values
    exp_xs = (x - max_val).exp()
    return exp_xs / exp_xs.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    # Q, K: (batch_size, .., seq_len, d_k)
    # V: (batch_size, ..., seq_len, d_v)
    # mask: (seq_len, seq_len)
    d_k = Q.shape[-1]

    # NOTE(einops): einsum(Q, K, "batch_size ... seq_len_q d_k, batch_size ... seq_len_k d_k -> batch_size ... seq_len_q seq_len_k")
    # NOTE(einsum): torch.einsum("b...qd,b...kd -> b...qk", Q, K) / d_k**0.5
    wei = Q @ K.transpose(-2, -1) / d_k**0.5
    return softmax(wei.masked_fill(mask == 0, float("-inf")), dim=-1) @ V


class MultiheadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope_theta: float | None = None,
        rope_max_seq_len: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        self.W_q = Linear(in_features=d_model, out_features=d_model, device=device, dtype=dtype)
        self.W_k = Linear(in_features=d_model, out_features=d_model, device=device, dtype=dtype)
        self.W_v = Linear(in_features=d_model, out_features=d_model, device=device, dtype=dtype)
        self.W_o = Linear(in_features=d_model, out_features=d_model, device=device, dtype=dtype)
        self.rope = None
        if rope_theta is not None and rope_max_seq_len is not None:
            self.rope = RotaryPositionalEmbedding(
                theta=rope_theta,
                d_k=self.d_model // self.num_heads,
                max_seq_len=rope_max_seq_len,
                device=device,
            )

    def _split_heads(self, x: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
        return x.view(*x.shape[:-1], num_heads, head_dim).transpose(-3, -2)

    def _merge_heads(slef, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(-3, -2).flatten(-2, -1)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        seq_len = x.shape[-2]
        Q_heads = self._split_heads(self.W_q(x), self.num_heads, self.d_k)
        K_heads = self._split_heads(self.W_k(x), self.num_heads, self.d_k)
        V_heads = self._split_heads(self.W_v(x), self.num_heads, self.d_v)
        if self.rope:
            Q_embd = self.rope(Q_heads, token_positions=token_positions)
            K_embd = self.rope(K_heads, token_positions=token_positions)
        else:
            Q_embd = Q_heads
            K_embd = K_heads
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
        att = scaled_dot_product_attention(
            Q=Q_embd,
            K=K_embd,
            V=V_heads,
            mask=mask,
        )
        return self.W_o(self._merge_heads(att))


class Transformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float | None = None,
        rope_max_seq_len: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.ln1 = RMSNorm(d_model=d_model, eps=1e-5, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model=d_model, eps=1e-5, device=device, dtype=dtype)
        self.sa = MultiheadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            rope_theta=rope_theta,
            rope_max_seq_len=rope_max_seq_len,
            device=device,
            dtype=dtype,
        )
        self.swiglu = SwiGLU(
            d_model=d_model,
            d_ff=d_ff,
            device=device,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None) -> torch.Tensor:
        x = x + self.sa(self.ln1(x), token_positions)
        x = x + self.swiglu(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        num_heads: int,
        d_model: int,
        d_ff: int,
        rope_theta: float = 10000,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.embedding = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            device=device,
            dtype=dtype,
        )
        self.layers = nn.ModuleList(
            [
                Transformer(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    rope_theta=rope_theta,
                    rope_max_seq_len=context_length,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.ff = Linear(in_features=d_model, out_features=vocab_size)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, token_positions=token_positions)
        x = self.ln(x)
        x = self.ff(x)
        return x
