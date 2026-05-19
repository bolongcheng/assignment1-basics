from collections.abc import Iterable

import torch


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    max_val = x.max(dim=dim, keepdim=True).values
    exp_xs = (x - max_val).exp()
    return exp_xs / exp_xs.sum(dim=dim, keepdim=True)


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    batch_size = logits.shape[0]
    log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    return -log_probs[torch.arange(batch_size), targets].mean()


def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float,
    eps: float = 1e-6,
) -> None:
    grads = [param.grad for param in parameters if param.grad is not None]

    grads_tensor = torch.cat([g.view(-1) for g in grads])
    total_norm = torch.linalg.norm(grads_tensor)

    clip_factor = max_l2_norm / (total_norm + eps)

    if clip_factor < 1:
        for g in grads:
            g *= clip_factor
