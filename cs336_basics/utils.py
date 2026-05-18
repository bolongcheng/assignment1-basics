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
    params_with_grad = [param.grad.view(-1) for param in parameters if param.grad is not None]

    params_with_grad_tensor = torch.cat(params_with_grad)
    total_norm = torch.linalg.norm(params_with_grad_tensor)

    clip_factor = max_l2_norm / (total_norm + eps)

    if clip_factor < 1:
        for p in parameters:
            if p.grad is not None:
                p.grad *= clip_factor
