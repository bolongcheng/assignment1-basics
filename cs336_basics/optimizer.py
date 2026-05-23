import math
from collections.abc import Callable

import torch


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.

        for p in group["params"]:
            if p.grad is None:
                continue
            state = self.state[p]  # Get state associated with p.
            t = state.get("t", 0)  # Get iteration number from the state, or 0.
            grad = p.grad.data  # Get the gradient of loss with respect to p.
            p.data -= lr / math.sqrt(t + 1) * grad  # Update weight tensor in-place.
            state["t"] = t + 1  # Increment iteration number.
        return loss


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.99),
        eps: float = 1e-8,
        weight_decay: float = 0,
    ) -> None:
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0 <= betas[0] < 1:
            raise ValueError(f"Invalid beta_1 (betas[0]): {betas[0]}")
        if not 0 <= betas[1] < 1:
            raise ValueError(f"Invalid beta_2 (betas[1]): {betas[1]}")
        if eps < 0:
            raise ValueError(f"Invalid eps: {eps}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            params_with_grad = group["params"]
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in params_with_grad:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 1)
                m = state.get("m", torch.zeros_like(p.data))
                v = state.get("v", torch.zeros_like(p.data))
                grad = p.grad.data
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad**2
                update = lr * weight_decay * p.data + lr * (1 - beta2**t) ** 0.5 / (1 - beta1**t) * m / (v.sqrt() + eps)
                p.data -= update
                t += 1
                state["m"] = m
                state["v"] = v
                state["t"] = t
                state["last_update_norm"] = (update).detach().norm()

        return loss


def lr_cosine_schedule(
    lr_min: float,
    lr_max: float,
    iter: int,
    T_w: int,
    T_c: int,
) -> float:
    if iter < T_w:
        return iter * lr_max / T_w

    if T_w <= iter <= T_c:
        return lr_min + 0.5 * (1 + math.cos((iter - T_w) / (T_c - T_w) * math.pi)) * (lr_max - lr_min)

    return lr_min
