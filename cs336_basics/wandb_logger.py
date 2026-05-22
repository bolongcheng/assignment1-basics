import math
import time

import torch
import torch.nn as nn
import wandb

from cs336_basics.model import Linear, RMSNorm


class WandbLogger:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        grad_clip_threshold: float,
        log_interval: int = 10,  # log scalars every N steps
        activation_log_interval: int = 100,  # log activations every N steps (expensive)
        starting_step: int = 0,
        num_tokens_processed: int = 0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.log_interval = log_interval
        self.activation_log_interval = activation_log_interval
        self.grad_clip_threshold = grad_clip_threshold

        self.step = starting_step
        self.num_tokens_processed = num_tokens_processed
        self.grad_clip_count = 0
        self.step_start_time = time.time()
        self._tokens_since_last_log = 0

        # stores activation norms, populated by forward hooks
        self._activation_norms: dict[str, float] = {}
        self._hooks = []
        self._register_activation_hooks()

    # ── Forward hooks to capture activation norms ──────────────────────────

    def _register_activation_hooks(self):
        """Registers hooks on every named Linear/LayerNorm layer."""
        for name, module in self.model.named_modules():
            if isinstance(module, (Linear, RMSNorm)):
                hook = module.register_forward_hook(self._make_hook(name))
                self._hooks.append(hook)

    def _make_hook(self, name: str):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                self._activation_norms[name] = output.detach().float().norm()

        return hook

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()

    def _compute_grad_norms(self) -> tuple[dict, float]:
        logs = {}
        total_norm_sq = 0.0

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                norm = param.grad.detach().float().norm().item()
                safe_name = name.replace(".", "/")
                logs[f"grad_norm/{safe_name}"] = norm
                total_norm_sq += norm**2

                # Gradient-to-weight ratio
                if param.data.norm().item() > 1e-8:
                    ratio = norm / param.data.detach().float().norm().item()
                    logs[f"grad_to_weight_ratio/{safe_name}"] = ratio

        global_norm = math.sqrt(total_norm_sq)
        return logs, global_norm

    def _compute_weight_metrics(self) -> dict:
        logs = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                safe_name = name.replace(".", "/")
                weight_norm = param.data.detach().float().norm().item()
                logs[f"weight_norm/{safe_name}"] = weight_norm

        return logs

    # ── Adam m/v estimates ──────────────────────────────────────────────────

    def _compute_adamw_metrics(self) -> dict:
        """Logs mean of Adam's m (momentum) and v (variance) estimates per layer."""
        logs = {}
        state = self.optimizer.state
        for name, param in self.model.named_parameters():
            if param in state and len(state[param]) > 0:
                safe_name = name.replace(".", "/")
                p_state = state[param]
                if "m" in p_state:  # m estimate
                    logs[f"adam_m/{safe_name}"] = p_state["m"].float().abs().mean().item()
                if "v" in p_state:  # v estimate
                    logs[f"adam_v/{safe_name}"] = p_state["v"].float().mean().item()
        return logs

    def log_step(
        self,
        train_loss: float,
        val_loss: float | None,
        batch_tokens: int,  # tokens in this batch
        skip_activation_log: bool = False,
    ):
        self.step += 1
        self.num_tokens_processed += batch_tokens
        self._tokens_since_last_log += batch_tokens
        step_time = time.time() - self.step_start_time
        self.step_start_time = time.time()

        if self.step % self.log_interval != 0:
            return

        log_dict = {}

        # ── Loss & perplexity ──────────────────────────────────────────────
        log_dict["train/loss"] = train_loss
        log_dict["train/perplexity"] = math.exp(min(train_loss, 20))
        if val_loss is not None:
            log_dict["val/loss"] = val_loss
            log_dict["val/perplexity"] = math.exp(min(val_loss, 20))

        # ── Learning rate ──────────────────────────────────────────────────
        log_dict["optim/lr"] = self.optimizer.param_groups[0]["lr"]

        # ── Gradient norms ─────────────────────────────────────────────────
        grad_logs, global_grad_norm = self._compute_grad_norms()
        log_dict.update(grad_logs)
        log_dict["optim/grad_norm_global"] = global_grad_norm

        # Track gradient clipping frequency
        if global_grad_norm > self.grad_clip_threshold:
            self.grad_clip_count += 1
        log_dict["optim/grad_clip_pct"] = self.grad_clip_count / self.step

        # ── Weight norms & update-to-weight ratios ─────────────────────────
        weight_logs = self._compute_weight_metrics()
        log_dict.update(weight_logs)

        # ── Adam optimizer internals (m, v estimates) ──────────────────────
        adam_logs = self._compute_adamw_metrics()
        log_dict.update(adam_logs)

        # ── Activation norms (captured by forward hooks) ───────────────────
        if not skip_activation_log and self.step % self.activation_log_interval == 0:
            for layer_name, norm in self._activation_norms.items():
                safe_name = layer_name.replace(".", "/")
                log_dict[f"activations/{safe_name}"] = norm.item() if isinstance(norm, torch.Tensor) else norm

        tokens_per_sec = self._tokens_since_last_log / max(step_time, 1e-6)
        self._tokens_since_last_log = 0
        log_dict["perf/tokens_per_sec"] = tokens_per_sec
        log_dict["perf/step_time_ms"] = step_time * 1000
        log_dict["perf/tokens_seen"] = self.num_tokens_processed

        wandb.log(log_dict, step=self.step)
