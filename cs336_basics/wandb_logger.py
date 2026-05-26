import math
import time

import torch
import torch.nn as nn
import wandb

from cs336_basics.model import Linear, MultiheadSelfAttention, RMSNorm


type LogDict = dict[str, float | int]


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
        # collected references to attention modules for head-redundancy metrics
        self._attn_modules: list[tuple[str, MultiheadSelfAttention]] = [
            (name, module) for name, module in self.model.named_modules() if isinstance(module, MultiheadSelfAttention)
        ]
        self._hooks = []
        self._register_activation_hooks()
        self._schedule_attn_capture()

    @staticmethod
    def _safe_name(name: str) -> str:
        return name.replace(".", "/")

    @staticmethod
    def _get_norm_item(param: torch.Tensor) -> float:
        return param.detach().norm().float().item()

    def _should_log(self, val_loss: float | None) -> bool:
        return self.step % self.log_interval == 0 or val_loss is not None

    def _should_log_activations(self, skip_activation_log: bool) -> bool:
        return not skip_activation_log and self.step % self.activation_log_interval == 0

    def _increment_step(self, batch_tokens: int) -> float:
        self.step += 1
        self.num_tokens_processed += batch_tokens
        self._tokens_since_last_log += batch_tokens
        step_time = time.time() - self.step_start_time
        self.step_start_time = time.time()
        return step_time

    def _performance_logs(self, step_time: float) -> LogDict:
        tokens_per_sec = self._tokens_since_last_log / max(step_time, 1e-6)
        self._tokens_since_last_log = 0
        return {
            "perf/tokens_per_sec": tokens_per_sec,
            "perf/step_time_ms": step_time * 1000,
            "perf/tokens_seen": self.num_tokens_processed,
        }

    def _loss_logs(self, train_loss: float, val_loss: float | None) -> LogDict:
        logs: LogDict = {
            "train/loss": train_loss,
            "train/perplexity": math.exp(min(train_loss, 20)) if math.isfinite(train_loss) else float("inf"),
        }
        if val_loss is not None:
            logs["val/loss"] = val_loss
            logs["val/perplexity"] = math.exp(min(val_loss, 20)) if math.isfinite(val_loss) else float("inf")
        return logs

    def _compute_weight_norms(self) -> LogDict:
        logs: LogDict = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                safe_name = self._safe_name(name)
                logs[f"weight_norm/{safe_name}"] = self._get_norm_item(param.data)

        return logs

    def _compute_grad_norms(self) -> tuple[LogDict, float]:
        logs: LogDict = {}
        total_norm_sq = 0.0

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                norm = self._get_norm_item(param.grad)
                safe_name = self._safe_name(name)
                logs[f"grad_norm/{safe_name}"] = norm
                total_norm_sq += norm**2

                # Gradient-to-weight ratio
                if param.data.norm().item() > 1e-8:
                    ratio = norm / self._get_norm_item(param.data)
                    logs[f"grad_to_weight_ratio/{safe_name}"] = ratio

        global_norm = math.sqrt(total_norm_sq)
        return logs, global_norm

    def _optimizer_logs(self, global_grad_norm: float) -> LogDict:
        if global_grad_norm > self.grad_clip_threshold:
            self.grad_clip_count += 1
        return {
            "optim/lr": self.optimizer.param_groups[0]["lr"],
            "optim/grad_norm_global": global_grad_norm,
            "optim/grad_clip_pct": self.grad_clip_count / self.step,
        }

    def _compute_adamw_metrics(self) -> LogDict:
        """Logs mean of AdamW's m (momentum) and v (variance) estimates per layer,
        and the estimated update-to-weight ratio."""
        logs: LogDict = {}
        state = self.optimizer.state
        total_update_sq = 0.0
        total_weight_sq = 0.0
        for name, param in self.model.named_parameters():
            if param not in state or len(state[param]) == 0:
                continue
            safe_name = self._safe_name(name)
            p_state = state[param]
            if "m" in p_state:
                logs[f"adam_m/{safe_name}"] = p_state["m"].abs().mean().item()
            if "v" in p_state:
                logs[f"adam_v/{safe_name}"] = p_state["v"].mean().item()

            if "last_update_norm" not in p_state:
                continue
            update_norm = p_state["last_update_norm"].item()
            weight_norm = self._get_norm_item(param.data)
            if weight_norm > 1e-8:
                logs[f"update_to_weight_ratio/{safe_name}"] = update_norm / weight_norm
            total_update_sq += update_norm**2
            total_weight_sq += weight_norm**2

        if total_weight_sq > 1e-16:
            logs["optim/update_to_weight_ratio_global"] = math.sqrt(total_update_sq) / math.sqrt(total_weight_sq)
        return logs

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
                # Normalize by number of elements to get per-element norm
                self._activation_norms[name] = output.detach().float().norm() / output.numel()

        return hook

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()

    def _schedule_attn_capture(self):
        capture = (self.step + 1) % self.activation_log_interval == 0
        for _, module in self._attn_modules:
            module.capture_internals = capture

    def pre_step(self):
        """Compatibility shim for older training loops."""
        self._schedule_attn_capture()

    def _disable_attn_capture(self):
        for _, module in self._attn_modules:
            module.capture_internals = False

    def _activation_logs(self) -> LogDict:
        logs: LogDict = {}
        for layer_name, norm in self._activation_norms.items():
            safe_name = self._safe_name(layer_name)
            logs[f"activations/{safe_name}"] = norm.item() if isinstance(norm, torch.Tensor) else norm
        return logs

    def _compute_head_redundancy(self) -> LogDict:
        """Reads cached attention probs and per-head SDPA outputs from each
        `MultiheadSelfAttention` module and computes redundancy/entropy stats."""
        logs: LogDict = {}
        for name, module in self._attn_modules:
            probs = module._last_attn_probs
            head_out = module._last_head_output
            if probs is None or head_out is None:
                continue
            safe_name = self._safe_name(name)
            stats = self._head_redundancy_stats(head_out.float(), probs.float())
            for metric_name, value in stats.items():
                logs[f"head_redundancy/{safe_name}/{metric_name}"] = value
            module._last_attn_probs = None
            module._last_head_output = None
        return logs

    @staticmethod
    def _head_redundancy_stats(head_out: torch.Tensor, probs: torch.Tensor) -> dict[str, float]:
        # head_out: (B, H, T, d_v); probs: (B, H, T, T)
        _, H, T, _ = head_out.shape
        # Pairwise cosine similarity between heads using flattened per-head outputs.
        flat = head_out.transpose(0, 1).reshape(H, -1)
        flat = flat / flat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        cos = flat @ flat.T
        off_mask = ~torch.eye(H, dtype=torch.bool, device=cos.device)
        off = cos[off_mask]

        # Per-head attention entropy, normalized by log of effective key count.
        log_probs = torch.log(probs.clamp_min(1e-12))
        entropy = -(probs * log_probs).sum(dim=-1)  # (B, H, T)
        eff_keys = torch.arange(1, T + 1, device=entropy.device, dtype=entropy.dtype)
        norm_entropy = entropy[:, :, 1:] / torch.log(eff_keys[1:])  # skip t=0 (entropy is 0)
        entropy_per_head = entropy.mean(dim=(0, 2))
        norm_entropy_per_head = norm_entropy.mean(dim=(0, 2))

        return {
            "cos_off_mean": off.mean().item(),
            "cos_off_max": off.max().item(),
            "cos_off_abs_mean": off.abs().mean().item(),
            "cos_off_std": off.std().item(),
            "entropy_mean": entropy_per_head.mean().item(),
            "entropy_min": entropy_per_head.min().item(),
            "entropy_max": entropy_per_head.max().item(),
            "norm_entropy_mean": norm_entropy_per_head.mean().item(),
            "norm_entropy_min": norm_entropy_per_head.min().item(),
            "norm_entropy_max": norm_entropy_per_head.max().item(),
            "norm_entropy_std": norm_entropy_per_head.std().item(),
        }

    def log_step(
        self,
        train_loss: float,
        val_loss: float | None,
        batch_tokens: int,  # tokens in this batch
        skip_activation_log: bool = False,
    ):
        step_time = self._increment_step(batch_tokens)
        if not self._should_log(val_loss):
            self._schedule_attn_capture()
            return

        grad_logs, global_grad_norm = self._compute_grad_norms()

        log_dict: LogDict = {}
        log_dict.update(self._loss_logs(train_loss, val_loss))
        log_dict.update(self._optimizer_logs(global_grad_norm))
        log_dict.update(grad_logs)
        log_dict.update(self._compute_weight_norms())
        log_dict.update(self._compute_adamw_metrics())

        if self._should_log_activations(skip_activation_log):
            log_dict.update(self._activation_logs())
            log_dict.update(self._compute_head_redundancy())
            self._disable_attn_capture()

        log_dict.update(self._performance_logs(step_time))

        wandb.log(log_dict, step=self.step)
        self._schedule_attn_capture()
