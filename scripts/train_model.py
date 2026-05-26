import argparse
import logging
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import wandb
import yaml
from tqdm import tqdm

from cs336_basics.model import TransformerLM
from cs336_basics.optimizer import AdamW, lr_cosine_schedule, lr_trapezoidal_schedule
from cs336_basics.train_utils import load_batch, load_checkpoint, load_dataset, save_checkpoint
from cs336_basics.utils import cross_entropy, gradient_clipping
from cs336_basics.wandb_logger import WandbLogger


DEVICE_TO_DTYPE = {
    "cuda": torch.bfloat16,
    "cpu": torch.float32,
}


EVAL_SEED = 23


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
    num_batches: int,
) -> float:

    model.eval()
    rng = np.random.default_rng(EVAL_SEED)
    total_loss = torch.zeros(1, device=device)
    for _ in range(num_batches):
        x, y = load_batch(
            dataset=dataset,
            batch_size=batch_size * 4,
            context_length=context_length,
            device=device,
            rng=rng,
        )
        pred = model(x)
        total_loss += cross_entropy(pred.view(-1, pred.shape[-1]), y.view(-1))

    model.train()
    return (total_loss / num_batches).item()


@torch.no_grad()
def evaluate_full(
    model: nn.Module,
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
) -> float:
    model.eval()
    if device == "cuda":
        torch.cuda.empty_cache()
    n_windows = (len(dataset) - 1) // context_length
    starts = np.arange(n_windows, dtype=np.int64) * context_length

    total_loss = torch.zeros(1, device=device)
    total_tokens = 0
    for i in range(0, n_windows, batch_size):
        batch_starts = starts[i : i + batch_size]
        x_np = np.stack([dataset[s : s + context_length] for s in batch_starts]).astype(np.int64, copy=False)
        y_np = np.stack([dataset[s + 1 : s + 1 + context_length] for s in batch_starts]).astype(np.int64, copy=False)
        x = torch.from_numpy(x_np).to(device, non_blocking=True)
        y = torch.from_numpy(y_np).to(device, non_blocking=True)
        pred = model(x)
        loss = cross_entropy(pred.view(-1, pred.shape[-1]), y.view(-1))
        ntok = y.numel()
        total_loss += loss * ntok
        total_tokens += ntok

    model.train()
    return (total_loss / total_tokens).item()


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: tuple[torch.Tensor, torch.Tensor],
    max_l2_norm: float,
) -> float:
    model.train()

    x, y = batch
    pred = model(x)
    loss = cross_entropy(pred.view(-1, pred.shape[-1]), y.view(-1))
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradient_clipping(
        model.parameters(),
        max_l2_norm=max_l2_norm,
    )
    optimizer.step()
    return loss.item()


def _make_checkpoint_path(checkpoint_dir: str, iter: int) -> Path:
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return ckpt_dir / f"owt_checkpoint_{iter}.pt"


# Hyperparameters that may be overridden by a W&B sweep agent via wandb.config.
# When running under `wandb agent`, wandb.init() will populate wandb.config from the
# sweep; we then mirror those values back into our local config dict so the rest of
# the training loop does not need to know about sweeps.
_SWEEPABLE_KEYS = ("lr_max", "warmup_pct", "lr_min_ratio", "cooldown_pct", "max_iters")


def _apply_wandb_sweep_overrides(config: dict[str, Any]) -> dict[str, Any]:
    sweep_cfg = dict(wandb.config)
    for key in _SWEEPABLE_KEYS:
        if key in sweep_cfg and sweep_cfg[key] is not None:
            config[key] = sweep_cfg[key]
    return config


def train(config: dict[str, Any]) -> None:
    device = config["device"]

    train_data = load_dataset(config["train_path"])
    valid_data = load_dataset(config["valid_path"])

    model = TransformerLM(
        vocab_size=config["vocab_size"],
        context_length=config["context_length"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_model=config["d_model"],
        d_ff=config["d_ff"],
        rope_theta=config["rope_theta"],
        device=device,
        dtype=DEVICE_TO_DTYPE[device],
        tie_word_embeddings=config.get("tie_word_embeddings", False),
        use_qk_norm=config.get("use_qk_norm", False),
    )
    model = torch.compile(model)

    optimizer = AdamW(
        model.parameters(),
        lr=config["lr_max"],
        betas=(config["beta1"], config["beta2"]),
        weight_decay=config["weight_decay"],
    )

    start_iter = 0
    num_tokens_processed = 0

    if config["resume_from"] is not None:
        logging.info(f"Loading checkpoint from: {config['resume_from']}")
        start_iter = load_checkpoint(src=config["resume_from"], model=model, optimizer=optimizer)
        logging.info(f"Resumed from iteration {start_iter}")
        num_tokens_processed = start_iter * config["batch_size"] * config["context_length"]

    wandb.init(project=config["wandb_project"], name=config.get("wandb_run_name"), config=config)
    config = _apply_wandb_sweep_overrides(config)
    wandb.config.update(config, allow_val_change=True)
    wandb_logger = WandbLogger(
        model=model,
        optimizer=optimizer,
        log_interval=config["log_interval"],
        activation_log_interval=config["activation_log_interval"],
        grad_clip_threshold=config["max_l2_norm"],
        starting_step=start_iter,
        num_tokens_processed=num_tokens_processed,
    )

    with ThreadPoolExecutor(max_workers=1) as data_pool, ThreadPoolExecutor(max_workers=1) as ckpt_pool:
        pending_ckpt: Future | None = None
        next_batch_future: Future = data_pool.submit(
            load_batch,
            dataset=train_data,
            batch_size=config["batch_size"],
            context_length=config["context_length"],
            device=device,
        )

        for iter in tqdm(range(start_iter, config["max_iters"])):
            train_batch = next_batch_future.result()
            next_batch_future = data_pool.submit(
                load_batch,
                dataset=train_data,
                batch_size=config["batch_size"],
                context_length=config["context_length"],
                device=device,
            )

            if config.get("scheduler_type") == "trapezoidal":
                lr = lr_trapezoidal_schedule(
                    lr_max=config["lr_max"],
                    iter=iter,
                    T_w=int(config["warmup_pct"] * config["max_iters"]),
                    T_c=int(config["cooldown_pct"] * config["max_iters"]),
                    max_iters=config["max_iters"],
                )
            else:
                lr = lr_cosine_schedule(
                    lr_min=config["lr_min_ratio"] * config["lr_max"],
                    lr_max=config["lr_max"],
                    iter=iter,
                    T_w=int(config["warmup_pct"] * config["max_iters"]),
                    T_c=int(config["cooldown_pct"] * config["max_iters"]),
                )

            optimizer.param_groups[0]["lr"] = lr

            train_loss = train_step(
                model=model,
                optimizer=optimizer,
                batch=train_batch,
                max_l2_norm=config["max_l2_norm"],
            )

            is_final_iter = iter == config["max_iters"] - 1
            val_loss = None
            if is_final_iter:
                val_loss = evaluate_full(
                    model=model,
                    dataset=valid_data,
                    batch_size=config["batch_size"],
                    context_length=config["context_length"],
                    device=device,
                )
            elif iter > 0 and iter % config["eval_interval"] == 0:
                val_loss = evaluate(
                    model=model,
                    dataset=valid_data,
                    batch_size=config["batch_size"],
                    context_length=config["context_length"],
                    device=device,
                    num_batches=config["eval_num_batches"],
                )
            wandb_logger.log_step(
                train_loss=train_loss,
                val_loss=val_loss,
                batch_tokens=train_batch[0].numel(),
            )

            ckpt_interval = config.get("checkpoint_interval") or 0
            if ckpt_interval > 0 and ((iter > 0 and iter % ckpt_interval == 0) or is_final_iter):
                if pending_ckpt is not None:
                    pending_ckpt.result()
                ckpt_path = _make_checkpoint_path(config["checkpoint_dir"], iter)
                pending_ckpt = ckpt_pool.submit(
                    save_checkpoint,
                    model=model,
                    optimizer=optimizer,
                    iteration=iter,
                    out=ckpt_path,
                )

        if pending_ckpt is not None:
            pending_ckpt.result()

    wandb_logger.remove_hooks()
    wandb.finish()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, required=True)

    # Model parameters
    parser.add_argument("--vocab_size", type=int)
    parser.add_argument("--context_length", type=int)
    parser.add_argument("--num_layers", type=int)
    parser.add_argument("--num_heads", type=int)
    parser.add_argument("--d_model", type=int)
    parser.add_argument("--d_ff", type=int)
    parser.add_argument("--rope_theta", type=float)

    # Optimizer parameters
    parser.add_argument("--beta1", type=float)
    parser.add_argument("--beta2", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--max_l2_norm", type=float)
    parser.add_argument("--lr_max", type=float)
    parser.add_argument("--lr_min_ratio", type=float)
    parser.add_argument("--warmup_pct", type=int)
    parser.add_argument("--cooldown_pct", type=int)

    # Training parameters
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--max_iters", type=int)

    # Logging and evaluation
    parser.add_argument("--log_interval", type=int)
    parser.add_argument("--eval_interval", type=int)
    parser.add_argument("--eval_num_batches", type=int)

    # Checkpointing
    parser.add_argument("--checkpoint_interval", type=int)
    parser.add_argument("--resume_from", type=str)

    # Miscellanous
    parser.add_argument("--device", type=str)

    return parser


def load_yaml_config(path: str) -> dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def merge_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    cli_dict = vars(args)

    for k, v in cli_dict.items():
        if v is not None and k != "config":
            config[k] = v

    return config


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    wandb.login()

    parser = build_parser()
    args = parser.parse_args()
    config = load_yaml_config(args.config)
    config = merge_cli_overrides(config, args)

    train(config)


if __name__ == "__main__":
    main()
