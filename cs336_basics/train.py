import argparse
from pathlib import Path
from typing import Any

import numpy.typing as npt
import torch
import torch.nn as nn
import wandb
import yaml

from cs336_basics.model import TransformerLM
from cs336_basics.optimizer import AdamW, lr_cosine_schedule
from cs336_basics.train_utils import load_batch, load_checkpoint, load_dataset, save_checkpoint
from cs336_basics.utils import cross_entropy, gradient_clipping
from cs336_basics.wandb_logger import WandbLogger


wandb.login()


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
    total_loss = 0.0
    for _ in range(num_batches):
        x, y = load_batch(
            dataset=dataset,
            batch_size=batch_size,
            context_length=context_length,
            device=device,
        )

        total_loss += cross_entropy(model(x), y).item()

    model.train()
    return total_loss / num_batches


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: tuple[torch.Tensor, torch.Tensor],
    max_l2_norm: float,
) -> float:
    model.train()

    x, y = batch
    loss = cross_entropy(model(x), y)
    optimizer.zero_grad(set_to_none=True)
    gradient_clipping(
        model.parameters(),
        max_l2_norm=max_l2_norm,
    )
    optimizer.step()
    return loss.item()


def _make_checkpoint_path(checkpoint_dir: str, iter: int) -> Path:
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return ckpt_dir / f"checkpoint_{iter}.pt"


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
    )

    optimizer = AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        betas=(config["beta1"], config["beta2"]),
        weight_decay=config["weight_decay"],
    )

    start_iter = 0
    num_tokens_processed = 0

    if config["resume_from"] is not None:
        print(f"Loading checkpoint from: {config['resume_from']}")
        start_iter = load_checkpoint(src=config["resume_from"], model=model, optimizer=optimizer)
        print(f"Resumed from iteration {start_iter}")
        num_tokens_processed = start_iter * config["batch_size"] * config["context_length"]

    wandb.init(project=config["wandb_project"], name=config["wandb_run_name"], config=config)
    logger = WandbLogger(
        model=model,
        optimizer=optimizer,
        log_interval=config["log_interval"],
        activation_log_interval=config["activation_log_interval"],
        grad_clip_threshold=config["max_l2_norm"],
        starting_step=start_iter,
        num_tokens_processed=num_tokens_processed,
    )

    for iter in range(start_iter, config["max_iters"]):
        train_batch = load_batch(
            dataset=train_data,
            batch_size=config["batch_size"],
            context_length=config["context_length"],
            device=device,
        )

        lr = lr_cosine_schedule(
            lr_min=config["lr_min"],
            lr_max=config["lr_max"],
            iter=iter,
            T_w=config["T_w"],
            T_c=config["T_c"],
        )
        optimizer.param_groups[0]["lr"] = lr

        train_loss = train_step(
            model=model,
            optimizer=optimizer,
            batch=train_batch,
            max_l2_norm=config["max_l2_norm"],
        )

        val_loss = None
        if iter > 0 and iter % config["eval_interval"] == 0:
            val_loss = evaluate(
                model=model,
                dataset=valid_data,
                batch_size=config["batch_size"],
                context_length=config["context_length"],
                device=device,
                num_batches=config["eval_num_batches"],
            )

        logger.log_step(
            train_loss=train_loss,
            val_loss=val_loss,
            batch_tokens=train_batch[0].numel(),
        )

        if (iter > 0 and iter % config["checkpoint_interval"] == 0) or iter == config["max_iters"]:
            ckpt_path = _make_checkpoint_path(config["checkpoint_dir"], iter)
            save_checkpoint(model=model, optimizer=optimizer, iteration=iter, out=ckpt_path)

    logger.remove_hooks()
    wandb.finish()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, required=True)

    # Model parameters
    parser.add_argument("--vocab-size", type=int, default=32_000)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--d-ff", type=int, default=3072)
    parser.add_argument("--rope-theta", type=float, default=10000.0)

    # Optimizer parameters
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-l2-norm", type=float, default=1.0)
    parser.add_argument("--lr-min", type=float, default=3e-5)
    parser.add_argument("--lr-max", type=float, default=3e-4)
    parser.add_argument("--T-w", type=int, default=800)
    parser.add_argument("--T-c", type=int, default=38000)

    # Training parameters
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-iters", type=int, default=10000)

    # Logging and evaluation
    parser.add_argument("--log-interval", type=int)
    parser.add_argument("--eval-interval", type=int)
    parser.add_argument("--eval-num-batches", type=int)

    # Checkpointing
    parser.add_argument("--checkpoint-interval", type=int)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")

    # Miscellanous
    parser.add_argument("--device", type=str, default="cpu")

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
    parser = build_parser()
    args = parser.parse_args()
    config = load_yaml_config(args.config)
    config = merge_cli_overrides(config, args)

    train(config)


if __name__ == "__main__":
    main()
