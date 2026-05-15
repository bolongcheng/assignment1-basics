import argparse
from pathlib import Path
from typing import Any
import yaml

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import wandb

from cs336_basics.train_utils import load_batch, load_checkpoint, save_checkpoint, load_dataset
from cs336_basics.transformer import cross_entropy, TransformerLM, AdamW, gradient_clipping

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
    losses = []
    for _ in range(num_batches):
        x, y = load_batch(
            dataset=dataset,
            batch_size=batch_size,
            context_length=context_length,
            device=device,
        )

        loss = cross_entropy(model(x), y)
        losses.append(loss.item())

    model.train()
    return float(np.mean(losses))


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

    if config["resume_from"] is not None:
        print(f"Loading checkpoint from: {config['resume_from']}")
        start_iter = load_checkpoint(src=config["resume_from"], model=model, optimizer=optimizer)
        print(f"Resumed from iteration {start_iter}")

    wandb.init(project=config["wandb_project"], name=config["wandb_run_name"], config=config)

    for iter in range(start_iter, config["max_iters"]):
        batch = load_batch(
            dataset=train_data,
            batch_size=config["batch_size"],
            context_length=config["context_length"],
            device=device,
        )
        loss = train_step(
            model=model,
            optimizer=optimizer,
            batch=batch,
            max_l2_norm=config["max_l2_norm"],
        )

        if iter % config["log_interval"] == 0:
            train_perplexity = np.exp(loss)
            wandb.log(
                {
                    "train_loss": loss,
                    "train_perplexity": train_perplexity,
                },
                step=iter,
            )

        if iter > 0 and iter % config["eval_interval"] == 0:
            val_loss = evaluate(
                model=model,
                dataset=valid_data,
                batch_size=config["batch_size"],
                context_length=config["context_length"],
                device=device,
                num_batches=config["eval_num_batches"],
            )
            val_perplexity = np.exp(val_loss)
            wandb.log(
                {
                    "val_loss": val_loss,
                    "val_perplexity": val_perplexity,
                },
                step=iter,
            )

        if iter > 0 and iter % config["checkpoint_interval"] == 0:
            ckpt_dir = Path(config["checkpoint_dir"])
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = ckpt_dir / f"checkpoint_{iter}.pt"

            save_checkpoint(model=model, optimizer=optimizer, iteration=iter, out=ckpt_path)

    ckpt_dir = Path(config["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"checkpoint_{config['max_iters']}.pt"
    save_checkpoint(model=model, optimizer=optimizer, iteration=config["max_iters"], out=ckpt_path)


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
