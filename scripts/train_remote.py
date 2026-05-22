from typing import Any

import modal
import yaml


app = modal.App("cs336-train")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("uv")
    .pip_install_from_pyproject("pyproject.toml")
    .add_local_python_source("cs336_basics")
    .add_local_dir("scripts", remote_path="/root/scripts", ignore=["bpe_*.py"])
)

data_vol = modal.Volume.from_name("cs336-data", create_if_missing=True)
ckpt_vol = modal.Volume.from_name("cs336-checkpoints", create_if_missing=True)


@app.function(
    gpu="H100",
    image=image,
    volumes={"/root/data": data_vol, "/root/checkpoints": ckpt_vol},
    secrets=[modal.Secret.from_name("wandb-secret")],
    timeout=1800,
)
def train_remote(config: dict[str, Any]) -> None:
    import sys

    sys.path.append("/root")
    from scripts.train_model import train

    train(config)


@app.local_entrypoint()
def main(config: str, resume_from: str | None = None) -> None:

    with open(config) as f:
        run_config = yaml.safe_load(f)

    if resume_from is not None:
        run_config["resume_from"] = resume_from

    train_remote.remote(run_config)
