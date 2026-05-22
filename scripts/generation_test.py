from typing import Any

import torch
import yaml

from cs336_basics.model import TransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.train_utils import load_checkpoint


def load_model(config: dict[str, Any]) -> TransformerLM:
    model = TransformerLM(
        vocab_size=config["vocab_size"],
        context_length=config["context_length"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_model=config["d_model"],
        d_ff=config["d_ff"],
        rope_theta=config["rope_theta"],
    )
    model.to(torch.float16)
    optimizer = AdamW(model.parameters())
    load_checkpoint("checkpoints/checkpoint_39999.pt", model, optimizer)
    return model


def main() -> None:
    with open("configs/remote-params.yaml", "r") as f:
        config = yaml.safe_load(f)

    model = load_model(config)
    tokenizer = Tokenizer.from_files(
        vocab_filepath="data/tokenizer/tinystories-vocab.json",
        merges_filepath="data/tokenizer/tinystories-merges.txt",
        special_tokens=["<|endoftext|>"],
    )

    prompt = "Once upon a time, in the land of make-belief"
    prompt_tokens = tokenizer.encode(prompt)

    # Generate text
    generated_tokens = model.generate(
        torch.tensor(prompt_tokens, dtype=torch.long), max_new_tokens=100, temperature=0.7
    )
    print(tokenizer.decode(generated_tokens[0].detach().cpu().tolist()))


if __name__ == "__main__":
    main()
