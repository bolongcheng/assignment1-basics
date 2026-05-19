import argparse
from collections.abc import Iterator

import numpy as np
from tqdm import tqdm

from cs336_basics.bpe import ENCODE_FMT
from cs336_basics.pretokenization_example import find_chunk_boundaries
from cs336_basics.tokenizer import Tokenizer


DATASET_MAPPER = {
    "tinystories": {
        "vocab-path": "./data/tokenizer/tinystories-vocab.json",
        "merges-path": "./data/tokenizer/tinystories-merges.txt",
        "input-train-path": "./data/TinyStoriesV2-GPT4-train.txt",
        "input-valid-path": "./data/TinyStoriesV2-GPT4-valid.txt",
        "output-train-path": "./data/tokens/tinystories-train.bin",
        "output-valid-path": "./data/tokens/tinystories-val.bin",
    },
    "owt": {
        "vocab-path": "./data/tokenizer/owt-vocab.json",
        "merges-path": "./data/tokenizer/owt-merges.txt",
        "input-train-path": "./data/owt_train.txt",
        "input-valid-path": "./data/owt_val.txt",
        "output-train-path": "./data/tokens/owt-train.bin",
        "output-valid-path": "./data/tokens/owt-val.bin",
    },
}


def load_tokenizer(
    vocab_path: str,
    merges_path: str,
    special_tokens: list[str] = ["<|endoftext|>"],
) -> Tokenizer:
    return Tokenizer.from_files(vocab_path, merges_path, special_tokens)


def chunk_file(input_path: str, eot_token: str = "<|endoftext|>") -> Iterator[str]:
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, 10_000, b"<|endoftext|>")
        for start, end in tqdm(zip(boundaries[:-1], boundaries[1:]), desc="Chunking", total=len(boundaries) - 1):
            with open(input_path, "rb") as f:
                f.seek(start)
                chunk = f.read(end - start).decode(ENCODE_FMT, errors="ignore")
                yield chunk + eot_token


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=DATASET_MAPPER.keys(), default="tinystories")
    parser.add_argument("--split", type=str, choices=["train", "valid"], default="train")
    args = parser.parse_args()

    dataset_info = DATASET_MAPPER[args.dataset]
    vocab_path = dataset_info["vocab-path"]
    merges_path = dataset_info["merges-path"]
    input_path = dataset_info[f"input-{args.split}-path"]
    output_path = dataset_info[f"output-{args.split}-path"]

    tokenizer = load_tokenizer(vocab_path, merges_path, special_tokens=["<|endoftext|>"])
    tok_ids = list(tokenizer.encode_iterable(chunk_file(input_path, eot_token=tokenizer.special_tokens[0])))

    mmap = np.memmap(output_path, dtype=np.uint32, mode="w+", shape=len(tok_ids))
    mmap[:] = tok_ids
    print(f"Saved {len(tok_ids)} tokens to {output_path}")
    mmap.flush()


if __name__ == "__main__":
    main()
