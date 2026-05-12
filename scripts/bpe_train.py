import json
import cProfile

from cs336_basics.bpe import train_bpe


if __name__ == "__main__":
    # cProfile.run('train_bpe("./tests/fixtures/tinystories_sample_5M.txt", 500, ["<|endoftext|>"], verbose=False)')

    vocab, merges = train_bpe(
        "./data/TinyStoriesV2-GPT4-train.txt",
        10_000,
        ["<|endoftext|>"],
        verbose=False,
    )
    serialize_vocab = {v.decode("latin-1"): k for k, v in vocab.items()}

    with open("./data/TinyStoriesV2-vocab.json", "w") as f:
        json.dump(serialize_vocab, f, indent=2)
    with open("./data/TinyStoriesV2-merges.txt", "w") as f:
        for merge in merges:
            f.write(f"{merge[0].decode('latin-1')} {merge[1].decode('latin-1')}\n")
