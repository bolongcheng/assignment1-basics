import json
import cProfile

from cs336_basics.bpe import train_bpe, ENCODE_FMT


if __name__ == "__main__":
    # cProfile.run('train_bpe("./tests/fixtures/tinystories_sample_5M.txt", 500, ["<|endoftext|>"], verbose=False)')

    vocab, merges = train_bpe(
        "./data/TinyStoriesV2-GPT4-valid.txt",
        1_000,
        ["<|endoftext|>"],
        verbose=False,
        save_checkpoint=True,
    )
    print(vocab)
    serialize_vocab = {v.decode(ENCODE_FMT, errors="replace"): k for k, v in vocab.items()}

    with open("./data/TinyStoriesV2-vocab.json", "w") as f:
        json.dump(serialize_vocab, f, indent=2)
    with open("./data/TinyStoriesV2-merges.txt", "w") as f:
        for merge in merges:
            f.write(
                f"{merge[0].decode(ENCODE_FMT, errors='replace')} {merge[1].decode(ENCODE_FMT, errors='replace')}\n"
            )
