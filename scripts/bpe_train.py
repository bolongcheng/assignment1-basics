from tests.common import gpt2_bytes_to_unicode
import json
import cProfile

from cs336_basics.bpe import train_bpe, render_bytes, UTF8_VOCAB_SIZE


if __name__ == "__main__":
    # cProfile.run('train_bpe("./tests/fixtures/tinystories_sample_5M.txt", 500, ["<|endoftext|>"], verbose=False)')
    special_tokens = ["<|endoftext|>"]
    vocab, merges = train_bpe(
        "./data/TinyStoriesV2-GPT4-valid.txt",
        10_000,
        special_tokens,
        verbose=False,
        save_checkpoint=True,
    )

    mapping = gpt2_bytes_to_unicode()
    serialize_vocab = {}
    for k, v in vocab.items():
        if UTF8_VOCAB_SIZE <= k < UTF8_VOCAB_SIZE + len(special_tokens):
            serialize_vocab[special_tokens[k - UTF8_VOCAB_SIZE]] = k
        else:
            serialize_vocab[render_bytes(v, mapping)] = k
    with open("./data/tokenizer/tinystories-vocab.json", "w") as f:
        json.dump(serialize_vocab, f, indent=2)
    with open("./data/tokenizer/tinystories-merges.txt", "w") as f:
        for merge in merges:
            f.write(f"{render_bytes(merge[0], mapping)} {render_bytes(merge[1], mapping)}\n")
