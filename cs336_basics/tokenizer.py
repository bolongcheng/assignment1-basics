import json
from collections.abc import Iterable, Iterator

from cs336_basics.bpe import ENCODE_FMT


class Tokenizer:
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

    @classmethod
    def from_files(
        cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None
    ) -> "Tokenizer":
        with open(vocab_filepath, encoding=ENCODE_FMT) as f:
            vocab_inverse = json.load(f)
            vocab = {int(v): k.encode(ENCODE_FMT) for k, v in vocab_inverse.items()}
        with open(merges_filepath, encoding=ENCODE_FMT) as f:
            merges = [
                (tok1.encode(ENCODE_FMT), tok2.encode(ENCODE_FMT))
                for line in f
                for tok1, tok2 in [line.split(" ", maxsplit=1)]
            ]
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        pass

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        pass

    def decode(self, ids: list[int]) -> str:
        return "".join(self.vocab[tok_id].decode(ENCODE_FMT) for tok_id in ids)
