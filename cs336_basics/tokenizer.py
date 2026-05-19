import json
from collections.abc import Iterable, Iterator

import regex as re

from cs336_basics.bpe import ENCODE_FMT, GPT2_PAT
from tests.common import gpt2_bytes_to_unicode


def recover_bytes(token: str, unicode_to_byte: dict[str, int]) -> bytes:
    return bytes([unicode_to_byte[c] for c in token])


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
        regex_pattern: str = GPT2_PAT,
    ):
        self.vocab: dict[int, bytes] = vocab
        self.vocab_inverse: dict[bytes, int] = {v: k for k, v in vocab.items()}
        self.merge_lookup: dict[tuple[int, int], int] = {
            (self.vocab_inverse[pair[0]], self.vocab_inverse[pair[1]]): self.vocab_inverse[pair[0] + pair[1]]
            for pair in merges
        }
        self.special_tokens = special_tokens
        self.compiled_pattern = re.compile(regex_pattern)
        if special_tokens:
            sorted_special = sorted(special_tokens, key=len, reverse=True)
            escaped_special = [re.escape(tok) for tok in sorted_special]
            self.special_pattern = re.compile("(" + "|".join(escaped_special) + ")")
        else:
            self.special_pattern = None

    @classmethod
    def from_files(
        cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None
    ) -> "Tokenizer":
        unicode_to_byte = {v: k for k, v in gpt2_bytes_to_unicode().items()}
        with open(vocab_filepath, encoding=ENCODE_FMT) as f:
            vocab_inverse = json.load(f)
            vocab = {int(v): recover_bytes(k, unicode_to_byte) for k, v in vocab_inverse.items()}
        with open(merges_filepath, encoding=ENCODE_FMT) as f:
            merges = [
                (recover_bytes(tok1, unicode_to_byte), recover_bytes(tok2, unicode_to_byte))
                for line in f
                for tok1, tok2 in [line.rstrip("\n").split(" ", maxsplit=1)]
            ]
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        tokens = []
        chunks = self.special_pattern.split(text) if self.special_pattern else [text]
        for chunk in chunks:
            if not chunk:
                continue
            if self.special_tokens and chunk in self.special_tokens:
                tokens.append(self.vocab_inverse[chunk.encode(ENCODE_FMT)])
            else:
                for pre_token_match in self.compiled_pattern.finditer(chunk):
                    tokens.extend(self._encode_pre_token_chunk(pre_token_match.group(0)))
        return tokens

    def _encode_pre_token_chunk(self, pre_token_str: str) -> list[int]:
        ids = [self.vocab_inverse[bytes([b])] for b in pre_token_str.encode(ENCODE_FMT)]
        while len(ids) >= 2:
            all_pairs = [(id1, id2) for id1, id2 in zip(ids, ids[1:])]
            best_pair = min(all_pairs, key=lambda pair: self.merge_lookup.get(pair, float("inf")))
            if best_pair not in self.merge_lookup:
                break

            ids = self._merge_ids(ids, best_pair)

        return ids

    def _merge_ids(self, ids: list[int], best_pair: tuple[int, int]) -> list[int]:
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == best_pair[0] and ids[i + 1] == best_pair[1]:
                new_ids.append(self.merge_lookup[best_pair])
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            chunks = self.special_pattern.split(text) if self.special_pattern else [text]
            for chunk in chunks:
                if not chunk:
                    continue
                if self.special_tokens and chunk in self.special_tokens:
                    yield self.vocab_inverse[chunk.encode(ENCODE_FMT)]
                else:
                    for pre_token_match in self.compiled_pattern.finditer(chunk):
                        yield from self._encode_pre_token_chunk(pre_token_match.group(0))

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[tok_id] for tok_id in ids).decode(ENCODE_FMT, errors="replace")
