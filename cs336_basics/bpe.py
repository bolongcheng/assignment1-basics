from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import os

import regex as re
from tqdm import tqdm

from cs336_basics.pretokenization_example import find_chunk_boundaries

ENCODE_FMT = "utf-8"
UTF8_VOCAB_SIZE = 256
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
CHECKPOINT_ITERS = 2000


def byte_pair_freq_counter(pre_tok_counter: dict[tuple[bytes, ...], int]) -> dict[tuple[bytes, ...], int]:
    counts = Counter()
    for pre_tok_tuple, count in pre_tok_counter.items():
        if len(pre_tok_tuple) == 1:
            pass
        for pair in zip(pre_tok_tuple, pre_tok_tuple[1:]):
            counts[pair] += count

    return counts


def merge_bytes(pre_tok_counter: dict[tuple[bytes, ...], int], pair: tuple[bytes, ...], idx: int):
    out_counter = Counter()
    for pre_tok_tuple, count in pre_tok_counter.items():
        if pair not in zip(pre_tok_tuple, pre_tok_tuple[1:]):
            out_counter[pre_tok_tuple] = count
        else:
            out_ids = []
            i = 0
            while i < len(pre_tok_tuple):
                if i < len(pre_tok_tuple) - 1 and pre_tok_tuple[i] == pair[0] and pre_tok_tuple[i + 1] == pair[1]:
                    out_ids.append(idx)
                    i += 2
                else:
                    out_ids.append(pre_tok_tuple[i])
                    i += 1

            out_counter[tuple(out_ids)] = count

    return out_counter


def process_chunk(args: tuple[str, int, int, list[str]]) -> Counter:
    input_path, start, end, special_tokens = args
    special_token_pat = "|".join([re.escape(token) for token in special_tokens])
    pre_tok_counter = Counter()
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode(ENCODE_FMT, errors="ignore")
        # Run pre-tokenization on your chunk and store the counts for each pre-token
        sub_chunks = re.split(special_token_pat, chunk)
        for sub_chunk in sub_chunks:
            pre_tokens = re.finditer(PAT, sub_chunk)  # stream everything
            pre_tok_counter += Counter(
                tuple(pre_tok.group(0).encode(ENCODE_FMT, errors="replace")) for pre_tok in pre_tokens
            )
    return pre_tok_counter


def get_pre_token_counter(
    input_path: str,
    special_tokens: list[str],
) -> Counter:
    # pre_tok_counter = Counter()
    with open(input_path, "rb") as f:
        num_processes = max(1, os.cpu_count())
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        args = [(input_path, start, end, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])]
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            counters = list(tqdm(executor.map(process_chunk, args), total=len(args), desc="Pretokenizing"))

    return sum(counters, Counter())


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    verbose: bool = False,
    save_checkpoint: bool = False,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    vocab: dict[int, bytes] = {}
    vocab.update({idx: bytes([idx]) for idx in range(UTF8_VOCAB_SIZE)})
    if special_tokens:
        vocab.update({UTF8_VOCAB_SIZE + idx: spe_tok.encode(ENCODE_FMT) for idx, spe_tok in enumerate(special_tokens)})
    merges: list[tuple[bytes, bytes]] = []

    pre_tok_counter = get_pre_token_counter(input_path, special_tokens)
    num_merges = vocab_size - UTF8_VOCAB_SIZE - len(special_tokens)

    for i in tqdm(range(num_merges)):
        counts = byte_pair_freq_counter(pre_tok_counter)
        byte_pair = max(counts, key=lambda k: (counts.get(k), (vocab[k[0]], vocab[k[1]])))  # tiebreak with vocab order
        idx = UTF8_VOCAB_SIZE + len(special_tokens) + i
        if verbose:
            print(f"merge ({vocab[byte_pair[0]], vocab[byte_pair[1]]}) -> {idx} (count: {counts.get(byte_pair)})")
        pre_tok_counter = merge_bytes(pre_tok_counter, byte_pair, idx)
        vocab[idx] = vocab[byte_pair[0]] + vocab[byte_pair[1]]
        merges.append((vocab[byte_pair[0]], vocab[byte_pair[1]]))

        if save_checkpoint and i % CHECKPOINT_ITERS == 0:
            print(f"Saving checkpoint at iteration {i}")
            with open(f"./data/owt-vocab-checkpoint-{i}.json", "w") as f:
                json.dump({v.decode(ENCODE_FMT, errors="replace"): k for k, v in vocab.items()}, f, indent=2)
            with open(f"./data/owt-merges-checkpoint-{i}.txt", "w") as f:
                for merge in merges:
                    f.write(
                        f"{merge[0].decode(ENCODE_FMT, errors='replace')} {merge[1].decode(ENCODE_FMT, errors='replace')}\n"
                    )

    return vocab, merges
