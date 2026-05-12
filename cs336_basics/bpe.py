from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import regex as re
from tqdm import tqdm

from cs336_basics.pretokenization_example import find_chunk_boundaries

ENCODE_FMT = "utf-8"
UTF8_VOCAB_SIZE = 256
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


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
    pre_tok_counter = Counter()
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode(ENCODE_FMT, errors="ignore")
        # Run pre-tokenization on your chunk and store the counts for each pre-token
        escaped_tokens = [re.escape(token) for token in special_tokens]
        special_token_pat = "|".join(escaped_tokens)
        sub_chunks = re.split(special_token_pat, chunk)
        for sub_chunk in sub_chunks:
            pre_tokens = re.findall(PAT, sub_chunk)
            pre_tok_byte_tuple = [tuple(pre_tok.encode(ENCODE_FMT)) for pre_tok in pre_tokens]
            pre_tok_counter += Counter(pre_tok_byte_tuple)
    return pre_tok_counter


def get_pre_token_counter(
    input_path: str,
    special_tokens: list[str],
) -> Counter:
    # pre_tok_counter = Counter()
    with open(input_path, "rb") as f:
        num_processes = 8
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        #     f.seek(start)
        #     chunk = f.read(end - start).decode(ENCODE_FMT, errors="ignore")
        #     # Run pre-tokenization on your chunk and store the counts for each pre-token
        #     escaped_tokens = [re.escape(token) for token in special_tokens]
        #     special_token_pat = "|".join(escaped_tokens)
        #     sub_chunks = re.split(special_token_pat, chunk)
        #     for sub_chunk in sub_chunks:
        #         pre_tokens = re.findall(PAT, sub_chunk)
        #         pre_tok_byte_tuple = [tuple(pre_tok.encode(ENCODE_FMT)) for pre_tok in pre_tokens]
        #         pre_tok_counter += Counter(pre_tok_byte_tuple)
        args = [(input_path, start, end, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])]
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            counters = list(tqdm(executor.map(process_chunk, args), total=len(args), desc="Pretokenizing"))

    return sum(counters, Counter())


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    verbose: bool = False,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    vocab: dict[int, bytes] = {}
    if special_tokens:
        for i, spe_tok in enumerate(special_tokens):
            vocab[int(UTF8_VOCAB_SIZE + i)] = spe_tok.encode(ENCODE_FMT)
    vocab.update({idx: bytes([idx]) for idx in range(UTF8_VOCAB_SIZE)})
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

    return vocab, merges
