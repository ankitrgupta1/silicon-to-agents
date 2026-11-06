"""A minimal byte-level BPE tokenizer, for learning.

Start from the 256 possible bytes, then repeatedly merge the most frequent
adjacent pair of tokens into a new token. Encoding replays the merges in the
order they were learned. Real tokenizers (GPT-2, tiktoken) also split text into
chunks with a regex first, so merges never cross word or punctuation
boundaries; `pretokenize=True` does a simple version of that here.
"""

from __future__ import annotations

import re
from collections import Counter

# Split into: optional leading space + letters, digits, other symbols, whitespace.
_CHUNK = re.compile(r" ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+")


def _chunks(text: str, pretokenize: bool) -> list[bytes]:
    if not pretokenize:
        return [text.encode("utf-8")]
    return [m.group().encode("utf-8") for m in _CHUNK.finditer(text)]


def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    def __init__(self, pretokenize: bool = True) -> None:
        self.pretokenize = pretokenize
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def train(self, text: str, vocab_size: int) -> "BPETokenizer":
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256 (one token per byte)")
        words = [list(c) for c in _chunks(text, self.pretokenize)]
        for new_id in range(256, vocab_size):
            counts: Counter[tuple[int, int]] = Counter()
            for w in words:
                counts.update(zip(w, w[1:]))
            if not counts:
                break  # nothing left to merge
            pair = max(counts, key=counts.get)  # ties: first seen wins
            words = [_merge(w, pair, new_id) for w in words]
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
        return self

    def encode(self, text: str) -> list[int]:
        out: list[int] = []
        for chunk in _chunks(text, self.pretokenize):
            ids = list(chunk)
            while len(ids) >= 2:
                # Apply the earliest-learned merge available in this chunk.
                pair = min(zip(ids, ids[1:]), key=lambda p: self.merges.get(p, float("inf")))
                if pair not in self.merges:
                    break
                ids = _merge(ids, pair, self.merges[pair])
            out.extend(ids)
        return out

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def tokens(self, text: str) -> list[str]:
        """The text pieces each token stands for (handy for inspection)."""
        return [self.vocab[i].decode("utf-8", errors="replace") for i in self.encode(text)]
