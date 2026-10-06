"""Pure text helpers: dedup normalisation, alphanumeric tokens and hashed n-gram matching."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import numpy as np

NGRAM_N = 13
_NON_ALNUM = re.compile(r"[\W_]+")
_WS = re.compile(r"\s+")
_MULT = np.array(
    [
        int.from_bytes(hashlib.blake2b(f"mult{k}".encode(), digest_size=8).digest(), "little") | 1
        for k in range(NGRAM_N)
    ],
    dtype=np.uint64,
)
_TOKEN_CACHE: dict[str, int] = {}
_TOKEN_CACHE_MAX = 4_000_000


def dedup_key(text: str) -> str:
    """Normalisation used for de-duplication and item ids: lowercase, collapsed whitespace."""
    return _WS.sub(" ", text.lower()).strip()


def tokens(text: str) -> list[str]:
    """Lowercase, non-alphanumerics replaced by spaces, split on whitespace."""
    return _NON_ALNUM.sub(" ", text.lower()).split()


def hash64(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little")


def _token_hash(tok: str) -> int:
    h = _TOKEN_CACHE.get(tok)
    if h is None:
        if len(_TOKEN_CACHE) >= _TOKEN_CACHE_MAX:
            _TOKEN_CACHE.clear()
        h = _TOKEN_CACHE[tok] = hash64(tok)
    return h


def ngram_hashes(toks: list[str], n: int = NGRAM_N) -> np.ndarray:
    """64-bit hash of every n-gram of `toks`, in order (position-sensitive, deterministic)."""
    if len(toks) < n:
        return np.empty(0, dtype=np.uint64)
    t = np.fromiter((_token_hash(w) for w in toks), dtype=np.uint64, count=len(toks))
    m = len(toks) - n + 1
    out = np.zeros(m, dtype=np.uint64)
    for k in range(n):
        out += t[k : k + m] * _MULT[k]
    return out


@dataclass(frozen=True)
class NgramIndex:
    """Sorted n-gram hashes with a parallel array of the item index each came from."""

    keys: np.ndarray  # uint64, sorted
    items: np.ndarray  # int32, same order as keys

    @classmethod
    def build(cls, token_lists: list[list[str]], n: int = NGRAM_N) -> NgramIndex:
        keys, items = [], []
        for i, toks in enumerate(token_lists):
            h = np.unique(ngram_hashes(toks, n))
            keys.append(h)
            items.append(np.full(len(h), i, dtype=np.int32))
        k = np.concatenate(keys) if keys else np.empty(0, dtype=np.uint64)
        it = np.concatenate(items) if items else np.empty(0, dtype=np.int32)
        order = np.argsort(k, kind="stable")
        return cls(k[order], it[order])

    def lookup(self, toks: list[str], n: int = NGRAM_N) -> list[tuple[int, int]]:
        """Return (position in `toks`, item index) for every n-gram that is in the index."""
        h = ngram_hashes(toks, n)
        if not len(h) or not len(self.keys):
            return []
        lo = np.searchsorted(self.keys, h, side="left")
        hi = np.searchsorted(self.keys, h, side="right")
        out = []
        for pos in np.nonzero(hi > lo)[0]:
            out.extend((int(pos), int(i)) for i in self.items[lo[pos] : hi[pos]])
        return out

    def items_for(self, key: int) -> list[int]:
        """Item indices whose hash equals `key` (used for exact full-text / prefix matches)."""
        k = np.uint64(key)
        lo, hi = np.searchsorted(self.keys, k, "left"), np.searchsorted(self.keys, k, "right")
        return [int(i) for i in self.items[lo:hi]]

    @classmethod
    def from_hashes(cls, hashes: list[int]) -> NgramIndex:
        k = np.array(hashes, dtype=np.uint64)
        order = np.argsort(k, kind="stable")
        return cls(k[order], np.arange(len(hashes), dtype=np.int32)[order])
