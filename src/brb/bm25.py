"""Okapi BM25 with precomputed per-posting weights."""
from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np


class BM25:
    def __init__(self, documents: list[list[str]], k1: float = 1.2, b: float = 0.75):
        self.n = len(documents)
        lengths = np.array([len(d) for d in documents], dtype=np.float32)
        norm = k1 * (1.0 - b + b * lengths / max(float(lengths.mean()) if self.n else 0.0, 1e-9))
        postings: dict[str, tuple[list[int], list[int]]] = defaultdict(lambda: ([], []))
        for doc_id, tokens in enumerate(documents):
            for term, tf in Counter(tokens).items():
                postings[term][0].append(doc_id)
                postings[term][1].append(tf)
        self.postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for term, (ids, tfs) in postings.items():
            idx = np.asarray(ids, dtype=np.int64)
            tf = np.asarray(tfs, dtype=np.float32)
            idf = math.log(1.0 + (self.n - len(ids) + 0.5) / (len(ids) + 0.5))
            self.postings[term] = (idx, idf * tf * (k1 + 1.0) / (tf + norm[idx]))

    def top(self, query: list[str], k: int) -> list[int]:
        scores = np.zeros(self.n, dtype=np.float32)
        for term in set(query):
            hit = self.postings.get(term)
            if hit:
                scores[hit[0]] += hit[1]
        k = min(k, self.n)
        return np.argsort(-scores, kind="stable")[:k].tolist()
