"""Ranking metrics for queries that each have exactly one relevant passage."""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Sequence

import numpy as np


def ranks(rankings: Sequence[Sequence[int]], gold: Sequence[int]) -> np.ndarray:
    """1-based rank of the relevant passage per query, 0 when it was not retrieved."""
    out = np.zeros(len(gold), dtype=np.int64)
    for i, (ranking, g) in enumerate(zip(rankings, gold)):
        for position, doc in enumerate(ranking, start=1):
            if doc == g:
                out[i] = position
                break
    return out


def evaluate(rankings: Sequence[Sequence[int]], gold: Sequence[int], seed: int = 0) -> dict:
    r = ranks(rankings, gold)
    hit10 = (r > 0) & (r <= 10)
    rr = np.where(hit10, 1.0 / np.maximum(r, 1), 0.0)
    ndcg = np.where(hit10, 1.0 / np.log2(np.maximum(r, 1) + 1.0), 0.0)  # one relevant doc, so ideal DCG is 1
    # 95% bootstrap interval on MRR@10: resample queries with replacement.
    rng = np.random.default_rng(seed)
    means = rr[rng.integers(0, len(rr), size=(1000, len(rr)))].mean(axis=1)
    return {
        "recall@1": float((r == 1).mean()),
        "recall@5": float(((r > 0) & (r <= 5)).mean()),
        "recall@10": float(hit10.mean()),
        "mrr@10": float(rr.mean()),
        "mrr@10_ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
        "ndcg@10": float(ndcg.mean()),
    }


def rrf(rankings: Sequence[Sequence[int]], k: int = 60, top: int = 100) -> list[int]:
    """Reciprocal rank fusion: score(d) = sum of 1 / (k + rank) over the input rankings."""
    scores: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            scores[doc] += 1.0 / (k + rank)
    return sorted(scores, key=lambda d: (-scores[d], d))[:top]
