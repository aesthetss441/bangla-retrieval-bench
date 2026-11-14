"""The retrievers under test. Each returns its top-100 passages per query plus timings."""
from __future__ import annotations

import time

import numpy as np

from .bm25 import BM25
from .text import TOKENIZERS

TOP = 100

SYSTEMS: dict[str, dict] = {
    "bm25-naive": {"kind": "bm25", "tokenizer": "naive", "label": "BM25, \\w+ tokenizer"},
    "bm25-bangla": {"kind": "bm25", "tokenizer": "bangla", "label": "BM25, Bangla-aware tokenizer"},
    "bm25-char3": {"kind": "bm25", "tokenizer": "char3", "label": "BM25, character 3-grams"},
    "minilm": {"kind": "dense", "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "label": "paraphrase-multilingual-MiniLM-L12-v2"},
    "labse": {"kind": "dense", "model": "sentence-transformers/LaBSE", "label": "LaBSE"},
    "e5-small": {"kind": "dense", "model": "intfloat/multilingual-e5-small", "query_prefix": "query: ", "passage_prefix": "passage: ", "label": "multilingual-e5-small"},
    "e5-small-noprefix": {"kind": "dense", "model": "intfloat/multilingual-e5-small", "label": "multilingual-e5-small, prefixes omitted"},
    "e5-base": {"kind": "dense", "model": "intfloat/multilingual-e5-base", "query_prefix": "query: ", "passage_prefix": "passage: ", "label": "multilingual-e5-base"},
    "bge-m3": {"kind": "dense", "model": "BAAI/bge-m3", "label": "bge-m3 (dense)"},
}


def run(name: str, corpus: list[str], queries: list[str]) -> dict:
    cfg = SYSTEMS[name]
    if cfg["kind"] == "bm25":
        tokenize = TOKENIZERS[cfg["tokenizer"]]
        start = time.perf_counter()
        index = BM25([tokenize(p) for p in corpus])
        index_seconds = time.perf_counter() - start
        start = time.perf_counter()
        rankings = [index.top(tokenize(q), TOP) for q in queries]
        query_ms = 1000 * (time.perf_counter() - start) / len(queries)
        extra = {"vocabulary": len(index.postings)}
    else:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(cfg["model"], device="cpu")
        model.max_seq_length = 512  # same truncation for every model; some default to 128
        encode = lambda texts, prefix: model.encode(
            [prefix + t for t in texts], batch_size=16, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        start = time.perf_counter()
        passages = encode(corpus, cfg.get("passage_prefix", ""))
        index_seconds = time.perf_counter() - start
        start = time.perf_counter()
        q = encode(queries, cfg.get("query_prefix", ""))
        scores = q @ passages.T  # exact search: the corpus is small, so no ANN error is mixed in
        rankings = np.argsort(-scores, axis=1)[:, :TOP].tolist()
        query_ms = 1000 * (time.perf_counter() - start) / len(queries)
        extra = {"dimensions": int(passages.shape[1]), "parameters_millions": round(sum(p.numel() for p in model.parameters()) / 1e6)}
    return {"system": name, **cfg, **extra, "index_seconds": round(index_seconds, 2), "query_ms": round(query_ms, 3), "rankings": rankings}
