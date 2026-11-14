"""Datasets: the built-in Bangla set, or your own JSONL files."""
from __future__ import annotations

import json
from pathlib import Path


def load_tydiqa_bengali() -> tuple[list[str], list[str], list[int]]:
    """Bengali portion of TyDi QA GoldP (Clark et al., 2020; Apache-2.0).

    Every question was written by a native speaker and is paired with the
    Wikipedia paragraph that answers it. The corpus is the set of distinct
    paragraphs; each question's relevant passage is its own paragraph.
    """
    from datasets import load_dataset

    corpus: list[str] = []
    index: dict[str, int] = {}
    queries: list[str] = []
    gold: list[int] = []
    for split in ("train", "validation"):
        for row in load_dataset("google-research-datasets/tydiqa", "secondary_task", split=split):
            if not row["id"].startswith("bengali"):
                continue
            passage = row["context"].strip()
            if passage not in index:
                index[passage] = len(corpus)
                corpus.append(passage)
            queries.append(row["question"].strip())
            gold.append(index[passage])
    return corpus, queries, gold


def load_jsonl(corpus_path: str | Path, queries_path: str | Path) -> tuple[list[str], list[str], list[int]]:
    """Your own data. corpus: {"id": ..., "text": ...} per line; queries: {"text": ..., "doc_id": ...}."""
    read = lambda p: [json.loads(line) for line in Path(p).read_text(encoding="utf-8").splitlines() if line.strip()]
    docs = read(corpus_path)
    position = {d["id"]: i for i, d in enumerate(docs)}
    rows = read(queries_path)
    return [d["text"] for d in docs], [q["text"] for q in rows], [position[q["doc_id"]] for q in rows]
