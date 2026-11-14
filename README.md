# bangla-retrieval-bench

[![CI](https://github.com/aesthetss441/bangla-retrieval-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/aesthetss441/bangla-retrieval-bench/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Which retriever should a Bangla RAG system use?** Most retrieval advice is measured on English. This benchmark runs BM25 with three tokenizers, five multilingual embedding models and hybrid fusion on 2,503 questions written by native Bangla speakers, and reports quality, latency and where each one fails.

**Live leaderboard: https://aesthetss441.github.io/bangla-retrieval-bench/**

Every number below was produced by a [GitHub Actions run](.github/workflows/bench.yml) and committed by the workflow itself ([results/](results)), so nothing is hand-entered.

## Results

TyDi QA GoldP, Bengali: 2,503 questions over 1,870 Wikipedia passages, one relevant passage per question. CPU only.

| Retriever | nDCG@10 | MRR@10 | R@1 | R@10 | ms/query | index s |
|---|---|---|---|---|---|---|
| bge-m3 (dense) | **0.871** | 0.837 | 0.745 | 0.974 | 55.8 | 1055 |
| bge-m3 + BM25 char 3-grams (RRF) | 0.822 | 0.779 | 0.665 | 0.952 | 56.0 | 1056 |
| multilingual-e5-base | 0.817 | 0.772 | 0.657 | 0.951 | 22.6 | 406 |
| multilingual-e5-small | 0.815 | 0.770 | 0.656 | 0.953 | 7.3 | 138 |
| multilingual-e5-small, prefixes omitted | 0.815 | 0.770 | 0.654 | 0.952 | 6.5 | 136 |
| BM25, character 3-grams | 0.739 | 0.685 | 0.563 | 0.907 | 0.2 | 1 |
| BM25, Bangla-aware tokenizer | 0.633 | 0.582 | 0.471 | 0.793 | 0.1 | 0 |
| LaBSE | 0.547 | 0.488 | 0.377 | 0.734 | 22.0 | 438 |
| BM25, `\w+` tokenizer | 0.493 | 0.441 | 0.339 | 0.658 | 0.2 | 0 |
| paraphrase-multilingual-MiniLM-L12-v2 | 0.053 | 0.042 | 0.025 | 0.087 | 4.4 | 90 |

Full table with confidence intervals and all hybrid rows: [results/RESULTS.md](results/RESULTS.md).

## What the numbers say

1. **The default tokenizer breaks Bangla.** `re.findall(r"\w+", text)` is what most BM25 tutorials use. Bangla vowel signs are combining marks, not letters, so `\w+` cuts "বাংলাদেশ" into four fragments. Keeping words whole lifts nDCG@10 from 0.493 to 0.633, and character 3-grams (which let inflected forms like দেশ / দেশের / দেশে share terms) reach 0.739. That is a 50% relative gain with no model at all.
2. **A well-tokenized BM25 beats two of the five embedding models**, at 0.2 ms per query and a one-second index.
3. **"Multilingual" does not mean "works for Bangla".** `paraphrase-multilingual-MiniLM-L12-v2`, a very common default, scores 0.053. Bengali is not among the languages listed on its model card. I have not ruled out other causes, so treat this as "do not assume; measure".
4. **e5-small matches e5-base** (0.815 vs 0.817, overlapping intervals) at a third of the latency. It is the best value here.
5. **bge-m3 is clearly best** (0.871) but costs about 8× the query time of e5-small on CPU.
6. **Naive hybrid fusion hurt every strong model.** Equal-weight reciprocal rank fusion with the best BM25 lowered bge-m3 from 0.871 to 0.822 and e5 from 0.815 to about 0.80. It only helped the weak models. Hybrid search is not automatically better; a weaker ranker given equal weight drags a stronger one down.
7. **The e5 `query:` / `passage:` prefixes made no measurable difference** on this data (0.815 with and without).

### Failure analysis

At top 10, bge-m3 and the best BM25 both find the passage for 2,245 questions; bge-m3 alone finds 194 more, BM25 alone 26, and 38 are missed by both. Only 16 questions are missed by every system. The 26 lexical-only wins explain why fusion cannot add much here, while fusion's cost is paid on all the questions where BM25 ranks the passage lower.

## Limits

- **One dataset, small corpus.** 1,870 passages is far easier than a production corpus; absolute scores will drop at scale and the ordering may change. Passages are clean Wikipedia paragraphs, not chunked PDFs or chat logs.
- **One relevant passage per question.** Other passages that also answer a question count as wrong.
- **Dense search is exact** (brute force), so no ANN error is included.
- **Hybrid is one method** (RRF, k=60, equal weight, top 100). Weighted fusion or a reranker may behave differently and is not tested.
- **Timings** are from a shared 4-core CI runner and vary between runs. Ranking metrics are deterministic.
- All models were truncated at 512 tokens.

## Run it on your own data

This is the part meant to be reused: point it at your documents and questions before choosing a model.

```bash
pip install "bangla-retrieval-bench[dense] @ git+https://github.com/aesthetss441/bangla-retrieval-bench"

# corpus.jsonl:  {"id": "doc-1", "text": "..."}        one passage per line
# queries.jsonl: {"text": "...", "doc_id": "doc-1"}    one question per line
brb run --systems bm25-char3,e5-small,bge-m3 --corpus corpus.jsonl --queries queries.jsonl
brb report
```

`brb list` shows the retrievers. To reproduce the table above: `pip install -e ".[dense,data]" && brb run --systems all && brb report` (about 30 minutes on a 4-core CPU, dominated by bge-m3).

## How it is built

| File | What it does |
|---|---|
| [`text.py`](src/brb/text.py) | the three tokenizers |
| [`bm25.py`](src/brb/bm25.py) | BM25 with precomputed posting weights |
| [`systems.py`](src/brb/systems.py) | registry of retrievers; each returns its top 100 per query plus timings |
| [`metrics.py`](src/brb/metrics.py) | recall, MRR, nDCG, bootstrap confidence intervals, RRF |
| [`report.py`](src/brb/report.py) | merges runs, builds hybrids, failure analysis, writes Markdown and JSON |

Each retriever runs as its own CI job and uploads its rankings; a final job scores them together and commits the results, which the leaderboard page reads.

## Data

[TyDi QA](https://github.com/google-research-datasets/tydiqa) (Clark et al., 2020), gold-passage task, Bengali subset, Apache-2.0. Downloaded at run time, not redistributed here.

MIT licensed.
