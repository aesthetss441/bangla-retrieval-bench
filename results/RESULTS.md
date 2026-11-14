# Results

TyDi QA GoldP, Bengali: 2,503 questions over 1,870 passages. One relevant passage per question.
Measured on x86_64, Python 3.12.16. Timings are CPU-only and vary between runs; ranking metrics are deterministic.

| Retriever | nDCG@10 | MRR@10 (95% CI) | R@1 | R@5 | R@10 | ms/query | index s |
|---|---|---|---|---|---|---|---|
| bge-m3 (dense) | 0.871 | 0.837 (0.825–0.848) | 0.745 | 0.952 | 0.974 | 55.8 | 1055 |
| bge-m3 (dense) + BM25, character 3-grams (RRF) | 0.822 | 0.779 (0.766–0.792) | 0.665 | 0.922 | 0.952 | 56.0 | 1056 |
| multilingual-e5-base | 0.817 | 0.772 (0.758–0.785) | 0.657 | 0.925 | 0.951 | 22.6 | 406 |
| multilingual-e5-small | 0.815 | 0.770 (0.756–0.782) | 0.656 | 0.918 | 0.953 | 7.3 | 138 |
| multilingual-e5-small, prefixes omitted | 0.815 | 0.770 (0.755–0.782) | 0.654 | 0.921 | 0.952 | 6.5 | 136 |
| multilingual-e5-small, prefixes omitted + BM25, character 3-grams (RRF) | 0.803 | 0.755 (0.741–0.768) | 0.636 | 0.907 | 0.947 | 6.7 | 137 |
| multilingual-e5-base + BM25, character 3-grams (RRF) | 0.802 | 0.755 (0.741–0.768) | 0.636 | 0.910 | 0.945 | 22.8 | 407 |
| multilingual-e5-small + BM25, character 3-grams (RRF) | 0.801 | 0.754 (0.741–0.767) | 0.634 | 0.909 | 0.944 | 7.6 | 139 |
| BM25, character 3-grams | 0.739 | 0.685 (0.669–0.699) | 0.563 | 0.847 | 0.907 | 0.2 | 1 |
| LaBSE + BM25, character 3-grams (RRF) | 0.696 | 0.640 (0.624–0.657) | 0.515 | 0.799 | 0.871 | 22.2 | 438 |
| BM25, Bangla-aware tokenizer | 0.633 | 0.582 (0.565–0.598) | 0.471 | 0.728 | 0.793 | 0.1 | 0 |
| LaBSE | 0.547 | 0.488 (0.472–0.506) | 0.377 | 0.646 | 0.734 | 22.0 | 438 |
| BM25, \w+ tokenizer | 0.493 | 0.441 (0.425–0.457) | 0.339 | 0.579 | 0.658 | 0.2 | 0 |
| paraphrase-multilingual-MiniLM-L12-v2 + BM25, character 3-grams (RRF) | 0.348 | 0.251 (0.239–0.264) | 0.131 | 0.395 | 0.670 | 4.7 | 91 |
| paraphrase-multilingual-MiniLM-L12-v2 | 0.053 | 0.042 (0.036–0.049) | 0.025 | 0.064 | 0.087 | 4.4 | 90 |

## Failure analysis

- Questions whose passage no system returned in its top 10: **16**
- Best dense (`bge-m3`) vs best lexical (`bm25-char3`) at top 10: both 2245, dense only 194, lexical only 26, neither 38
