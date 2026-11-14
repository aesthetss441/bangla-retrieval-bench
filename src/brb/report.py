"""Merge per-system runs into a leaderboard, add hybrids, and write the outputs."""
from __future__ import annotations

import json
from pathlib import Path

from .metrics import evaluate, ranks, rrf


def build(run_dir: Path) -> dict:
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(run_dir.glob("*.json"))]
    if not runs:
        raise SystemExit(f"no runs in {run_dir}")
    meta = {k: runs[0][k] for k in ("dataset", "n_passages", "n_queries", "environment")}
    gold = runs[0]["gold"]
    rows = []
    for r in runs:
        assert r["gold"] == gold, "runs were made on different data"
        rows.append({k: v for k, v in r.items() if k not in ("rankings", "gold", "dataset", "n_passages", "n_queries", "environment")}
                    | evaluate(r["rankings"], gold))
    by_name = {r["system"]: r for r in runs}

    # Hybrid = reciprocal rank fusion of the best lexical run with each dense run.
    sparse = [r for r in rows if r["kind"] == "bm25"]
    if sparse:
        best_sparse = max(sparse, key=lambda r: r["mrr@10"])
        for r in [r for r in rows if r["kind"] == "dense"]:
            fused = [rrf([a, b]) for a, b in zip(by_name[best_sparse["system"]]["rankings"], by_name[r["system"]]["rankings"])]
            rows.append({"system": f"hybrid:{r['system']}", "kind": "hybrid", "label": f"{r['label']} + {best_sparse['label']} (RRF)",
                         "query_ms": round(r["query_ms"] + best_sparse["query_ms"], 3),
                         "index_seconds": round(r["index_seconds"] + best_sparse["index_seconds"], 2)} | evaluate(fused, gold))
            by_name[f"hybrid:{r['system']}"] = {"rankings": fused}
    rows.sort(key=lambda r: -r["ndcg@10"])

    # Failure analysis: how many questions does nobody answer, and what does the best lexical run add to the best dense run?
    all_ranks = {name: ranks(by_name[name]["rankings"], gold) for name in by_name}
    hit = {name: (r > 0) & (r <= 10) for name, r in all_ranks.items()}
    analysis = {"missed_by_every_system@10": int((~_any(hit.values())).sum())}
    dense = [r for r in rows if r["kind"] == "dense"]
    if sparse and dense:
        d, s = max(dense, key=lambda r: r["mrr@10"])["system"], max(sparse, key=lambda r: r["mrr@10"])["system"]
        analysis |= {"best_dense": d, "best_sparse": s,
                     "dense_only@10": int((hit[d] & ~hit[s]).sum()), "sparse_only@10": int((hit[s] & ~hit[d]).sum()),
                     "both@10": int((hit[d] & hit[s]).sum()), "neither@10": int((~hit[d] & ~hit[s]).sum())}
    return {**meta, "systems": rows, "analysis": analysis}


def _any(arrays):
    arrays = list(arrays)
    out = arrays[0].copy()
    for a in arrays[1:]:
        out |= a
    return out


def markdown(result: dict) -> str:
    lines = [
        "# Results",
        "",
        f"{result['dataset']}: {result['n_queries']:,} questions over {result['n_passages']:,} passages. One relevant passage per question.",
        f"Measured on {result['environment']}. Timings are CPU-only and vary between runs; ranking metrics are deterministic.",
        "",
        "| Retriever | nDCG@10 | MRR@10 (95% CI) | R@1 | R@5 | R@10 | ms/query | index s |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in result["systems"]:
        lo, hi = r["mrr@10_ci95"]
        lines.append(f"| {r['label']} | {r['ndcg@10']:.3f} | {r['mrr@10']:.3f} ({lo:.3f}–{hi:.3f}) | {r['recall@1']:.3f} | {r['recall@5']:.3f} | "
                     f"{r['recall@10']:.3f} | {r['query_ms']:.1f} | {r['index_seconds']:.0f} |")
    a = result["analysis"]
    lines += ["", "## Failure analysis", "", f"- Questions whose passage no system returned in its top 10: **{a['missed_by_every_system@10']}**"]
    if "best_dense" in a:
        lines += [f"- Best dense (`{a['best_dense']}`) vs best lexical (`{a['best_sparse']}`) at top 10: both {a['both@10']}, "
                  f"dense only {a['dense_only@10']}, lexical only {a['sparse_only@10']}, neither {a['neither@10']}"]
    return "\n".join(lines) + "\n"


def write(run_dir: Path, out_dir: Path, docs_dir: Path) -> dict:
    result = build(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(result, ensure_ascii=False, indent=1)
    (out_dir / "results.json").write_text(payload, encoding="utf-8")
    (docs_dir / "data.json").write_text(payload, encoding="utf-8")
    (out_dir / "RESULTS.md").write_text(markdown(result), encoding="utf-8")
    return result
