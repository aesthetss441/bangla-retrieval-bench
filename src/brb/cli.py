"""brb run | report | list"""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

from . import report
from .systems import SYSTEMS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="brb", description="Bangla retrieval benchmark")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="show the available retrievers")
    run = sub.add_parser("run", help="run retrievers and save their rankings")
    run.add_argument("--systems", required=True, help="comma-separated names, or 'all'")
    run.add_argument("--out", type=Path, default=Path("runs"))
    run.add_argument("--corpus", type=Path, help="your own corpus JSONL (default: TyDi QA Bengali)")
    run.add_argument("--queries", type=Path, help="your own queries JSONL")
    run.add_argument("--limit", type=int, help="use only the first N queries (for a quick look)")
    rep = sub.add_parser("report", help="score saved runs and write the leaderboard")
    rep.add_argument("--runs", type=Path, default=Path("runs"))
    rep.add_argument("--out", type=Path, default=Path("results"))
    rep.add_argument("--docs", type=Path, default=Path("docs"))
    args = parser.parse_args(argv)

    if args.command == "list":
        for name, cfg in SYSTEMS.items():
            print(f"{name:<20} {cfg['label']}")
        return 0
    if args.command == "report":
        print(report.markdown(report.write(args.runs, args.out, args.docs)))
        return 0

    from . import data, systems
    if args.corpus and args.queries:
        corpus, queries, gold = data.load_jsonl(args.corpus, args.queries)
        dataset = f"custom ({args.corpus.name})"
    else:
        corpus, queries, gold = data.load_tydiqa_bengali()
        dataset = "TyDi QA GoldP, Bengali"
    if args.limit:
        queries, gold = queries[: args.limit], gold[: args.limit]
    names = list(SYSTEMS) if args.systems == "all" else [s.strip() for s in args.systems.split(",")]
    args.out.mkdir(parents=True, exist_ok=True)
    for name in names:
        result = systems.run(name, corpus, queries)
        result |= {"gold": gold, "dataset": dataset, "n_passages": len(corpus), "n_queries": len(queries),
                   "environment": f"{platform.processor() or platform.machine()}, Python {platform.python_version()}"}
        (args.out / f"{name}.json").write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        print(f"{name}: {len(queries)} queries, index {result['index_seconds']}s, {result['query_ms']} ms/query", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
