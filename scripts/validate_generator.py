"""Validate that `Generator` produces parseable `QueryResponse` objects.

Runs the full retrieval → generation pipeline on the first N questions
from `data/golden/golden_set_v1.jsonl` and reports, per query:

- whether the Ollama response parsed cleanly into `QueryResponse`
- whether the model cited at least one chunk
- whether all cited `chunk_id`s were actually present in the retrieved context
  (i.e. whether the model is grounded or hallucinating references)
- generation latency

The exit code is non-zero if any query failed to parse, so this script
doubles as the acceptance gate for `garag-zqc.17` ("QueryResponse
валидно парсится на 20 запросах").
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from app.config import settings
from app.rag.generator import GenerationError, Generator
from app.rag.pipeline import HybridRetriever
from app.rag.reranker import Reranker

GOLDEN_FILE = Path(__file__).resolve().parents[1] / "data" / "golden" / "golden_set_v1.jsonl"


def _load_golden(path: Path, limit: int) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with path.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            items.append(json.loads(line))
            if len(items) >= limit:
                break
    return items


def main() -> int:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--golden", type=Path, default=GOLDEN_FILE)
    ap.add_argument("--limit", type=int, default=20, help="number of queries to run")
    ap.add_argument("--candidate-k", type=int, default=settings.retrieve_top_k)
    ap.add_argument("--top-k", type=int, default=settings.rerank_top_k)
    ap.add_argument("--model", type=str, default=None, help="override settings.ollama_model")
    ap.add_argument("--no-rerank", action="store_true", help="skip cross-encoder reranker")
    args = ap.parse_args()

    golden = _load_golden(args.golden, args.limit)
    print(f"[load] {len(golden)} queries from {args.golden}")

    print("[init] hybrid retriever (dense + sparse + alpha fusion)")
    reranker = None if args.no_rerank else Reranker()
    retriever = HybridRetriever(
        reranker=reranker,
        fusion=settings.fusion_method,
        alpha=settings.fusion_alpha,
    )

    print(f"[init] generator model={args.model or settings.ollama_model}")
    failures: list[tuple[str, str]] = []
    rows: list[dict[str, Any]] = []

    with Generator(model=args.model) as gen:
        for item in golden:
            qid = item["qid"]
            question = item["question"]
            chunks = retriever.retrieve(question, candidate_k=args.candidate_k, top_k=args.top_k)
            retrieved_ids = {c.chunk_id for c in chunks}

            t0 = time.perf_counter()
            try:
                resp = gen.generate(question, chunks)
            except GenerationError as exc:
                elapsed = time.perf_counter() - t0
                failures.append((qid, str(exc)))
                rows.append(
                    {
                        "qid": qid,
                        "parsed": False,
                        "error": str(exc),
                        "latency_s": round(elapsed, 2),
                    }
                )
                print(f"[FAIL] {qid} latency={elapsed:.1f}s err={exc}")
                continue
            elapsed = time.perf_counter() - t0

            cited = list(resp.used_chunks)
            grounded = all(cid in retrieved_ids for cid in cited)
            rows.append(
                {
                    "qid": qid,
                    "parsed": True,
                    "latency_s": round(elapsed, 2),
                    "confidence": resp.confidence,
                    "n_citations": len(resp.citations),
                    "grounded": grounded,
                    "answer_preview": resp.answer[:120],
                }
            )
            flag = "OK" if grounded else "UNGROUNDED"
            print(
                f"[{flag}] {qid} latency={elapsed:.1f}s "
                f"conf={resp.confidence:.2f} cites={len(resp.citations)}"
            )

    parsed_count = sum(1 for r in rows if r["parsed"])
    grounded_count = sum(1 for r in rows if r.get("grounded"))
    latencies = sorted(r["latency_s"] for r in rows)
    p50 = latencies[len(latencies) // 2] if latencies else 0.0
    p95 = latencies[int(len(latencies) * 0.95)] if latencies else 0.0

    print()
    print("=" * 60)
    print(f"Parsed     : {parsed_count}/{len(rows)}")
    print(f"Grounded   : {grounded_count}/{parsed_count}")
    print(f"Latency p50: {p50:.1f}s")
    print(f"Latency p95: {p95:.1f}s")
    print("=" * 60)

    report = Path(__file__).resolve().parents[1] / "evaluation" / "reports" / "generator_smoke.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("w") as fh:
        fh.write("# Generator smoke test\n\n")
        fh.write(
            f"Ran `{args.model or settings.ollama_model}` over first "
            f"{len(rows)} queries of `golden_set_v1.jsonl` via the full "
            "hybrid retrieval + reranker pipeline.\n\n"
        )
        fh.write(f"- Parsed: **{parsed_count}/{len(rows)}**\n")
        fh.write(f"- Grounded citations: **{grounded_count}/{parsed_count}**\n")
        fh.write(f"- Latency p50 / p95: **{p50:.1f}s / {p95:.1f}s**\n\n")
        fh.write("| qid | parsed | grounded | conf | cites | latency (s) |\n")
        fh.write("|---|---|---|---|---|---|\n")
        for r in rows:
            fh.write(
                f"| {r['qid']} "
                f"| {'✓' if r['parsed'] else '✗'} "
                f"| {'✓' if r.get('grounded') else '✗'} "
                f"| {r.get('confidence', 0):.2f} "
                f"| {r.get('n_citations', 0)} "
                f"| {r['latency_s']:.1f} |\n"
            )
    print(f"[done] wrote {report}")

    if failures:
        print(f"\n[FAIL] {len(failures)} queries failed to parse", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
