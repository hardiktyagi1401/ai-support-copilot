# eval/run_eval.py
"""
RAG Evaluation Runner

Usage:
    python eval/run_eval.py
    python eval/run_eval.py --dataset eval/datasets/golden.json
    python eval/run_eval.py --top-k 10 --threshold 0.60
    python eval/run_eval.py --save-results

The script hits your running FastAPI server on localhost:8001.
Start the server before running evals.
"""

import argparse
import asyncio
import json
import statistics
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import httpx

from eval.metrics.retrieval import RetrievalMetrics, compute_retrieval_metrics
from eval.metrics.generation import GenerationMetrics, compute_generation_metrics

API_BASE = "http://localhost:8001/api/v1"
DEFAULT_DATASET = "eval/datasets/golden.json"
RESULTS_DIR = Path("eval/results")


@dataclass
class EvalCase:
    query_id: str
    query: str
    category: str
    retrieval: RetrievalMetrics
    generation: GenerationMetrics
    latency_ms: int
    top_score: float | None


@dataclass
class EvalSummary:
    run_id: str
    timestamp: str
    total_queries: int
    config: dict

    # Retrieval aggregate metrics
    mean_recall_at_k: float
    mean_mrr: float
    mean_precision_at_k: float

    # Generation aggregate metrics
    answer_rate: float           # fraction of answerable queries that were answered
    refusal_accuracy: float      # fraction of unanswerable queries correctly refused
    mean_citation_rate: float
    mean_term_coverage: float
    forbidden_term_rate: float   # fraction of answers with forbidden terms

    # Latency
    mean_latency_ms: float
    p95_latency_ms: float

    # Per-query results for inspection
    cases: list[dict]


async def run_single(
    client: httpx.AsyncClient,
    case: dict,
    top_k: int,
    threshold: float | None,
) -> EvalCase:
    payload = {
        "query": case["query"],
        "tenant_id": case["tenant_id"],
        "top_k": top_k,
    }
    if threshold is not None:
        payload["score_threshold"] = threshold

    resp = await client.post(f"{API_BASE}/query/ask", json=payload)
    resp.raise_for_status()
    data = resp.json()

    # Extract retrieval chunks for metric computation
    # We rebuild chunk list from citation data (available in response)
    # In a richer setup, use the /query/inspect endpoint for full chunk data
    retrieved_chunks = [
        {"filename": c["filename"], "passed_threshold": True}
        for c in data.get("citations", [])
    ]

    retrieval_metrics = compute_retrieval_metrics(
        retrieved_chunks=retrieved_chunks,
        relevant_filenames=case.get("relevant_filenames", []),
        top_k=top_k,
    )

    gen_metrics = compute_generation_metrics(
    answer=data.get("answer"),
    was_answered=data.get("answered", False),
    citations=data.get("citations", []),
    citation_coverage=data.get("citation_coverage"),
    expected_contains=case.get("expected_contains", []),
    should_not_contain=case.get("should_not_contain", []),
    should_be_refused=case.get("should_be_refused", False),   
    )

    return EvalCase(
    query_id=case["id"],
    query=case["query"],
    category=case.get("category", "unknown"),
    retrieval=retrieval_metrics,
    generation=gen_metrics,
    latency_ms=data.get("latency", {}).get("total_ms", 0),
    top_score=data.get("retrieval_stats", {}).get("top_similarity_score"),
)


def summarize(cases: list[EvalCase], config: dict) -> EvalSummary:
    answerable = [c for c in cases if not c.generation.correctly_refused is not None or c.generation.answered]
    unanswerable = [c for c in cases if c.generation.correctly_refused is not None]
    answered_cases = [c for c in cases if c.generation.answered]

    recalls = [c.retrieval.recall_at_k for c in cases if c.retrieval.chunks_relevant > 0]
    mrrs = [c.retrieval.mrr for c in cases if c.retrieval.chunks_relevant > 0]
    precisions = [c.retrieval.precision_at_k for c in cases if c.retrieval.chunks_relevant > 0]

    latencies = [c.latency_ms for c in cases]
    sorted_lat = sorted(latencies)
    p95_idx = int(len(sorted_lat) * 0.95)

    refusal_correct = [c for c in unanswerable if c.generation.correctly_refused]

    return EvalSummary(
        run_id=datetime.now().strftime("%Y%m%d_%H%M%S"),
        timestamp=datetime.now().isoformat(),
        total_queries=len(cases),
        config=config,
        mean_recall_at_k=round(statistics.mean(recalls), 3) if recalls else 0.0,
        mean_mrr=round(statistics.mean(mrrs), 3) if mrrs else 0.0,
        mean_precision_at_k=round(statistics.mean(precisions), 3) if precisions else 0.0,
        answer_rate=round(
            len(answered_cases) / max(len(answerable), 1), 3
        ),
        refusal_accuracy=round(
            len(refusal_correct) / max(len(unanswerable), 1), 3
        ),
        mean_citation_rate=round(
            statistics.mean(c.generation.citation_rate for c in answered_cases), 3
        ) if answered_cases else 0.0,
        mean_term_coverage=round(
            statistics.mean(c.generation.term_coverage for c in answered_cases), 3
        ) if answered_cases else 0.0,
        forbidden_term_rate=round(
            sum(1 for c in answered_cases if c.generation.forbidden_term_present)
            / max(len(answered_cases), 1), 3
        ),
        mean_latency_ms=round(statistics.mean(latencies), 0) if latencies else 0,
        p95_latency_ms=sorted_lat[p95_idx] if sorted_lat else 0,
        cases=[asdict(c) for c in cases],
    )


def print_summary(summary: EvalSummary) -> None:
    print("\n" + "═" * 60)
    print(f"  RAG EVAL RESULTS — {summary.timestamp[:16]}")
    print("═" * 60)
    print(f"\n  Queries evaluated:    {summary.total_queries}")
    print(f"\n  ── Retrieval ──────────────────────────────")
    print(f"  Recall@K:             {summary.mean_recall_at_k:.1%}")
    print(f"  MRR:                  {summary.mean_mrr:.3f}")
    print(f"  Precision@K:          {summary.mean_precision_at_k:.1%}")
    print(f"\n  ── Generation ─────────────────────────────")
    print(f"  Answer rate:          {summary.answer_rate:.1%}")
    print(f"  Refusal accuracy:     {summary.refusal_accuracy:.1%}")
    print(f"  Citation rate:        {summary.mean_citation_rate:.1%}")
    print(f"  Term coverage:        {summary.mean_term_coverage:.1%}")
    print(f"  Forbidden term rate:  {summary.forbidden_term_rate:.1%}")
    print(f"\n  ── Latency ────────────────────────────────")
    print(f"  Mean:                 {summary.mean_latency_ms:.0f}ms")
    print(f"  p95:                  {summary.p95_latency_ms}ms")
    print("\n  ── Per-query breakdown ────────────────────")

    for c in summary.cases:
        ret = c["retrieval"]
        gen = c["generation"]
        status = "✓" if gen["answered"] else ("✓ refused" if gen.get("correctly_refused") else "✗")
        print(f"  [{status}] {c['query_id']:8} recall={ret['recall_at_k']:.0%} "
              f"coverage={gen['term_coverage']:.0%} {c['query'][:45]}")

    print("\n" + "═" * 60)

    # Flag issues
    if summary.mean_recall_at_k < 0.7:
        print("  ⚠ Low recall — retrieval is missing relevant chunks")
        print("    → check threshold, try lower score_threshold")
    if summary.refusal_accuracy < 0.8:
        print("  ⚠ Poor refusal accuracy — system answers when it shouldn't")
        print("    → raise score_threshold")
    if summary.forbidden_term_rate > 0.1:
        print("  ⚠ Cross-domain contamination detected")
        print("    → check tenant isolation in ChromaDB queries")
    if summary.mean_citation_rate < 0.5:
        print("  ⚠ Low citation rate — answers not grounded in sources")
        print("    → review system prompt citation instructions")
    print()


async def main(args: argparse.Namespace) -> None:
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        sys.exit(1)

    cases_raw = json.loads(dataset_path.read_text())
    config = {
        "top_k": args.top_k,
        "threshold": args.threshold,
        "dataset": str(dataset_path),
        "model": "gpt-4.1-mini",
        "embedding_model": "text-embedding-3-small",
    }

    print(f"Running eval: {len(cases_raw)} queries, top_k={args.top_k}")
    if args.threshold:
        print(f"  threshold override: {args.threshold}")

    results: list[EvalCase] = []
    async with httpx.AsyncClient(timeout=60.0) as client:
        for i, case in enumerate(cases_raw, 1):
            print(f"  [{i}/{len(cases_raw)}] {case['id']}: {case['query'][:50]}...")
            try:
                result = await run_single(client, case, args.top_k, args.threshold)
                results.append(result)
            except Exception as exc:
                print(f"    ERROR: {exc}")

    summary = summarize(results, config)
    print_summary(summary)

    if args.save_results:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = RESULTS_DIR / f"eval_{summary.run_id}.json"
        out_path.write_text(json.dumps(asdict(summary), indent=2))
        print(f"  Results saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Evaluation Runner")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--save-results", action="store_true")
    asyncio.run(main(parser.parse_args()))