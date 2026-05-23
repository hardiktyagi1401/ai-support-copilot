# eval/metrics/retrieval.py

from dataclasses import dataclass
from typing import Any


@dataclass
class RetrievalMetrics:
    recall_at_k: float         # fraction of relevant chunks in top-k
    mrr: float                 # mean reciprocal rank — how high is the first hit?
    precision_at_k: float      # fraction of retrieved chunks that are relevant
    chunks_retrieved: int
    chunks_relevant: int
    threshold_pass_rate: float # fraction of retrieved that passed threshold
    answered: bool


def compute_retrieval_metrics(
    retrieved_chunks: list[dict],   # from your AskResponse.retrieval_stats
    relevant_filenames: list[str],  # from golden dataset
    top_k: int,
) -> RetrievalMetrics:
    """
    Compute retrieval quality metrics for one query.

    We match by filename rather than exact chunk index because chunk
    indices can shift when documents are re-ingested with different
    chunking parameters. Filename matching is more stable.

    For stricter evaluation once your system is stable, match by
    chunk_index too — but start with filename, which is forgiving enough
    to be useful immediately.
    """
    if not relevant_filenames:
        # Unanswerable query — retrieval metrics don't apply
        return RetrievalMetrics(
            recall_at_k=1.0,  # vacuously true: no relevant chunks to miss
            mrr=0.0,
            precision_at_k=0.0,
            chunks_retrieved=len(retrieved_chunks),
            chunks_relevant=0,
            threshold_pass_rate=0.0,
            answered=False,
        )

    retrieved_filenames = [c.get("filename", "") for c in retrieved_chunks]
    relevant_set = set(relevant_filenames)

    # Recall@K: what fraction of relevant files appear in top-K results?
    hit_filenames = set(retrieved_filenames) & relevant_set
    recall = len(hit_filenames) / len(relevant_set) if relevant_set else 0.0

    # Precision@K: what fraction of retrieved results are actually relevant?
    relevant_hits = sum(1 for f in retrieved_filenames if f in relevant_set)
    precision = relevant_hits / len(retrieved_chunks) if retrieved_chunks else 0.0

    # MRR: reciprocal rank of the first relevant result
    rr = 0.0
    for rank, filename in enumerate(retrieved_filenames, start=1):
        if filename in relevant_set:
            rr = 1.0 / rank
            break

    passed = sum(1 for c in retrieved_chunks if c.get("passed_threshold", False))
    pass_rate = passed / len(retrieved_chunks) if retrieved_chunks else 0.0

    return RetrievalMetrics(
        recall_at_k=round(recall, 3),
        mrr=round(rr, 3),
        precision_at_k=round(precision, 3),
        chunks_retrieved=len(retrieved_chunks),
        chunks_relevant=len(relevant_set),
        threshold_pass_rate=round(pass_rate, 3),
        answered=True,
    )