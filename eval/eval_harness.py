"""
eval_harness.py — Retrieval evaluation for a RAG pipeline (recall@k, hit-rate@k, MRR).

Design decisions (these are the things worth being able to defend in an interview):

1. The harness is DECOUPLED from the vector store. Anything implementing the
   `Retriever` protocol — dense, BM25, or hybrid — can be scored on the SAME
   labeled set, so you can report before/after deltas when you change strategy.
2. Metrics are implemented FROM SCRATCH (no eval library). They are pure
   functions, unit-checkable in isolation.
3. Gold labels are stable CHUNK IDs, not raw text — re-chunking with the same
   IDs doesn't invalidate your hand labels.
4. recall@k and hit-rate@k are kept SEPARATE on purpose (most RAG blogs conflate
   them; with multiple gold chunks per query they are not the same metric).

Eval set format (JSONL, one object per line, hand-labeled):
    {"query": "What were the 2023 Scope 1 emissions?", "relevant_chunk_ids": ["c12", "c45"]}
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class EvalExample:
    """One labeled query.

    relevant_chunk_ids: every chunk a *correct* retriever should surface for this
    query — minimal but complete. Include all chunks that genuinely answer it,
    nothing that doesn't. Quality of these labels caps the quality of your numbers.
    """
    query: str
    relevant_chunk_ids: frozenset[str]

    @staticmethod
    def from_dict(d: dict) -> "EvalExample":
        return EvalExample(
            query=d["query"],
            relevant_chunk_ids=frozenset(d["relevant_chunk_ids"]),
        )


def load_eval_set(path: str | Path) -> list[EvalExample]:
    """Load a JSONL eval set. Fails loudly on a malformed line (with line number)."""
    examples: list[EvalExample] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                examples.append(EvalExample.from_dict(json.loads(line)))
            except (json.JSONDecodeError, KeyError) as e:
                raise ValueError(f"Bad eval example on line {line_no}: {e}") from e
    if not examples:
        raise ValueError(f"No eval examples found in {path}")
    return examples


# --------------------------------------------------------------------------- #
# Retriever interface — implement this around Qdrant (or BM25, or hybrid).
# See QdrantDenseRetriever at the bottom for a reference implementation.
# --------------------------------------------------------------------------- #

class Retriever(Protocol):
    def retrieve(self, query: str, k: int) -> list[str]:
        """Return up to k chunk IDs, ranked best-first."""
        ...


# --------------------------------------------------------------------------- #
# Metrics (per query). Pure functions.
# --------------------------------------------------------------------------- #

def recall_at_k(retrieved: Sequence[str], relevant: frozenset[str], k: int) -> float:
    """True recall@k:  |top_k ∩ relevant| / |relevant|.

    Measures COVERAGE — of all the chunks that should have been found, how many were.
    Not the same as hit-rate when a query has more than one gold chunk.
    """
    if not relevant:
        return 0.0  # undefined for empty gold; treat as 0 (or filter these out upstream)
    top_k = set(retrieved[:k])
    return len(top_k & relevant) / len(relevant)


def hit_rate_at_k(retrieved: Sequence[str], relevant: frozenset[str], k: int) -> float:
    """Binary hit-rate@k: 1.0 if ANY relevant chunk is in the top-k, else 0.0.

    This is what many RAG tutorials loosely call 'recall@k'. Reported separately
    so the distinction is explicit.
    """
    if not relevant:
        return 0.0
    return 1.0 if set(retrieved[:k]) & relevant else 0.0


def reciprocal_rank(retrieved: Sequence[str], relevant: frozenset[str]) -> float:
    """Reciprocal of the 1-indexed rank of the FIRST relevant chunk; 0.0 if none.

    Measures RANKING QUALITY — how near the top the first good chunk lands.
    Averaged across queries this is MRR.
    """
    for rank, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in relevant:
            return 1.0 / rank
    return 0.0


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #

@dataclass
class EvalReport:
    n_queries: int
    k_values: tuple[int, ...]
    recall_at_k: dict[int, float]
    hit_rate_at_k: dict[int, float]
    mrr: float

    def pretty(self) -> str:
        lines = [
            f"Queries evaluated : {self.n_queries}",
            f"MRR               : {self.mrr:.3f}",
            "",
            f"{'k':>4} | {'recall@k':>9} | {'hit@k':>6}",
            "-" * 26,
        ]
        for k in self.k_values:
            lines.append(
                f"{k:>4} | {self.recall_at_k[k]:>9.3f} | {self.hit_rate_at_k[k]:>6.3f}"
            )
        return "\n".join(lines)


def evaluate(
    retriever: Retriever,
    eval_set: Sequence[EvalExample],
    k_values: tuple[int, ...] = (1, 3, 5, 10),
) -> EvalReport:
    """Run the retriever over the labeled set and aggregate.

    Retrieves once at max(k_values) per query, then slices — so MRR can find the
    first relevant chunk anywhere up to the deepest k, and every k shares one call.
    Using multiple k values lets you separate a RETRIEVAL problem (low recall@10)
    from a RANKING problem (good recall@10 but low recall@1 / low MRR).
    """
    max_k = max(k_values)
    recall_sums = {k: 0.0 for k in k_values}
    hit_sums = {k: 0.0 for k in k_values}
    rr_sum = 0.0

    for ex in eval_set:
        retrieved = retriever.retrieve(ex.query, max_k)
        for k in k_values:
            recall_sums[k] += recall_at_k(retrieved, ex.relevant_chunk_ids, k)
            hit_sums[k] += hit_rate_at_k(retrieved, ex.relevant_chunk_ids, k)
        rr_sum += reciprocal_rank(retrieved, ex.relevant_chunk_ids)

    n = len(eval_set)
    return EvalReport(
        n_queries=n,
        k_values=k_values,
        recall_at_k={k: recall_sums[k] / n for k in k_values},
        hit_rate_at_k={k: hit_sums[k] / n for k in k_values},
        mrr=rr_sum / n,
    )


# --------------------------------------------------------------------------- #
# Reference Retriever implementation against Qdrant + sentence-transformers.
# Imports are LOCAL so this module runs without those packages installed
# (e.g. for the self-check below). Adapt collection name / payload key to your
# ingestion code, then: report = evaluate(QdrantDenseRetriever("esg_chunks"), eval_set)
# --------------------------------------------------------------------------- #

class QdrantDenseRetriever:
    def __init__(
        self,
        collection: str,
        model_name: str = "BAAI/bge-small-en-v1.5",
        host: str = "localhost",
        port: int = 6333,
        id_payload_key: str = "chunk_id",
    ) -> None:
        from qdrant_client import QdrantClient
        from sentence_transformers import SentenceTransformer

        self._client = QdrantClient(host=host, port=port)
        self._model = SentenceTransformer(model_name)
        self._collection = collection
        self._id_key = id_payload_key

    def retrieve(self, query: str, k: int) -> list[str]:
        vector = self._model.encode(query, normalize_embeddings=True).tolist()
        hits = self._client.query_points(
            collection_name=self._collection,
            query=vector,
            limit=k,
        ).points
        return [h.payload[self._id_key] for h in hits]


# --------------------------------------------------------------------------- #
# Self-check: a stub retriever with KNOWN output so you can verify the harness
# before wiring in Qdrant.  Run:  python eval_harness.py
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    class _StubRetriever:
        """Returns a fixed ranked list regardless of query — for testing metrics."""
        def __init__(self, ranked: list[str]) -> None:
            self._ranked = ranked

        def retrieve(self, query: str, k: int) -> list[str]:
            return self._ranked[:k]

    # gold = {c2, c5};  ranked = [c9, c2, c7, c5, c1]
    #   first relevant (c2) at rank 2          -> RR = 1/2 = 0.5
    #   recall@1 = 0/2=0.0  recall@3 = 1/2=0.5  recall@5 = 2/2=1.0
    #   hit@1   = 0.0       hit@3    = 1.0       hit@5    = 1.0
    stub = _StubRetriever(["c9", "c2", "c7", "c5", "c1"])
    example = EvalExample(query="(stub)", relevant_chunk_ids=frozenset({"c2", "c5"}))
    report = evaluate(stub, [example], k_values=(1, 3, 5))

    assert abs(report.mrr - 0.5) < 1e-9, report.mrr
    assert report.recall_at_k[1] == 0.0
    assert report.recall_at_k[3] == 0.5
    assert report.recall_at_k[5] == 1.0
    assert report.hit_rate_at_k[1] == 0.0
    assert report.hit_rate_at_k[3] == 1.0
    print("Self-check passed.\n")
    print(report.pretty())
