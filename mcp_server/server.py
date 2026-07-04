"""MCP server exposing the RAG pipeline as tools for AI agents.

Wraps the existing shared components (retriever, generator, eval harness) so any
MCP-capable client can search the corpus, ask grounded questions, and check
retrieval quality — without bespoke integration code.

Run for local inspection:  mcp dev mcp_server/server.py
Requires the Qdrant container to be running (docker compose up -d).
"""

import sys
from pathlib import Path

# mcp dev loads this file directly, without the project root on sys.path —
# add it so the rag/ and eval/ packages resolve.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp.server.fastmcp import FastMCP

from eval.eval_harness import evaluate, load_eval_set
from rag.generate import answer_question
from rag.retriever import DenseRetriever

mcp = FastMCP("rag-eval-pipeline")

# One retriever instance shared across tool calls (model loads once).
_retriever = DenseRetriever()


@mcp.tool()
def search_documents(query: str, k: int = 5) -> list[dict]:
    """Search the research-paper corpus and return the top-k chunks.

    Args:
        query: Natural-language search query.
        k: Number of chunks to return (default 5).

    Returns:
        Ranked list of {chunk_id, source, text} dicts.
    """
    hits = _retriever._client.query_points(
        collection_name="documents",
        query=_retriever._model.encode(query, normalize_embeddings=True).tolist(),
        limit=k,
    ).points
    return [
        {
            "chunk_id": h.payload["chunk_id"],
            "source": h.payload["source"],
            "text": h.payload["text"],
        }
        for h in hits
    ]


@mcp.tool()
def ask_question(question: str, k: int = 3) -> dict:
    """Answer a question grounded in the paper corpus, with source chunk IDs.

    Args:
        question: The question to answer.
        k: Number of chunks to retrieve as context (default 3).

    Returns:
        {answer, sources} where sources are the chunk IDs the answer drew on.
    """
    answer, sources = answer_question(question, k)
    return {"answer": answer, "sources": sources}


@mcp.tool()
def evaluate_retrieval() -> dict:
    """Score the retriever against the hand-labeled eval set.

    Returns:
        {mrr, per_k} with recall@k and hit_rate@k for k in 1, 3, 5, 10.
    """
    eval_set = load_eval_set("eval/eval_set.jsonl")
    report = evaluate(_retriever, eval_set, k_values=(1, 3, 5, 10))
    return {
        "queries_evaluated": len(eval_set),
        "mrr": round(report.mrr, 3),
        "per_k": {
            str(k): {
                "recall": round(report.recall_at_k[k], 3),
                "hit_rate": round(report.hit_rate_at_k[k], 3),
            }
            for k in report.recall_at_k
        },
    }


if __name__ == "__main__":
    mcp.run()
