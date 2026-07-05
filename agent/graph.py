"""Agentic RAG flow with LangGraph: retrieve -> generate -> verify, with one retry.

The verify node asks the model whether the generated answer is actually supported
by the retrieved context. If not, the graph retries retrieval once with a larger k
before finishing — a minimal, inspectable self-correction loop.

Run a demo:  python -m agent.graph
Requires the Qdrant container (docker compose up -d) and AWS credentials.
"""

from typing import TypedDict

from langgraph.graph import END, StateGraph

from rag.generate import (
    BEDROCK_MODEL_ID,
    _bedrock,
    _format_context,
    generate_from_chunks,
    retrieve_chunks,
)

MAX_ATTEMPTS = 2  # initial pass + one retry
RETRY_K_MULTIPLIER = 2  # retry widens retrieval (k=3 -> k=6)

VERIFY_PROMPT = (
    "You are checking a question-answering system. Given the context and the answer, "
    "reply with exactly one word: GROUNDED if the answer is supported by the context, "
    "or NOT_GROUNDED if it makes claims the context does not support or fails to answer."
)


class GraphState(TypedDict):
    question: str
    k: int
    chunks: list
    answer: str
    sources: list[str]
    verdict: str
    attempts: int


def retrieve(state: GraphState) -> dict:
    hits = retrieve_chunks(state["question"], state["k"])
    return {
        "chunks": hits,
        "sources": [h.payload["chunk_id"] for h in hits],
        "attempts": state["attempts"] + 1,
    }


def generate(state: GraphState) -> dict:
    answer = generate_from_chunks(state["question"], state["chunks"])
    return {"answer": answer}


def verify(state: GraphState) -> dict:
    response = _bedrock().converse(
        modelId=BEDROCK_MODEL_ID,
        system=[{"text": VERIFY_PROMPT}],
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "text": (
                            f"Context:\n{_format_context(state['chunks'])}\n\n"
                            f"Question: {state['question']}\n\n"
                            f"Answer: {state['answer']}"
                        )
                    }
                ],
            }
        ],
        inferenceConfig={"maxTokens": 5, "temperature": 0.0},
    )
    text = response["output"]["message"]["content"][0]["text"].strip().upper()
    verdict = "GROUNDED" if "NOT" not in text else "NOT_GROUNDED"
    return {"verdict": verdict}


def route_after_verify(state: GraphState) -> str:
    if state["verdict"] == "NOT_GROUNDED" and state["attempts"] < MAX_ATTEMPTS:
        return "retry"
    return "done"


def widen_retrieval(state: GraphState) -> dict:
    # Retry with a wider net: more chunks give the generator more to work with.
    return {"k": state["k"] * RETRY_K_MULTIPLIER}


def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_node("verify", verify)
    graph.add_node("widen_retrieval", widen_retrieval)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "verify")
    graph.add_conditional_edges(
        "verify", route_after_verify, {"retry": "widen_retrieval", "done": END}
    )
    graph.add_edge("widen_retrieval", "retrieve")
    return graph.compile()


def run(question: str, k: int = 3) -> GraphState:
    app = build_graph()
    return app.invoke(
        {
            "question": question,
            "k": k,
            "chunks": [],
            "answer": "",
            "sources": [],
            "verdict": "",
            "attempts": 0,
        }
    )


def main() -> None:
    question = "What is the BenCoref dataset?"
    result = run(question)
    print(f"Q: {question}\n")
    print(result["answer"])
    print(f"\nVerdict: {result['verdict']}  |  attempts: {result['attempts']}  |  k used: {result['k']}")
    print(f"Sources: {result['sources']}")


if __name__ == "__main__":
    main()
