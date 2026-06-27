"""Answer questions over the indexed documents using retrieved context and Amazon Bedrock."""

from functools import lru_cache

import boto3
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from rag.build_index import MODEL_NAME, QDRANT_HOST, QDRANT_PORT, search

REGION = "eu-central-1"
# Frankfurt invokes models through a regional inference profile (the "eu." prefix).
BEDROCK_MODEL_ID = "eu.amazon.nova-lite-v1:0"

SYSTEM_PROMPT = (
    "Answer the question using only the provided context. "
    "If the context does not contain the answer, say you don't know. "
    "Keep the answer concise."
)


# Loaded once and reused across requests.
@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)


@lru_cache(maxsize=1)
def _qdrant() -> QdrantClient:
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


@lru_cache(maxsize=1)
def _bedrock():
    return boto3.client("bedrock-runtime", region_name=REGION)


def _format_context(hits) -> str:
    return "\n\n".join(f"[{h.payload['source']}]\n{h.payload['text']}" for h in hits)


def answer_question(question: str, k: int = 3) -> tuple[str, list[str]]:
    hits = search(_qdrant(), _model(), question, k)
    response = _bedrock().converse(
        modelId=BEDROCK_MODEL_ID,
        system=[{"text": SYSTEM_PROMPT}],
        messages=[
            {
                "role": "user",
                "content": [
                    {"text": f"Context:\n{_format_context(hits)}\n\nQuestion: {question}"}
                ],
            }
        ],
        inferenceConfig={"maxTokens": 512, "temperature": 0.2},
    )
    answer = response["output"]["message"]["content"][0]["text"]
    return answer, [hit.payload["chunk_id"] for hit in hits]


def main() -> None:
    question = "What is language identification from audio?"
    answer, sources = answer_question(question)
    print(f"Q: {question}\n")
    print(answer)
    print(f"\nRetrieved chunks: {sources}")


if __name__ == "__main__":
    main()
