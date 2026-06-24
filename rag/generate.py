"""Answer questions over the indexed documents using retrieved context and Amazon Bedrock."""

import boto3
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from rag.build_index import MODEL_NAME, STORAGE_PATH, search

REGION = "eu-central-1"
# Frankfurt invokes models through a regional inference profile (the "eu." prefix),
# not the plain foundation-model ID.
BEDROCK_MODEL_ID = "eu.amazon.nova-lite-v1:0"

SYSTEM_PROMPT = (
    "Answer the question using only the provided context. "
    "If the context does not contain the answer, say you don't know. "
    "Keep the answer concise."
)


def format_context(hits) -> str:
    return "\n\n".join(f"[{h.payload['source']}]\n{h.payload['text']}" for h in hits)


def answer_question(question: str, k: int = 3) -> tuple[str, list[str]]:
    model = SentenceTransformer(MODEL_NAME)
    qdrant = QdrantClient(path=STORAGE_PATH)
    hits = search(qdrant, model, question, k)

    bedrock = boto3.client("bedrock-runtime", region_name=REGION)
    response = bedrock.converse(
        modelId=BEDROCK_MODEL_ID,
        system=[{"text": SYSTEM_PROMPT}],
        messages=[
            {
                "role": "user",
                "content": [
                    {"text": f"Context:\n{format_context(hits)}\n\nQuestion: {question}"}
                ],
            }
        ],
        inferenceConfig={"maxTokens": 512, "temperature": 0.2},
    )
    answer = response["output"]["message"]["content"][0]["text"]
    return answer, [h.payload["chunk_id"] for h in hits]


def main() -> None:
    question = "What is language identification from audio?"
    answer, sources = answer_question(question)
    print(f"Q: {question}\n")
    print(answer)
    print(f"\nRetrieved chunks: {sources}")


if __name__ == "__main__":
    main()