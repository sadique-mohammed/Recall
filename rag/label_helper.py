"""Labelling helper: for each question, show candidate chunks (with their IDs) so you
can decide which ones actually answer it when building eval/eval_set.jsonl."""

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from rag.build_index import COLLECTION, MODEL_NAME, STORAGE_PATH

QUESTIONS = [
    "What is language identification from audio?",
    "What is the baseline model and its accuracy for language identification?",
    "Why does having only 5 speakers per language cause a problem?",
    "What constraints apply to the model in the language identification project?",
    "What is Bangla-BERT?",
    "How large is the BanglaLM dataset used to train Bangla-BERT?",
    "What two limitations of mBERT does Bangla-BERT address?",
    "What is the mGAP dataset?",
    "What two models are proposed for coreference resolution?",
    "Which coreference model performed better, JEM or CAM?",
]

CANDIDATES = 8


def main() -> None:
    model = SentenceTransformer(MODEL_NAME)
    client = QdrantClient(path=STORAGE_PATH)
    for question in QUESTIONS:
        print("=" * 72)
        print(f"Q: {question}\n")
        vector = model.encode(question, normalize_embeddings=True).tolist()
        hits = client.query_points(
            collection_name=COLLECTION, query=vector, limit=CANDIDATES
        ).points
        for hit in hits:
            snippet = hit.payload["text"].replace("\n", " ")[:160]
            print(f"  {hit.payload['chunk_id']}  {hit.payload['source']}")
            print(f"     {snippet} ...\n")


if __name__ == "__main__":
    main()