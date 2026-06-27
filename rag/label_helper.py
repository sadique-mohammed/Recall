"""Print top candidate chunks per question to help build the eval set by hand."""

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from rag.build_index import COLLECTION, MODEL_NAME, QDRANT_HOST, QDRANT_PORT

QUESTIONS = [
    "What is the BenCoref dataset?",
    "What is the main contribution of the end-to-end neural coreference resolution model?",
    "How does SpanBERT change BERT's pre-training objective?",
    "What is MuRIL and what languages was it pre-trained on?",
    "What is the mGAP dataset?",
    "What two models does the Bridge the GAP paper propose for coreference resolution?",
    "What dataset is Bangla-BERT pre-trained on and how large is it?",
    "What does the paper 'What Would ELSA Do?' investigate about transformer fine-tuning?",
    "What does the BanglaBERT paper propose for low-resource Bangla language understanding?",
    "How is the multilingual coreference dataset for South Asian languages created?",
]

TOP_K = 8


def main() -> None:
    model = SentenceTransformer(MODEL_NAME)
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    for question in QUESTIONS:
        print("=" * 72)
        print(f"Q: {question}\n")
        vector = model.encode(question, normalize_embeddings=True).tolist()
        hits = client.query_points(
            collection_name=COLLECTION, query=vector, limit=TOP_K
        ).points
        for hit in hits:
            text = hit.payload["text"].replace("\n", " ")
            print(f"  {hit.payload['chunk_id']}  {hit.payload['source']}")
            print(f"     {text[:110]} ...\n")


if __name__ == "__main__":
    main()
