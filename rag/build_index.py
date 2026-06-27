"""Embed chunks, index them in Qdrant, and run a similarity search."""

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from rag.ingest import load_chunks

COLLECTION = "documents"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
VECTOR_SIZE = 384
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333


def build_index(client: QdrantClient, model: SentenceTransformer) -> int:
    chunks = load_chunks()
    if not chunks:
        return 0

    vectors = model.encode(
        [c["text"] for c in chunks],
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )

    # Qdrant point IDs must be ints or UUIDs, so the chunk_id is carried in the payload.
    client.upsert(
        collection_name=COLLECTION,
        points=[
            PointStruct(id=i, vector=vectors[i].tolist(), payload=chunks[i])
            for i in range(len(chunks))
        ],
    )
    return len(chunks)


def search(client: QdrantClient, model: SentenceTransformer, query: str, k: int = 3):
    vector = model.encode(query, normalize_embeddings=True).tolist()
    return client.query_points(collection_name=COLLECTION, query=vector, limit=k).points


def main() -> None:
    model = SentenceTransformer(MODEL_NAME)
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    indexed = build_index(client, model)
    if not indexed:
        print("No chunks to index. Add PDFs to data/ and check rag/ingest.py.")
        return
    print(f"Indexed {indexed} chunks.\n")

    query = "What is language identification from audio?"
    print(f"Query: {query}\n")
    for rank, hit in enumerate(search(client, model, query), start=1):
        text = hit.payload["text"].replace("\n", " ")
        print(f"#{rank}  {hit.score:.3f}  {hit.payload['source']}")
        print(f"    {text[:220]} ...\n")


if __name__ == "__main__":
    main()
