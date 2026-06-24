"""Dense retriever over the Qdrant index. Returns chunk IDs ranked by relevance."""

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from rag.build_index import COLLECTION, MODEL_NAME, STORAGE_PATH


class DenseRetriever:
    def __init__(self) -> None:
        self._model = SentenceTransformer(MODEL_NAME)
        self._client = QdrantClient(path=STORAGE_PATH)

    def retrieve(self, query: str, k: int) -> list[str]:
        vector = self._model.encode(query, normalize_embeddings=True).tolist()
        hits = self._client.query_points(
            collection_name=COLLECTION, query=vector, limit=k
        ).points
        return [hit.payload["chunk_id"] for hit in hits]
