# core/vector_store.py
import numpy as np
import uuid
import chromadb

CHROMA_PATH = "corpus/processed/chroma"
COLLECTION_NAME = "rag_showdown"


class ChromaVectorStore:
    """Persistent vector store backed by Chroma (embedded, no server).

    Keeps the same interface as the old SimpleVectorStore —
    search(query_vector, top_k) -> [(chunk_text, score)] — so the
    strategies don't change. Scores are cosine similarity (1 - distance),
    matching the old store's scale.
    """

    def __init__(self, path=CHROMA_PATH, collection=COLLECTION_NAME):
        self._client = chromadb.PersistentClient(path=path)
        self._collection = self._client.get_or_create_collection(
            collection, metadata={"hnsw:space": "cosine"}
        )

    def reset(self):
        """Drop and recreate the collection (used by ingest for a clean rebuild)."""
        self._client.delete_collection(self._collection.name)
        self._collection = self._client.get_or_create_collection(
            COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )

    def add(self, chunks, vectors, source=None):
        """Add chunks with precomputed embeddings. `source` tags each chunk
        with the file it came from (enables metadata filtering later)."""
        self._collection.add(
            ids=[str(uuid.uuid4()) for _ in range(len(chunks))],
            documents=list(chunks),
            embeddings=np.asarray(vectors).tolist(),
            metadatas=[{"source": source or "unknown"}] * len(chunks),
        )

    def search(self, query_vector, top_k=3):
        top_k = min(top_k, self._collection.count())
        res = self._collection.query(
            query_embeddings=[np.asarray(query_vector).tolist()],
            n_results=top_k,
        )
        docs = res["documents"][0]
        dists = res["distances"][0]
        return [(doc, 1.0 - dist) for doc, dist in zip(docs, dists)]

    @property
    def chunks(self):
        """All stored chunk texts (used by diagnostic scripts)."""
        return self._collection.get()["documents"]

    def count(self):
        return self._collection.count()


class SimpleVectorStore:
    """Legacy in-memory store — kept only so the old corpus/processed/store.pkl
    remains unpicklable-compatible. New code should use ChromaVectorStore."""

    def __init__(self):
        self.chunks = []
        self.vectors = None

    def add(self, chunks, vectors):
        self.chunks.extend(chunks)
        self.vectors = vectors if self.vectors is None else np.vstack([self.vectors, vectors])

    def search(self, query_vector, top_k=3):
        sims = np.dot(self.vectors, query_vector) / (
            np.linalg.norm(self.vectors, axis=1) * np.linalg.norm(query_vector)
        )
        top_indices = np.argsort(sims)[::-1][:top_k]
        return [(self.chunks[i], float(sims[i])) for i in top_indices]
