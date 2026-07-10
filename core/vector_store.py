import numpy as np

class SimpleVectorStore:
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
    