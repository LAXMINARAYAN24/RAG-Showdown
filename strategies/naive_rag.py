# strategies/naive_rag.py
import time
from core.vector_store import SimpleVectorStore
from core.embedder import embed_texts
from core.llm_client import ask_llm          # <-- changed from ask_claude

class NaiveRAG:
    name = "Naive RAG"

    def __init__(self, store: SimpleVectorStore):
        self.store = store

    def answer_question(self, question):
        start = time.time()
        query_vec = embed_texts([question])[0]
        # results = self.store.search(query_vec, top_k=3)
        results = self.store.search(query_vec, top_k=5)

        context = "\n\n".join([f"[Source: {i+1}]\n{text}" for i, (text, score) in enumerate(results)])
        prompt = f"""Answer the question using ONLY the context below. If the answer isn't in the context, say "I don't know."

Context:
{context}

Question: {question}

Answer:"""

        answer = ask_llm(prompt)              # <-- changed from ask_claude
        latency = time.time() - start

        return {
            "answer": answer,
            "chunks_used": [text for text, score in results],
            "strategy_name": self.name,
            "latency_seconds": latency
        }