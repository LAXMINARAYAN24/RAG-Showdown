import time
from core.vector_store import SimpleVectorStore
from core.embedder import embed_texts
from core.reranker import rerank
from core.llm_client import ask_llm

class AdvancedRAG:
    name = "Advanced RAG (Reranked)"

    def __init__(self, store: SimpleVectorStore):
        self.store = store

    def answer_question(self, question):
        start = time.time()

        query_vec = embed_texts([question])[0]
        wide_results = self.store.search(query_vec, top_k=15)   # cast a wider net first

        reranked = rerank(question, wide_results, top_n=5)       # then narrow it down properly

        context = "\n\n".join([f"[Source: {i+1}]\n{text}" for i, (text, score) in enumerate(reranked)])
        prompt = f"""Answer the question using ONLY the context below. If the answer isn't in the context, say "I don't know."

Context:
{context}

Question: {question}

Answer:"""

        answer = ask_llm(prompt)
        latency = time.time() - start

        return {
            "answer": answer,
            "chunks_used": [text for text, score in reranked],
            "strategy_name": self.name,
            "latency_seconds": latency
        }