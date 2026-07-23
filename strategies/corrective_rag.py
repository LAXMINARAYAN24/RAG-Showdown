# strategies/corrective_rag.py
import time
from core.vector_store import ChromaVectorStore
from core.embedder import embed_texts
from core.reranker import rerank
from core.llm_client import ask_llm


class CorrectiveRAG:
    """Corrective RAG: retrieve -> grade chunks -> answer, with one retry.

    Grading uses a single combined LLM call that scores all chunks at once
    (cheaper than per-chunk grading). If too few chunks pass, the query is
    reformulated by the LLM and retrieval runs once more with a wider net
    before falling back to "I don't know".
    """
    name = "Corrective RAG (Graded)"

    MIN_RELEVANT = 2   # need at least this many RELEVANT chunks to answer

    def __init__(self, store: ChromaVectorStore):
        self.store = store

    def _retrieve(self, query, top_k, top_n):
        query_vec = embed_texts([query])[0]
        wide_results = self.store.search(query_vec, top_k=top_k)
        return rerank(query, wide_results, top_n=top_n)

    def _grade_chunks(self, question, chunks):
        """One combined call: returns the subset of chunks graded RELEVANT."""
        numbered = "\n\n".join(
            f"[Chunk {i+1}]\n{text}" for i, (text, score) in enumerate(chunks)
        )
        prompt = f"""You are grading retrieved chunks for relevance to a question.

Question: {question}

{numbered}

For each chunk, decide if it contains information that helps answer the question.
Reply with ONLY one line per chunk, in the form:
1: RELEVANT
2: IRRELEVANT
(etc. for all {len(chunks)} chunks — no other text.)"""

        reply = ask_llm(prompt)

        relevant = []
        for line in reply.strip().splitlines():
            parts = line.split(":")
            if len(parts) < 2:
                continue
            try:
                idx = int(parts[0].strip()) - 1
            except ValueError:
                continue
            if 0 <= idx < len(chunks) and "IRRELEVANT" not in parts[1].upper():
                relevant.append(chunks[idx])
        return relevant

    def _reformulate(self, question):
        prompt = f"""Rewrite the following question as a short search query using different
wording (synonyms, related terms) so a semantic search can find relevant passages.
Reply with ONLY the rewritten query.

Question: {question}"""
        return ask_llm(prompt).strip()

    def answer_question(self, question):
        start = time.time()

        # Round 1: same wide-net + rerank as Advanced RAG
        chunks = self._retrieve(question, top_k=60, top_n=5)
        relevant = self._grade_chunks(question, chunks)
        rounds = 1

        # Round 2: not enough good chunks -> reformulate and search wider
        if len(relevant) < self.MIN_RELEVANT:
            new_query = self._reformulate(question)
            chunks2 = self._retrieve(new_query, top_k=120, top_n=8)
            relevant += [c for c in self._grade_chunks(question, chunks2)
                         if c[0] not in {text for text, _ in relevant}]
            rounds = 2

        if relevant:
            context = "\n\n".join(
                f"[Source: {i+1}]\n{text}" for i, (text, score) in enumerate(relevant)
            )
            prompt = f"""Answer the question using ONLY the context below. If the context is relevant but doesn't state the answer directly, reason it out from what the context does say. Say "I don't know" ONLY if the context contains nothing relevant to the question.

Context:
{context}

Question: {question}

Answer:"""
            answer = ask_llm(prompt)
        else:
            answer = "I don't know."

        latency = time.time() - start

        return {
            "answer": answer,
            "chunks_used": [text for text, score in relevant],
            "strategy_name": self.name,
            "latency_seconds": latency,
            "retrieval_rounds": rounds,
        }
