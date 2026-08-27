# strategies/conflict_aware_rag.py
"""
Conflict-Aware RAG: the same retrieve-rerank pipeline as Advanced RAG,
but with a pairwise NLI contradiction check *before* generation.

If contradictions are detected among the top chunks, the generation
prompt explicitly tells the LLM which sources conflict and asks it to
acknowledge the disagreement rather than silently picking one side.
"""
import time
from core.vector_store import ChromaVectorStore
from core.embedder import embed_texts
from core.reranker import rerank
from core.llm_client import ask_llm
from core.contradiction_detector import (
    detect_contradictions,
    format_conflicts_for_prompt,
)


class ConflictAwareRAG:
    name = "Conflict-Aware RAG"

    def __init__(self, store: ChromaVectorStore, contradiction_threshold: float = 0.5):
        self.store = store
        self.contradiction_threshold = contradiction_threshold

    def answer_question(self, question):
        start = time.time()

        # ── Step 1: Retrieve + rerank (same as Advanced RAG) ──────────
        query_vec = embed_texts([question])[0]
        wide_results = self.store.search(query_vec, top_k=60)
        reranked = rerank(question, wide_results, top_n=5)

        chunk_texts = [text for text, _ in reranked]

        # ── Step 2: Pairwise NLI contradiction check ──────────────────
        conflicts = detect_contradictions(
            chunk_texts, threshold=self.contradiction_threshold
        )

        # ── Step 3: Build the generation prompt ───────────────────────
        context = "\n\n".join(
            f"[Source: {i+1}]\n{text}"
            for i, (text, score) in enumerate(reranked)
        )

        if conflicts:
            conflict_block = format_conflicts_for_prompt(conflicts)
            prompt = f"""You are answering a question using retrieved source documents.
IMPORTANT: The system has detected CONTRADICTIONS between some of your source documents (listed below). You MUST:
1. Acknowledge the contradiction explicitly in your answer
2. Present what each conflicting source says
3. If possible, explain which claim is more likely correct and why
4. Do NOT silently pick one side — the user needs to know the sources disagree

{conflict_block}

Context (all retrieved sources):
{context}

Question: {question}

Answer (be sure to flag any contradictions):"""
        else:
            prompt = f"""Answer the question using ONLY the context below. If the context is relevant but doesn't state the answer directly, reason it out from what the context does say. Say "I don't know" ONLY if the context contains nothing relevant to the question.

Context:
{context}

Question: {question}

Answer:"""

        answer = ask_llm(prompt)
        latency = time.time() - start

        return {
            "answer": answer,
            "chunks_used": chunk_texts,
            "strategy_name": self.name,
            "latency_seconds": latency,
            "conflicts_detected": [
                {
                    "source_i": c.chunk_i_idx + 1,
                    "source_j": c.chunk_j_idx + 1,
                    "contradiction_score": round(c.contradiction_score, 3),
                    "snippet_i": c.chunk_i_text[:150],
                    "snippet_j": c.chunk_j_text[:150],
                }
                for c in conflicts
            ],
        }
