# scripts/chunk_retrieval.py
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.embedder import embed_texts
from core.vector_store import ChromaVectorStore

store = ChromaVectorStore()

# questions_to_check = [
#     "Name three vector stores that Confluent has native integrations with.",
#     "What is the difference between how RAG retrieves information and how a traditional database query works?",
#     "Is RAG always better than fine-tuning an LLM?",
# ]
# scripts/check_retrieval.py — temporarily replace the question list with just this one
questions_to_check = [
    "What is the difference between how RAG retrieves information and how a traditional database query works?",
]

for question in questions_to_check:
    print("=" * 80)
    print(f"QUESTION: {question}")
    print("=" * 80)

    query_vec = embed_texts([question])[0]
    results = store.search(query_vec, top_k=10)   # <-- changed from 3 to 8

    for i, (chunk_text, score) in enumerate(results):
        print(f"\n--- Chunk {i+1} (score: {score:.3f}) ---")
        print(chunk_text[:300])

    print("\n")