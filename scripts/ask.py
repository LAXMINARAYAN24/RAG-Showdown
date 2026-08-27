# scripts/ask.py
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
from datetime import datetime

from strategies.naive_rag import NaiveRAG
from strategies.advanced_rag import AdvancedRAG
from strategies.corrective_rag import CorrectiveRAG
from strategies.conflict_aware_rag import ConflictAwareRAG

# --- Command-line argument: which strategy to use ---
parser = argparse.ArgumentParser()
parser.add_argument(
    "--strategy",
    choices=["naive", "advanced", "corrective", "conflict_aware"],
    default="naive",
    help="Which strategy to answer with: naive, advanced, corrective, or conflict_aware"
)
args = parser.parse_args()

from core.vector_store import ChromaVectorStore
store = ChromaVectorStore()

# --- Pick the strategy based on the flag ---
all_strategies = {
    "naive": NaiveRAG(store),
    "advanced": AdvancedRAG(store),
    "corrective": CorrectiveRAG(store),
    "conflict_aware": ConflictAwareRAG(store),
}
rag = all_strategies[args.strategy]

question = input("Ask a question: ")
result = rag.answer_question(question)

print("\n--- Answer ---")
print(result["answer"])
print(f"\n(took {result['latency_seconds']:.2f}s, strategy: {result['strategy_name']})")

# --- Log this run for your report ---
os.makedirs("logs", exist_ok=True)
log_entry = {
    "timestamp": datetime.now().isoformat(),
    "question": question,
    "strategy": result["strategy_name"],
    "answer": result["answer"],
    "latency_seconds": result["latency_seconds"],
    "chunks_used": result["chunks_used"]
}

with open("logs/qa_log.jsonl", "a", encoding="utf-8") as f:
    f.write(json.dumps(log_entry) + "\n")

print("\n(logged to logs/qa_log.jsonl)")