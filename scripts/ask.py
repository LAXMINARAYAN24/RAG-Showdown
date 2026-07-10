# scripts/ask.py
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import json
from datetime import datetime
from strategies.naive_rag import NaiveRAG

with open("corpus/processed/store.pkl", "rb") as f:
    store = pickle.load(f)

rag = NaiveRAG(store)
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