# eval/run_eval.py
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import yaml
import json
import csv
import time
import argparse
from datetime import datetime

from strategies.naive_rag import NaiveRAG
from strategies.advanced_rag import AdvancedRAG
from strategies.corrective_rag import CorrectiveRAG
from strategies.conflict_aware_rag import ConflictAwareRAG
from eval.judge import judge_answer

# --- Command-line argument: which strategy to run ---
parser = argparse.ArgumentParser()
parser.add_argument(
    "--strategy",
    choices=["naive", "advanced", "corrective", "conflict_aware", "all"],
    default="all",
    help="Which strategy to run: naive, advanced, or all"
)
args = parser.parse_args()

# Load the vector store
from core.vector_store import ChromaVectorStore
store = ChromaVectorStore()

# Load the question bank
with open("questions/question_bank.yaml", "r", encoding="utf-8") as f:
    questions = yaml.safe_load(f)

# --- Build the list of strategies to actually run, based on the flag ---
all_strategies = {
    "naive": NaiveRAG(store),
    "advanced": AdvancedRAG(store),
    "corrective": CorrectiveRAG(store),
    "conflict_aware": ConflictAwareRAG(store),
}

if args.strategy == "all":
    strategies_to_test = list(all_strategies.values())
else:
    strategies_to_test = [all_strategies[args.strategy]]

results = []

for strategy in strategies_to_test:
    print(f"\n=== Testing strategy: {strategy.name} ===")

    for q in questions:
        print(f"  Q: {q['question'][:60]}...")

      
        rag_result = strategy.answer_question(q["question"])
        time.sleep(0.5)   # small safety buffer, not the multi-second Gemini pacing

        judged = judge_answer(q["question"], q["expected_answer_contains"], rag_result["answer"])
        time.sleep(0.5)
        results.append({
            "question_id": q["id"],
            "category": q["category"],
            "strategy": strategy.name,
            "question": q["question"],
            "answer": rag_result["answer"],
            "score": judged["score"],
            "judge_reason": judged["reason"],
            "latency_seconds": rag_result["latency_seconds"]
        })

        print(f"    -> Score: {judged['score']}/5 — {judged['reason']}")

# Save results
os.makedirs("eval/results", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_path = f"eval/results/eval_{args.strategy}_{timestamp}.csv"

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print(f"\nDone. {len(results)} results saved to {csv_path}")

avg_by_strategy = {}
for r in results:
    avg_by_strategy.setdefault(r["strategy"], []).append(r["score"])

print("\n=== Summary ===")
for strategy_name, scores in avg_by_strategy.items():
    avg = sum(scores) / len(scores)
    print(f"{strategy_name}: average score {avg:.2f}/5 across {len(scores)} questions")