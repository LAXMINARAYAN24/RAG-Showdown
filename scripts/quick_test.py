# scripts/quick_test.py
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
from strategies.advanced_rag import AdvancedRAG

with open("corpus/processed/store.pkl", "rb") as f:
    store = pickle.load(f)

rag = AdvancedRAG(store)

test_questions = [
    "Name three vector stores that Confluent has native integrations with.",
    "Is RAG always better than fine-tuning an LLM?",
]

for q in test_questions:
    result = rag.answer_question(q)
    print(f"\nQ: {q}")
    print(f"A: {result['answer']}")