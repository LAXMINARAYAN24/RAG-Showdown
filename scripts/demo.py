# scripts/demo.py
"""
Interactive seminar demo — runs a question through all 4 strategies
side-by-side and highlights the difference contradiction detection makes.

Usage:
    python scripts/demo.py
    python scripts/demo.py --question "When was the Transformer introduced?"
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time

from core.vector_store import ChromaVectorStore
from strategies.naive_rag import NaiveRAG
from strategies.advanced_rag import AdvancedRAG
from strategies.corrective_rag import CorrectiveRAG
from strategies.conflict_aware_rag import ConflictAwareRAG


# ── ANSI color codes for terminal output ────────────────────────────
class C:
    HEADER  = "\033[95m"
    BLUE    = "\033[94m"
    CYAN    = "\033[96m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    RED     = "\033[91m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    RESET   = "\033[0m"


def print_banner():
    print(f"""
{C.BOLD}{C.CYAN}╔══════════════════════════════════════════════════════════════╗
║                   RAG  S H O W D O W N                       ║
║          Contradiction Detection Demo — Live                 ║
╚══════════════════════════════════════════════════════════════╝{C.RESET}
""")


def print_strategy_result(result, show_conflicts=False):
    """Pretty-print one strategy's result."""
    name = result["strategy_name"]
    answer = result["answer"]
    latency = result["latency_seconds"]

    # Truncate long answers for the demo view
    if len(answer) > 600:
        answer = answer[:600] + "..."

    print(f"  {C.DIM}Latency: {latency:.1f}s{C.RESET}")
    print(f"  {answer}")

    if show_conflicts and "conflicts_detected" in result:
        conflicts = result["conflicts_detected"]
        if conflicts:
            print(f"\n  {C.RED}{C.BOLD}⚠ {len(conflicts)} contradiction(s) detected:{C.RESET}")
            for c in conflicts:
                print(f"    {C.RED}• Source {c['source_i']} vs Source {c['source_j']} "
                      f"(confidence: {c['contradiction_score']:.0%}){C.RESET}")
                print(f"      {C.DIM}A: \"{c['snippet_i'][:100]}...\"{C.RESET}")
                print(f"      {C.DIM}B: \"{c['snippet_j'][:100]}...\"{C.RESET}")
        else:
            print(f"\n  {C.GREEN}✓ No contradictions detected among retrieved chunks.{C.RESET}")
    print()


def run_demo(question: str):
    store = ChromaVectorStore()
    print(f"{C.DIM}Loaded {store.count()} chunks from vector store.{C.RESET}\n")

    strategies = [
        ("1", "Naive RAG",          NaiveRAG(store),          False),
        ("2", "Advanced RAG",       AdvancedRAG(store),       False),
        ("3", "Corrective RAG",     CorrectiveRAG(store),     False),
        ("4", "Conflict-Aware RAG", ConflictAwareRAG(store),  True),
    ]

    print(f"{C.BOLD}Question:{C.RESET} {C.YELLOW}{question}{C.RESET}\n")
    print(f"{'─' * 64}\n")

    results = []
    for num, label, strategy, show_conflicts in strategies:
        print(f"{C.BOLD}{C.BLUE}┌─ Strategy {num}: {label}{C.RESET}")
        result = strategy.answer_question(question)
        print_strategy_result(result, show_conflicts=show_conflicts)
        results.append(result)
        print(f"{'─' * 64}\n")

    # ── Summary table ─────────────────────────────────────────────
    print(f"{C.BOLD}{C.CYAN}╔═══ SUMMARY ═══════════════════════════════════════════════╗{C.RESET}")
    print(f"  {'Strategy':<30} {'Latency':>8}  {'Conflicts?':>12}")
    print(f"  {'─' * 54}")
    for r in results:
        conflicts = r.get("conflicts_detected", [])
        conflict_str = (f"{C.RED}{len(conflicts)} found{C.RESET}" if conflicts
                        else f"{C.DIM}N/A{C.RESET}")
        if r["strategy_name"] in ("Naive RAG", "Advanced RAG (Reranked)", "Corrective RAG (Graded)"):
            conflict_str = f"{C.DIM}not checked{C.RESET}"
        print(f"  {r['strategy_name']:<30} {r['latency_seconds']:>7.1f}s  {conflict_str:>12}")
    print(f"{C.BOLD}{C.CYAN}╚═══════════════════════════════════════════════════════════╝{C.RESET}\n")

    print(f"{C.GREEN}{C.BOLD}Key takeaway:{C.RESET} Strategies 1–3 silently pick one answer.")
    print(f"Strategy 4 detects the conflict and tells the user the sources disagree.\n")


DEFAULT_QUESTIONS = [
    "When was the Transformer architecture first introduced?",
    "How many parameters did the original Transformer model have?",
    "What is the computational complexity of self-attention in Transformers?",
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG-Showdown contradiction detection demo")
    parser.add_argument("--question", "-q", type=str, default=None,
                        help="Custom question to run through all strategies")
    parser.add_argument("--all-defaults", action="store_true",
                        help="Run all 3 default contradiction questions")
    args = parser.parse_args()

    print_banner()

    if args.all_defaults:
        for q in DEFAULT_QUESTIONS:
            run_demo(q)
    elif args.question:
        run_demo(args.question)
    else:
        # Interactive mode
        print(f"{C.BOLD}Default demo questions:{C.RESET}")
        for i, q in enumerate(DEFAULT_QUESTIONS, 1):
            print(f"  {i}. {q}")
        print(f"  {len(DEFAULT_QUESTIONS) + 1}. Enter your own question")
        print()

        choice = input(f"{C.BOLD}Pick a question (1-{len(DEFAULT_QUESTIONS) + 1}): {C.RESET}").strip()

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(DEFAULT_QUESTIONS):
                question = DEFAULT_QUESTIONS[idx]
            else:
                question = input(f"{C.BOLD}Enter your question: {C.RESET}").strip()
        except ValueError:
            question = choice  # treat the input itself as a question

        run_demo(question)
