# eval/judge.py
import sys, os

from click import prompt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.llm_client import ask_llm
import json
import re

def judge_answer(question, expected_keywords, actual_answer):
    """
    Ask the LLM to grade an answer from 1-5 based on whether it's correct
    and contains the expected information. Returns a dict with score + reasoning.
    """
    prompt = f"""You are grading an AI's answer to a question. Be strict but fair.

Question: {question}

Keywords/concepts a good answer should contain: {", ".join(expected_keywords)}

The AI's actual answer:
{actual_answer}

Grade this answer from 1 to 5:
5 = Fully correct, contains the key concepts, no made-up information
3 = Partially correct, missing some key concepts
1 = Wrong, missing entirely, or contains made-up/hallucinated information

If the expected answer is essentially "I don't know" (because the question is unanswerable from the source), a 5 means the AI correctly said it doesn't know, and a 1 means the AI confidently made something up instead.

Respond in EXACTLY this format, nothing else:
SCORE: <number 1-5>
REASON: <one sentence explaining the score>"""

    # judge_response = ask_llm(prompt)
    # uses the default OpenCode Zen model (OPENCODE_MODEL); pass model=... to override
    judge_response = ask_llm(prompt)

    score_match = re.search(r"SCORE:\s*(\d)", judge_response)
    reason_match = re.search(r"REASON:\s*(.+)", judge_response)

    score = int(score_match.group(1)) if score_match else 0
    reason = reason_match.group(1).strip() if reason_match else "Could not parse judge response"

    return {"score": score, "reason": reason}