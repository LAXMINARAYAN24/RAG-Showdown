# RAG Showdown — Corrective RAG Strategy Report

*The third contestant: what "Corrective RAG (Graded)" adds on top of Advanced RAG, why it was built, and an honest, data-backed assessment — including the one question where it made things worse.*

*This report assumes you've read [naive_REPORT.md](naive_REPORT.md) (shared pipeline) and [advanced_REPORT.md](advanced_REPORT.md) (wide-net + reranking). Here we focus on **what's different**.*

---

## 1. What this strategy is

**Corrective RAG (Graded)** is the third strategy in `strategies/` ([strategies/corrective_rag.py](strategies/corrective_rag.py)). It keeps Advanced RAG's entire retrieval stack (wide net of 15 → cross-encoder rerank → top 5) and adds a **self-checking layer** on top:

1. **Grade** — before answering, ask the LLM to judge each retrieved chunk as RELEVANT or IRRELEVANT to the question (one combined call for all chunks).
2. **Correct** — if fewer than 2 chunks pass, assume retrieval failed: **reformulate the query** with the LLM and re-search with a wider net (top 25 → rerank → top 8), then grade again.
3. **Answer** — generate only from chunks that survived grading; if nothing survives both rounds, answer *"I don't know"* outright.

**In one sentence:** *Don't just trust whatever retrieval returns — check it, and if it looks bad, search again differently before answering.*

The bet: Advanced RAG's known ceiling (documented in its report) is that reranking can only reorder what the first-pass search hands it. q004 was the poster child — the right chunk sat at **rank 19 of 36**, below the wide net's top 15, so the reranker never saw it. Corrective RAG's round-2 search (`top_k=25`) was designed to reach exactly that depth.

### Why this strategy was built (the q004 diagnostic)

Before building it, a free local diagnostic pinned down q004's failure precisely:

- The answer passage ("distinguish RAG from... traditional database queries... vector embedding... instead of exact keyword matches") **does exist** in the corpus — in **chunk 0**.
- Chunk 0 ranked **19th of 36** for q004 (cosine 0.640 vs 0.724 for the top hit) — outside Advanced RAG's top-15 net.
- Why so low: chunk 0 is a bloated mega-chunk containing the title page, the full table of contents, *and* the intro prose. The three relevant sentences are diluted by ~2,000 characters of boilerplate, dragging the embedding away from the question.
- What outranks it: noisy OCR chunks from the handwritten-notes PDF — topically RAG-ish, answering nothing.

So q004 is simultaneously a **chunking problem** (fix the mega-chunk) and a **retrieval-depth problem** (rank 19 is reachable with a wider net). Corrective RAG attacks the second; the first remains open (see §8).

---

## 2. High-level architecture

```
   question
      │
      ▼
  embed → store.search(top_k=15) → rerank(top_n=5)     ◀── identical to Advanced RAG
      │
      ▼
  GRADE (1 LLM call): "chunk 1: RELEVANT / chunk 2: IRRELEVANT / ..."   ◀── NEW
      │
      ├── ≥2 chunks RELEVANT ──────────────────────────┐
      │                                                 │
      └── <2 chunks RELEVANT (retrieval looks bad):     │
            REFORMULATE query (1 LLM call)              │   ◀── NEW round 2
            embed → search(top_k=25) → rerank(top_n=8)  │
            GRADE again (1 LLM call), merge survivors ──┤
                                                        ▼
                            any relevant chunks?  ──── no ──▶  "I don't know."
                                    │ yes
                                    ▼
                  build "[Source: N]" prompt from SURVIVORS ONLY
                                    │
                                    ▼
                        ask_llm → answer + latency + retrieval_rounds
```

Cost per question: **2 LLM calls** on the happy path (grade + answer) vs 1 for naive/advanced; **4 calls** when correction triggers (grade, reformulate, re-grade, answer). The result dict adds a `retrieval_rounds` field so logs show when the correction kicked in.

---

## 3. The new machinery, step by step

Inside `CorrectiveRAG.answer_question()`:

1. **`_retrieve(question, top_k=15, top_n=5)`** — same embed → wide search → cross-encoder rerank as Advanced RAG (shared `core/` modules, nothing duplicated).
2. **`_grade_chunks(question, chunks)`** — one combined prompt lists all chunks as `[Chunk N]` blocks and demands a strict `N: RELEVANT` / `N: IRRELEVANT` line per chunk. The parser is deliberately forgiving (skips malformed lines, ignores stray text) and **fails open per line** — a chunk is kept unless explicitly graded IRRELEVANT.
3. **Decision gate** — `MIN_RELEVANT = 2`. Fewer survivors than that triggers round 2.
4. **`_reformulate(question)`** — asks the LLM to rewrite the question as a search query with different wording/synonyms, aiming to shift the embedding toward passages the original phrasing missed.
5. **Round 2** — `_retrieve(new_query, top_k=25, top_n=8)`, grade *against the original question*, merge survivors (deduplicated by chunk text).
6. **Answer or abstain** — survivors go into the same grounding prompt naive/advanced use; zero survivors → hard-coded `"I don't know."` with **no** generation call.

Same return contract as the other strategies, so `run_eval.py` treats all three identically (`--strategy corrective`, or included in `all`).

---

## 4. Tech stack (delta from advanced)

| Layer | Advanced RAG | Corrective RAG |
|---|---|---|
| First-pass retrieval | top-15 → rerank → 5 | identical |
| Chunk grading | — | **1 combined LLM call** (RELEVANT/IRRELEVANT per chunk) |
| Retry on bad retrieval | — | **LLM query reformulation + top-25 → rerank → 8** |
| Abstention | LLM decides from prompt | **structural**: zero graded-relevant chunks → "I don't know" without calling the LLM |
| LLM calls per question | 1 | **2 (happy path) – 4 (corrective path)** |
| LLM backend | OpenCode Zen (`deepseek-v4-flash-free`) | same |

---

## 5. Evaluation results — the three-way showdown

Corrective RAG ran through the same 10-question bank and LLM judge. Sources: [eval/results/eval_corrective_20260717_092409.csv](eval/results/eval_corrective_20260717_092409.csv) (corrective), [eval/results/eval_all_20260714_162711.csv](eval/results/eval_all_20260714_162711.csv) (naive + advanced baseline).

### Overall

| Strategy | Average score | Total | Avg latency |
|---|---|---|---|
| Naive RAG | 3.8 / 5 | 38 / 50 | ~0.1 s* |
| **Advanced RAG (Reranked)** | **4.2 / 5** | **42 / 50** | ~0.5 s* |
| Corrective RAG (Graded) | 4.0 / 5 | 40 / 50 | **35.7 s** |

*\*The naive/advanced run benefited from the LLM response cache added in `llm_client.py` — repeat questions hit the cache, hence sub-second latencies. Corrective's grading prompts are unique per run, so it paid full API latency on every call. Latency columns are not apples-to-apples; call counts are the honest cost metric (see §6).*

**The headline finding: Corrective RAG scored *worse* than Advanced RAG.** More machinery, more LLM calls, lower score. Here's exactly where and why.

### Head-to-head, question by question

| # | Question | Category | Naive | Advanced | Corrective | Verdict |
|---|---|---|:---:|:---:|:---:|---|
| q001 | What does RAG stand for? | simple_factual | 5 | 5 | 5 | Tie ✅ |
| q002 | Four key steps for RAG? | simple_factual | 5 | 5 | 5 | Tie ✅ |
| q003 | Three vector stores? | simple_factual | 5 | 5 | 5 | Tie ✅ |
| q004 | RAG vs traditional DB query? | compound | 1 | 1 | 1 | Tie ❌ — still unsolved |
| q005 | Fine-tuning worse for freshness? | compound | 3 | 3 | 3 | Tie ~ |
| q006 | Is RAG always better than fine-tuning? | ambiguous | 3 | 3 | **1** | **Corrective REGRESSED ▼** |
| q007 | Airline chatbot steps? | multi_hop | 3 | 5 | 5 | Advanced-era gain held ✅ |
| q008 | Data NOT to vectorize? | needs_structure | 3 | 5 | 5 | Advanced-era gain held ✅ |
| q009 | Autonomous vehicles? (trick) | trick_no_answer | 5 | 5 | 5 | Tie ✅ |
| q010 | Enterprise pricing? (trick) | trick_no_answer | 5 | 5 | 5 | Tie ✅ |

### The regression: q006 (3 → 1)

This is the most instructive result in the run. Naive and Advanced both gave a partial nuanced answer to *"Is RAG always better than fine-tuning?"* and earned a 3. Corrective answered **"I don't know"** and earned a 1.

What happened: the grading step **rejected chunks that the other strategies happily answered from**. q006 is an ambiguous/nuanced question — the corpus contains trade-off discussion spread thinly across chunks, none of which screams "this answers the question" to a binary RELEVANT/IRRELEVANT grader. Strict grading filtered them out, correction couldn't find anything better, and the strategy abstained where a partial answer would have scored higher.

**The lesson: a relevance gate is a double-edged sword.** It's exactly right for trick questions (abstain instead of hallucinating) — but on nuanced questions where *weak* context still supports a *partial* answer, binary grading throws away usable signal. The judge rewards partial answers (3) over honest abstention (1), and Corrective RAG optimizes for the wrong side of that trade.

### The target miss: q004 (still 1)

The question this strategy was *built for* still failed. The corrective round ran (grading found the round-1 chunks irrelevant, reformulation + top-25 search triggered) but the final answer was still "I don't know." The likely chain of failure:

- The reformulated query re-ranks the whole store — chunk 0's rank under the *new* query isn't guaranteed to be ≤25, and even when retrieved, it must then survive the cross-encoder (top 8) **and** the relevance grader.
- Chunk 0 is the diluted mega-chunk (title + TOC + intro). A grader shown 2,000 characters that are 90% boilerplate can reasonably call it IRRELEVANT even though three key sentences are buried inside.

**The lesson: no amount of retrieval-time cleverness reliably rescues a badly-chunked document.** q004 is a *chunking* problem first. Every strategy inherits the same broken chunk 0; they all fail the same way. The fix belongs in `core/chunker.py`/ingestion, not in a fourth retrieval strategy.

### What held up

- **Both Advanced-era gains persisted** (q007, q008 at 5) — grading did not disturb questions where retrieval was already good. The graded-context prompt still produced complete answers.
- **Trick questions stayed perfect** (q009, q010 at 5) — and structurally so: abstention is now enforced by the grading gate, not just by hoping the LLM follows the prompt. That's a genuine robustness improvement even though the score didn't move.

---

## 6. Cost accounting

| Strategy | LLM calls / question | This run (10 q) |
|---|---|---|
| Naive | 1 | 10 calls |
| Advanced | 1 | 10 calls |
| Corrective | 2–4 | **~26 calls** (2 on happy path; 4 on the ~3 questions that triggered round 2) |

Corrective RAG costs **2–4× the API calls** of Advanced RAG for **−0.2 average score** in this run. On the current corpus and question bank, it is not the efficient frontier — that remains Advanced RAG.

---

## 7. Design assessment

### What's done well
- ✅ **Correct CRAG-style pattern** — retrieve → grade → conditionally re-search → generate-or-abstain is the canonical Corrective RAG loop, implemented compactly.
- ✅ **One combined grading call** — grading all chunks in a single prompt keeps cost at 2 calls (happy path) instead of 6+ with per-chunk grading.
- ✅ **Structural abstention** — "I don't know" on zero relevant chunks is enforced in code, not delegated to prompt obedience. Trick-question safety no longer depends on the LLM's mood.
- ✅ **Robust grade parsing** — tolerant line parser; malformed grader output degrades gracefully instead of crashing (verified offline with a mocked LLM before spending any API calls).
- ✅ **Clean integration** — same return contract, shared `core/` modules, `--strategy corrective` flag; the showdown harness needed only three lines of changes.
- ✅ **Built on a diagnosis, evaluated honestly** — the strategy targeted a measured failure (q004 at rank 19), and the report records that it missed, and why.

### Limitations & risks
- ⚠️ **Binary grading is too blunt for nuanced questions** — the q006 regression. RELEVANT/IRRELEVANT has no notion of "weakly supportive," so partial-answer material gets discarded. A 1–5 relevance scale with a low keep-threshold, or "keep if nothing better exists," would soften this.
- ⚠️ **Grading is only as good as the grader** — one noisy LLM judgment now gates the whole pipeline. A single bad grading call can zero out a question.
- ⚠️ **Reformulation is a lottery** — one paraphrase, unverified. If the new query doesn't move the target chunk into range, round 2 buys nothing (q004).
- ⚠️ **Can't fix upstream chunking** — the strategy's target failure (q004) is ultimately a corpus-preparation defect it cannot reach.
- ⚠️ **2–4× call cost** — meaningful at scale, and this run shows the extra spend isn't currently buying score.
- ⚠️ **`MIN_RELEVANT=2` is a guess** — untuned; on a 36-chunk corpus even 1 genuinely relevant chunk is often enough to answer well.

---

## 8. Suggested next steps

1. **Fix chunk 0 in the chunker** — split title/TOC boilerplate from intro prose (or strip TOC pages at ingestion). This is the highest-leverage change in the whole project right now: it likely fixes q004 for *all three* strategies at zero query-time cost.
2. **Soften the grading scale** — grade 1–5 and keep chunks ≥2, or fall back to "best available chunks" instead of abstaining when everything is graded weak. Directly targets the q006 regression.
3. **Verify reformulation helped before using it** — check whether the round-2 pool actually contains new chunks; if not, widen `top_k` again instead of re-grading the same material.
4. **Hybrid search (dense + BM25)** — q004's phrasing shares exact keywords with the target passage ("traditional database"); keyword search would rank chunk 0 near the top immediately. Still the most promising retrieval-side fix.
5. **Rerun the full three-way eval after the chunking fix** — the current comparison is fair but every strategy is handicapped by the same corpus defect; the ranking could shift once it's removed.

---

## 9. How to run it

```bash
# Ensure OPENCODE_API_KEY is set in .env

# Build the store once (shared by all strategies)
python scripts/ingest.py

# Run just the corrective strategy through the question bank + judge
python eval/run_eval.py --strategy corrective

# Or the full three-way showdown
python eval/run_eval.py --strategy all
# -> writes eval/results/eval_*.csv and prints per-strategy averages
```

---

*Report generated from a full read of `strategies/corrective_rag.py`, the q004 retrieval diagnostic (chunk 0 at rank 19/36), and the real scored results in `eval/results/eval_corrective_20260717_092409.csv` vs the `eval_all_20260714_162711.csv` baseline (not predictions).*
