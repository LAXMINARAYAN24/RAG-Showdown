# RAG Showdown — Advanced RAG Strategy Report

*A walkthrough of the second contestant in the showdown: what "Advanced RAG (Reranked)" changes over the naive baseline, how the reranking stage works, and an honest, data-backed assessment of whether it actually helped.*

*This report assumes you've read [naive_REPORT.md](naive_REPORT.md), which covers the shared pipeline (ingestion, chunking, embeddings, vector store). Here we focus on **what's different**.*

---

## 1. What this strategy is

**Advanced RAG (Reranked)** is the second strategy in the `strategies/` folder — the first real challenger to `NaiveRAG`. It keeps the entire ingestion pipeline identical (same PDFs, same chunks, same `bge-base` embeddings, same pickled store) and changes **only the retrieval step**.

The core idea is a classic, well-proven RAG upgrade: **retrieve wide, then rerank narrow.**

- **Naive RAG** embeds the question, grabs the **top 5** chunks by cosine similarity, and stuffs them straight into the prompt.
- **Advanced RAG** embeds the question, grabs a **wider net of the top 15** chunks by cosine similarity, then passes all 15 through a **cross-encoder reranker** that scores each chunk *specifically against the question*, and keeps only the **best 5** for the prompt.

**In one sentence:** *Cast a wider net with cheap embedding search, then use a smarter (but slower) model to pick the truly relevant chunks before answering.*

The bet: embedding (bi-encoder) similarity is fast but coarse — it can rank a loosely-related chunk above the one that actually answers the question. A cross-encoder reads the question and chunk *together*, so it judges true relevance far better. If the answer was sitting at rank 6–15 in the naive version (and therefore never made it into the prompt), the reranker can promote it into the top 5.

---

## 2. High-level architecture

The **only** part that changes is the query-time retrieval. Everything upstream (ingestion) and downstream (LLM call, logging) is shared.

```
   question
      │
      ▼
  embedder.py ─▶ query vector
      │
      ▼
  store.search(top_k=15)          ◀── WIDER net than naive's top_k=5
      │
      │  15 (chunk, cosine_score) pairs
      ▼
  reranker.rerank(top_n=5)         ◀── NEW STAGE (cross-encoder)
      │  re-scores all 15 against the question,
      │  sorts by relevance, keeps best 5
      ▼
  build "[Source: N]" prompt  ─▶  llm_client (OpenCode Zen / DeepSeek)
      │
      ▼
  answer + latency ─▶ eval/judge.py ─▶ eval/results/*.csv
```

Compared to naive, there are **two extra costs**: a bigger similarity search (15 vs 5 — negligible on this tiny store) and one cross-encoder forward pass over 15 pairs (the real added latency).

---

## 3. The retrieval upgrade, step by step

Inside `AdvancedRAG.answer_question()` ([strategies/advanced_rag.py](strategies/advanced_rag.py)):

1. **Embed the question** → `query_vec` (same as naive).
2. **Wide retrieval** → `store.search(query_vec, top_k=15)` returns 15 `(chunk_text, cosine_score)` pairs — 3× naive's window.
3. **Rerank** → `rerank(question, wide_results, top_n=5)`:
   - Build 15 `(question, chunk_text)` pairs.
   - Run the cross-encoder `_reranker.predict(pairs)` to get a fresh relevance score per chunk.
   - Sort descending, keep the **top 5**.
4. **Assemble the prompt** — the surviving 5 chunks become `[Source: 1..5]` blocks in the exact same grounding prompt naive uses ("answer only from context / say I don't know").
5. **Call the LLM** and measure latency.
6. **Return** the same dict contract (`answer`, `chunks_used`, `strategy_name`, `latency_seconds`) — so the eval harness treats both strategies identically.

The uniform return contract is what makes the "showdown" work: `run_eval.py` loops over both strategies with the same question bank and judge, and neither knows the other exists.

---

## 4. The new module — `core/reranker.py`

This is the heart of the strategy. It's small ([core/reranker.py](strategies/../core/reranker.py)):

```python
from sentence_transformers import CrossEncoder
_reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(question, chunks_with_scores, top_n=5):
    pairs = [(question, chunk_text) for chunk_text, _ in chunks_with_scores]
    rerank_scores = _reranker.predict(pairs)
    reranked = list(zip([c for c, _ in chunks_with_scores], rerank_scores))
    reranked.sort(key=lambda x: x[1], reverse=True)
    return reranked[:top_n]
```

**Bi-encoder vs cross-encoder — why this helps:**

| | Bi-encoder (embeddings, `bge-base`) | Cross-encoder (reranker, `ms-marco-MiniLM`) |
|---|---|---|
| How it scores | Embeds question and chunk **separately**, compares vectors | Reads question + chunk **together** in one pass |
| Speed | Very fast (vectors precomputed once) | Slower (one model pass **per chunk, per query**) |
| Accuracy | Coarse — good recall, weaker precision | Sharp — much better at true relevance |
| Role here | Cheap **first-pass filter** (231 chunks → 15) | Expensive **precision re-ranker** (15 → 5) |

The two-stage design is deliberate: you can't afford to run the cross-encoder over all 231 chunks for every question, but you *can* afford it over 15. So the embedding search does the cheap bulk filtering, and the cross-encoder does the expensive fine-grained ranking on the shortlist. This is exactly how production retrieval stacks (e.g. Cohere Rerank, ColBERT-style pipelines) are built.

- The `ms-marco-MiniLM-L-6-v2` model is trained on the MS MARCO passage-ranking dataset — i.e. it's specifically tuned for *"how relevant is this passage to this query."* A great fit for RAG.
- Like the embedder, it's loaded **once at import** (`_reranker`), so the model weights aren't reloaded per question.
- The incoming cosine scores are discarded (`_`) — the cross-encoder score fully replaces them.

---

## 5. Tech stack (delta from naive)

| Layer | Naive RAG | Advanced RAG |
|---|---|---|
| Retrieval window | top-5 cosine | **top-15 cosine → rerank → top-5** |
| Reranker | — | **`cross-encoder/ms-marco-MiniLM-L-6-v2`** via `sentence-transformers` |
| Everything else (chunker, OCR, embedder, store, prompt) | shared | shared |
| LLM | *(now)* OpenCode Zen — `deepseek-v4-flash-free` | same |

> **Note:** The LLM backend was migrated from Google Gemini to **OpenCode Zen** (OpenAI-compatible endpoint, default model `deepseek-v4-flash-free`) for higher rate limits. Both strategies now share this backend, so the eval below is an apples-to-apples comparison — the only variable between them is the retrieval/rerank stage.

---

## 6. Evaluation results — the actual showdown

Both strategies were run through the same 10-question bank and scored 1–5 by the LLM judge (`eval/judge.py`). Source: [eval/results/eval_all_20260710_180216.csv](eval/results/eval_all_20260710_180216.csv).

### Overall

| Strategy | Average score | Total |
|---|---|---|
| Naive RAG | **3.8 / 5** | 38 / 50 |
| **Advanced RAG (Reranked)** | **4.2 / 5** | **42 / 50** |

**Advanced RAG wins by +0.4 (+8 percentage points).**

### Head-to-head, question by question

| # | Question | Category | Naive | Advanced | Verdict |
|---|---|---|:---:|:---:|---|
| q001 | What does RAG stand for? | simple_factual | 5 | 5 | Tie ✅ |
| q002 | Four key steps for RAG architecture? | simple_factual | 5 | 5 | Tie ✅ |
| q003 | Three vector stores Confluent integrates with? | simple_factual | 5 | 5 | Tie ✅ |
| q004 | RAG vs traditional DB query? | compound | 1 | 1 | Tie ❌ (both "I don't know") |
| q005 | Why fine-tuning worse than RAG for freshness? | compound | 3 | 3 | Tie ~ |
| q006 | Is RAG always better than fine-tuning? | ambiguous | 3 | 3 | Tie ~ |
| q007 | Airline chatbot reasoning steps? | multi_hop | 3 | **5** | **Advanced wins ▲** |
| q008 | Data that should NOT be vectorized? | needs_structure | 3 | **5** | **Advanced wins ▲** |
| q009 | RAG for autonomous vehicles? (trick) | trick_no_answer | 5 | 5 | Tie ✅ |
| q010 | Confluent Enterprise pricing? (trick) | trick_no_answer | 5 | 5 | Tie ✅ |

### Where the reranker actually helped

The entire +0.4 gain comes from **two questions**, and the reason is exactly what the reranker is designed to fix:

- **q007 (multi-hop, 3 → 5):** The airline example is spread across several chunks. Naive retrieved a partial view and omitted the "checked bag" step. Advanced's wider net + rerank surfaced the full sequence — it correctly listed **flight upgrade → checked bag → frequent flyer miles → prompt the LLM**, the complete chain.
- **q008 (needs_structure, 3 → 5):** Naive listed identifiers and product IDs but **missed "aggregated analytics."** Advanced's reranker pulled in the chunk containing the full list, so the answer covered all three required concepts.

In both cases the missing information existed in the corpus but sat *below rank 5* in the naive version — precisely the failure mode reranking targets.

### Where it did *not* help (the honest part)

- **q004 (compound, 1 → 1):** Both strategies said *"I don't know."* The vector/embedding/semantic-vs-keyword content the judge wanted was either not in the corpus in a retrievable form, or buried beyond the top 15. Reranking can only reorder what the first-pass search retrieves — **if the answer isn't in the wide-net 15, the reranker can't invent it.** This is the strategy's ceiling.
- **q005 & q006 (both 3 → 3):** Partial credit for both. These are nuance/keyword-completeness misses, not retrieval misses — the judge wanted specific words (`forget`, `expertise`, `investment`; `depends`, `however`, `both`). The right chunk was likely present in both; the **generation/prompt**, not retrieval, is the bottleneck here. Reranking doesn't touch that.

### Latency cost

| Strategy | Avg latency | Range |
|---|---|---|
| Naive RAG | ~18.0 s | 5.8 – 57.1 s |
| Advanced RAG | ~16.6 s | 4.5 – 35.2 s |

Counter-intuitively, Advanced was **not** meaningfully slower on average here — the cross-encoder pass over 15 short chunks is cheap, and both runs are dominated by LLM API latency and occasional rate-limit retries (the 57 s and 35 s outliers). On a larger corpus the reranker cost would grow, but on ~231 chunks it's noise.

### Prediction vs reality

The appendix of the naive report *predicted* Advanced would hit ~4.6/5 with gains on q003, q004, and q006. **Reality: 4.2/5, with gains on q007 and q008 instead.** The prediction was directionally right (advanced > naive) but wrong about *which* questions would improve — q003 was already solved by naive in this run, and q004/q006 turned out to be corpus/generation limits that reranking can't fix. A good reminder to measure, not guess.

---

## 7. Design assessment

### What's done well
- ✅ **Textbook two-stage retrieval** — cheap bi-encoder recall → expensive cross-encoder precision. This is the correct, production-grade pattern.
- ✅ **Right model choice** — `ms-marco-MiniLM` is purpose-trained for query-passage relevance.
- ✅ **Minimal, surgical change** — reuses the entire pipeline and the same prompt/return contract, so the comparison is clean and the code stays DRY.
- ✅ **Measurable, real improvement** — +0.4/5 on a blind judge, driven by exactly the questions reranking should fix (multi-chunk / completeness).
- ✅ **Model loaded once** — no per-query weight reloading.

### Limitations & risks
- ⚠️ **Can't fix retrieval it never sees** — if the answer isn't in the top-15 wide net (q004), reranking is powerless. The fix is hybrid search (dense + BM25) or a larger wide-net `top_k`, not a better reranker.
- ⚠️ **Doesn't help generation-bound misses** — q005/q006 are prompt/answer-completeness problems; reranking left them unchanged.
- ⚠️ **Fixed windows (15 → 5)** — both are hard-coded. On some questions 5 final chunks may still be too few (or too many, diluting context).
- ⚠️ **Reranker latency scales with wide-net size** — cheap at 15, would need batching/limits if the first pass grew to hundreds.
- ⚠️ **No score threshold** — it always returns 5 chunks even if the cross-encoder thinks all 15 are weakly relevant; a relevance cutoff could reduce noise on trick questions.

---

## 8. Suggested next steps

1. **Add hybrid search** (dense + BM25 keyword) as the first stage — this is the most likely fix for q004, where pure embedding search missed the answer entirely.
2. **Tune the windows** — try wide-net top-20/30 and final top-3/7 and measure the score/latency trade-off.
3. **Add a relevance threshold** in `rerank()` — drop chunks below a cross-encoder score cutoff so weak contexts don't dilute the prompt.
4. **Attack the generation-bound questions** (q005, q006) with prompt tweaks or query rewriting — reranking has already done its job there.
5. **Add a third contestant** (e.g. `HybridRAG` or query-rewriting RAG) to keep the showdown going.

---

## 9. How to run it

```bash
# Ensure OPENCODE_API_KEY is set in .env (see naive_REPORT.md / .env)

# Build the store once (shared with naive)
python scripts/ingest.py

# Run the advanced strategy through the full question bank + judge
python eval/run_eval.py --strategy advanced

# Or run both and compare head-to-head
python eval/run_eval.py --strategy all
# -> writes eval/results/eval_all_<timestamp>.csv and prints a per-strategy average
```

---

*Report generated from a full read of `strategies/advanced_rag.py`, `core/reranker.py`, `eval/run_eval.py`, `eval/judge.py`, the question bank, and the real scored results in `eval/results/eval_all_20260710_180216.csv` (not predictions).*
