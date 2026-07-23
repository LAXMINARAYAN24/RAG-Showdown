# RAG Showdown — Upgrade Report: How Every Strategy Got Better

*A record of the two pipeline upgrades — the chunker rebuild and the grounding-prompt fix — that lifted the whole showdown from its 3.8 / 4.2 / 4.0 plateau to 4.2 / 4.6 / 4.8, with the full diagnostic trail behind each change.*

*This report covers changes **shared by all strategies**. For the individual strategies themselves, see [naive_REPORT.md](naive_REPORT.md), [advanced_REPORT.md](advanced_REPORT.md), and [corrective_REPORT.md](corrective_REPORT.md).*

---

## 1. The scoreboard, before and after

| Strategy | Baseline (old chunker, strict prompt) | After chunker fix | After prompt fix (final) | Net |
|---|:---:|:---:|:---:|:---:|
| Naive RAG | 3.80 | 3.80 | 4.20* | +0.40 |
| Advanced RAG (Reranked) | 4.20 | 4.60 | 4.60 | +0.40 |
| **Corrective RAG (Graded)** | 4.00 | 4.20 | **4.80** | **+0.80** |

*Sources: baseline [eval_all_20260714_162711.csv](eval/results/eval_all_20260714_162711.csv) + [eval_corrective_20260717_092409.csv](eval/results/eval_corrective_20260717_092409.csv); mid [eval_all_20260717_123541.csv](eval/results/eval_all_20260717_123541.csv); final [eval_all_20260717_125836.csv](eval/results/eval_all_20260717_125836.csv).*

*\*Naive's 4.20 includes one judge-noise point — see §5.*

Neither upgrade touched any strategy's architecture. Both were **shared-pipeline changes**: one at ingestion time (chunking), one at generation time (the grounding prompt). That's the headline lesson of this phase: after the reranker was in place, the remaining points were not in retrieval strategy code at all.

---

## 2. Upgrade 1 — the chunker rebuild (fixes retrieval-bound failures)

### The diagnosis (q004: "RAG vs traditional database query" — 1/5 for every strategy)

A free, local diagnostic (no API calls) traced the failure end to end:

- The answer passage **existed in the corpus** — but inside **chunk 0**, a 500-word mega-chunk containing the title page, the full table of contents, *and* the intro prose. ~2,000 characters of boilerplate diluted the three relevant sentences.
- That dilution dragged chunk 0's embedding to **rank 19 of 36** for q004 (cosine 0.640 vs 0.724 top) — below Advanced RAG's top-15 wide net, so **the cross-encoder reranker never saw it**. Reranking can't reorder what retrieval doesn't hand it.
- Corrective RAG's re-search couldn't reliably rescue it either: even when retrieved, a chunk that is 90% boilerplate gets graded IRRELEVANT.

### The change ([core/chunker.py](core/chunker.py), [core/ocr_chunker.py](core/ocr_chunker.py))

| Aspect | Before | After |
|---|---|---|
| Window | flat 500-word slabs, no overlap | **200-word windows, 50-word overlap** |
| TOC pages | chunked like prose | **detected and skipped** ("Contents" header + mostly-numeric lines) |
| Boilerplate | kept | **stripped** (© Confluent footers, page numbers) |
| OCR path | separate copy-pasted windowing | **shares `chunk_words()`** with the text path |
| Store size | 36 chunks | **118 chunks** |

The overlap matters: a passage straddling a window boundary now survives intact in at least one chunk, so an answer can't be lost to an unlucky cut.

### Verified effect

- The q004 passage moved into its own clean 200-word chunk (chunk 2).
- Raw embedding rank only improved to 27/118 (OCR-notes noise still crowds the top) — but with the wide net raised to **top-30**, the cross-encoder now ranks it **position 1 of the reranked top-5**, score 5.52 vs 3.25 runner-up. The reranker was always capable of winning this; it just needed to be shown the chunk.
- Retrieval windows were raised to match the 3× larger store: Advanced 15→30; Corrective round 1 15→30, round 2 25→60.
- Regression check: the previously-passing retrieval-bound questions (q002, q003, q008) still land their answers in the reranked top-5.

### Score impact

**q004: 1 → 5 for Advanced and Corrective.** Naive stayed at 1 (rank 27 is invisible to plain top-5 — the expected, honest limitation of the baseline).

### The trade-off it exposed (worth citing)

The same re-chunking **broke q002 for Naive RAG** (5 → 1). The new window boundary fell between the heading *"…Four Steps for Building RAG"* and the actual four steps, welding the heading to a *different* four-item list (Stream/Connect/Process/Govern — the platform capabilities). That created a **keyword-bearing decoy chunk** that cosine similarity ranks #1 with a huge margin (0.861), while the true steps sat at rank 20. Naive confidently answered with the wrong list; Advanced and Corrective survived because top-30 + cross-encoder reaches rank 20.

**Lesson: chunk-boundary choices redistribute failures, they don't just remove them.** The corpus contains two structurally similar "list of four" passages, and plain similarity search conflates them — a clean demonstration of why reranking earns its keep.

---

## 3. Upgrade 2 — the grounding-prompt fix (fixes generation-bound failures)

### The diagnosis (q006: "Is RAG always better than fine-tuning?" — regressed to 1/5)

After the chunker fix, a second diagnostic produced a surprise: retrieval for q006 was now the **best it had ever been** — the fine-tuning trade-off passage at cross-encoder position 1 (score 4.32), the handwritten "RAG vs Fine-tune" notes also in the top-5 — and Advanced RAG *still* answered "I don't know."

The bottleneck had moved. The old prompt:

> *"Answer the question using ONLY the context below. If the answer isn't in the context, say 'I don't know.'"*

For an opinion-shaped question, the context never literally states a verdict ("it depends"). Reaching one requires **synthesizing a stance from trade-off material** — and the strict prompt forbade exactly that. The LLM was obeying instructions perfectly; the instructions were wrong for nuanced questions. (Earlier partial credit on q006 had been luck — slightly different context nudging the model into answering despite the same prompt.)

### The change (one line, all three strategies)

> *"Answer the question using ONLY the context below. **If the context is relevant but doesn't state the answer directly, reason it out from what the context does say. Say 'I don't know' ONLY if the context contains nothing relevant to the question.**"*

Applied identically in [strategies/naive_rag.py](strategies/naive_rag.py), [strategies/advanced_rag.py](strategies/advanced_rag.py), and [strategies/corrective_rag.py](strategies/corrective_rag.py) to keep the comparison clean. The design intent: loosen abstention **only when relevant context exists** — trick questions (irrelevant context) must still abstain.

### Guardrail protocol

Prompt changes that weaken "I don't know" risk breaking the trick questions (q009/q010), whose full marks *depend* on abstention. So the fix was validated in two stages:

1. **Cheap spot-check first** (4 questions, one strategy): q006 1→3 ✓, q005 3 ✓, q009 5 ✓, q010 5 ✓ — only then
2. **Full 30-question three-way eval.**

### Score impact (final run, [eval_all_20260717_125836.csv](eval/results/eval_all_20260717_125836.csv))

| Question | Change | Why |
|---|---|---|
| q006 | Corrective **1 → 5**, Naive 3 → 5, Advanced 1 → 3 | permission to synthesize "it depends" from trade-off context |
| q007 (Corrective) | **3 → 5** | with room to reason, it assembled the full airline sequence incl. "checked bag" |
| q009 / q010 | **all 5/5, unchanged** | guardrail held — abstention survived for truly-irrelevant context |

Corrective RAG gained the most because it had the most to lose from strictness: its grading gate already filters context hard, and the strict prompt was double-penalizing anything that survived grading but wasn't a verbatim answer. With the prompt fixed, the gate's precision finally pays off — **4.80, the new leader**.

---

## 4. The method that found both upgrades

Each upgrade followed the same loop, and the loop is arguably the real deliverable:

1. **Diagnose locally before touching anything** — rank-check scripts against the store cost nothing and pinpointed the failing layer each time (chunk dilution → retrieval; perfect retrieval + refusal → generation).
2. **Fix the layer the evidence points at**, not the layer that's most interesting to build. Both fixes were shared-pipeline, not new strategies.
3. **Spot-check cheaply, then run the full eval** — with explicit guardrail questions chosen *before* the change (q009/q010 for the prompt; q002/q003/q008 for the chunker).
4. **Record what broke, not just what improved** (q002-naive decoy; judge noise below).

Notably, what was *not* built: a fourth retrieval strategy. The evidence twice said the remaining points weren't in retrieval — and twice it was right.

---

## 5. Honest caveats on the final numbers

- **Naive's q004 "5/5" is judge noise, not a fix.** Naive still abstained ("cannot determine from the given text") — the same behavior scored 1/5 in every previous run — but this time the judge praised it as an appropriate "I don't know." Same answer, opposite score. Naive's real level is ~3.8–4.0. The judge has variance on abstention-type answers; single-run scores on such questions should be treated as ±1.
- **q005 is stuck at 3/5 for all three strategies.** Answers correctly explain forgetting/retraining but never use the judge's keywords "expertise" and "investment." It's a keyword-coverage miss, not a comprehension failure — and the last consistently-lost point.
- **Latency/cost columns across runs are not comparable** — the LLM cache (keyed on prompt text) makes repeat runs nearly free but first runs pay full API latency; call counts remain the honest cost metric (Naive/Advanced: 1 call per question; Corrective: 2–4).

---

## 6. Where the showdown stands

| Failure mode | Status |
|---|---|
| Retrieval-bound (q004) | **solved** — chunker + wider net |
| Generation-bound (q005 partially, q006, q007-corrective) | **solved** except q005's keyword miss |
| Trick-question safety (q009, q010) | **held throughout** — structurally enforced in Corrective |
| Boundary-decoy conflation (q002-naive) | **open by design** — kept as a documented baseline weakness |
| Judge variance on abstention | **open** — affects score stability ±1 on abstention answers |

**Final standings: Corrective 4.80 > Advanced 4.60 > Naive 4.20** — with the caveat that the gap between Corrective and Advanced (one question, q006) is within single-run judge variance. A stability re-run of `--strategy all` would firm that up cheaply, since most responses are now cached.

---

## 7. How to reproduce

```bash
# 1. Rebuild the store with the new chunker (re-runs OCR, ~2 min)
python scripts/ingest.py       # -> 118 chunks

# 2. Run the full three-way showdown with the new prompt
python -u eval/run_eval.py --strategy all
# -> eval/results/eval_all_<timestamp>.csv + per-strategy averages
```

---

*Report generated from the actual diagnostic outputs (chunk ranks, cross-encoder scores) and the four real eval runs listed in §1 — not predictions.*
