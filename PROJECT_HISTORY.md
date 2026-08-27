# RAG Showdown — Complete Project History & Build Report

*The full record of this project from scratch: what was built, every dependency and setup step, every bug fixed, every upgrade made, the scores after each change, the migration to Chroma, and the addition of Conflict-Aware RAG for contradiction detection. This is the master document; the per-phase reports ([naive_REPORT.md](naive_REPORT.md), [advanced_REPORT.md](advanced_REPORT.md), [corrective_REPORT.md](corrective_REPORT.md), [conflict_aware_REPORT.md](conflict_aware_REPORT.md), [upgrade_REPORT.md](upgrade_REPORT.md)) go deeper on individual pieces.*

*Last updated: 2026-08-27.*

---

## 1. What this project is

A head-to-head comparison ("showdown") of four Retrieval-Augmented Generation strategies — **Naive**, **Advanced (Reranked)**, **Corrective (Graded)**, and **Conflict-Aware (NLI-based)** — over a document corpus, scored by an LLM-as-judge harness on a 13-question bank (including 3 contradiction-focused questions). The deliverable is not just the strategies but a **documented methodology**: diagnose locally with free scripts → fix the layer the evidence points at → guardrail-check → re-run the eval.

---

## 2. Project layout

```
rag-showdown/
├── core/                    # shared building blocks
│   ├── chunker.py           # PDF/text → text chunks (200-word windows, 50 overlap, TOC/boilerplate stripping)
│   ├── ocr_chunker.py       # scanned-PDF fallback: Tesseract OCR (lazy-loaded)
│   ├── embedder.py          # BAAI/bge-base-en-v1.5 sentence-transformer
│   ├── reranker.py          # cross-encoder/ms-marco-MiniLM-L-6-v2
│   ├── contradiction_detector.py # cross-encoder/nli-deberta-v3-small (pairwise NLI conflict detection)
│   ├── vector_store.py      # ChromaVectorStore (persistent, uuid4 indexing) + legacy SimpleVectorStore
│   └── llm_client.py        # OpenAI client (OpenCode Zen / Nemotron / DeepSeek) + retries + cache
├── strategies/
│   ├── naive_rag.py         # embed → top-5 cosine → answer
│   ├── advanced_rag.py      # embed → top-60 wide net → cross-encoder rerank → top-5 → answer
│   ├── corrective_rag.py    # advanced + LLM relevance grading + reformulate-and-retry + structural abstention
│   └── conflict_aware_rag.py # advanced + pairwise NLI conflict detection + conflict-aware prompt
├── eval/
│   ├── judge.py             # LLM-as-judge, 1–5 score per answer + rationale
│   ├── run_eval.py          # harness: --strategy naive|advanced|corrective|conflict_aware|all
│   └── results/*.csv        # every scored run, timestamped
├── questions/question_bank.yaml   # 13 questions across 7 categories (including contradictions)
├── scripts/                 # ingest.py, ask.py, demo.py, chunk_retrieval.py, quick_test.py, export_log.py
├── corpus/
│   ├── raw/                 # 18 source PDFs + 3 contradiction test text files (A, B, C)
│   └── processed/chroma/    # persistent Chroma DB (git-ignored)
└── logs/                    # qa_log.jsonl Q&A audit trail
```

---

## 3. Environment & dependencies

### Runtime
- **Windows 11**, Git Bash / PowerShell
- **System Python 3.12** (`AppData\Local\Programs\Python\Python312`) — note: this, **not** the project `venv/`, is what actually runs the project; packages must be installed there
- All scripts must run **from inside `rag-showdown/`** — every path in the codebase is relative (this bit us early: `FileNotFoundError: logs/qa_log.jsonl` when running from the parent folder)

### Python packages (installed over the course of the project)
| Package | Purpose |
|---|---|
| `sentence-transformers` | embeddings (bge-base) + cross-encoder reranker |
| `pypdf` | PDF text extraction |
| `pytesseract` + `pdf2image` | OCR fallback for scanned PDFs |
| `numpy` | legacy vector math |
| `pyyaml` | question bank |
| `python-dotenv` | .env secrets |
| `openai` | OpenAI-compatible client for OpenCode Zen |
| `google-genai` | (legacy) Gemini client, since retired |
| `chromadb` (1.5.9) | persistent vector DB (final store) |

### External (non-pip) tools
- **Tesseract OCR** at `C:\Program Files\Tesseract-OCR\tesseract.exe`
- **Poppler** at `C:\Users\sahul\Downloads\Release-26.02.0-0\poppler-26.02.0\Library\bin` (for pdf2image)

### Models used
| Role | Model | Where it runs |
|---|---|---|
| Embeddings | `BAAI/bge-base-en-v1.5` | local (sentence-transformers) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | local |
| Generation + judging | `deepseek-v4-flash-free` via **OpenCode Zen** (`https://opencode.ai/zen/v1`, OpenAI-compatible) | API |

### Secrets
`.env` (git-ignored) holds `OPENCODE_API_KEY`; `.env.example` is the committed template.

---

## 4. The LLM-backend saga (how we got to OpenCode Zen)

The project started on **Google Gemini** (`gemini-2.5-flash` → `-lite`) and hit escalating problems:

1. **429 RESOURCE_EXHAUSTED** — free-tier rate limits. Fixed first with retry/backoff in `ask_llm()` (15/30/45/60/75s), then with pacing sleeps in the eval loop.
2. **Persistent 429 across 75s+** — diagnosed as the **daily quota (RPD)**, which no backoff can fix.
3. **403 PERMISSION_DENIED on a second account** — a decision point: we explicitly **rejected key/account rotation** as a workaround (against ToS, detected within one request anyway) and instead moved to a legitimate second provider.
4. **Migrated to OpenCode Zen** (own account, own key, OpenAI-compatible) — retired the Gemini path.
5. **`httpx.ConnectError [WinError 10054]`** (TLS connection reset) crashed a run — fixed by broadening the retry handler to also catch transport-level errors (`httpx.TransportError`) and server 5xx, with their own 5/10/15s backoff.
6. **Response cache added** to `llm_client.py` — SHA-256 keyed on `(model, prompt)`; identical prompts are free and instant on re-runs, which made stability re-runs and guardrail checks nearly zero-cost.

Bugs fixed along the way in the eval runner: a stray pasted block at the top of `run_eval.py` that referenced variables before definition (`NameError`), and `from datetime import datetime, time` shadowing the `time` module (breaking `time.sleep`).

---

## 5. The three strategies

### Naive RAG (baseline)
Embed the question → top-5 cosine chunks → grounding prompt → answer. One LLM call.

### Advanced RAG (Reranked)
Embed → **wide net** (top-k cosine; currently 60) → **cross-encoder reranks** all candidates against the question → keep best 5 → same prompt. One LLM call. The two-stage design: cheap bi-encoder recall, expensive cross-encoder precision.

### Corrective RAG (Graded)
Advanced retrieval → **grade** all chunks RELEVANT/IRRELEVANT in **one combined LLM call** → if <2 relevant, **reformulate the query** (LLM) and re-search wider (currently 120) → answer only from graded-relevant chunks → **structural abstention**: zero survivors ⇒ "I don't know" with no generation call. 2–4 LLM calls. Returns `retrieval_rounds` for observability.

All three share the same grounding prompt and return contract (`answer`, `chunks_used`, `strategy_name`, `latency_seconds`), so `run_eval.py` treats them identically.

---

## 6. Evaluation harness

- `questions/question_bank.yaml` — 10 questions: 3 simple-factual, 2 compound, 1 ambiguous, 1 multi-hop, 1 needs-structure, 2 trick (unanswerable).
- `eval/judge.py` — LLM judge scores each answer 1–5 against `expected_answer_contains` keywords.
- `eval/run_eval.py --strategy naive|advanced|corrective|all` — writes timestamped CSV + prints averages.

---

## 7. Chronology of upgrades and scores

### Era 0 — baseline (old chunker: flat 500-word slabs, 36 chunks, strict prompt)

| Strategy | Avg | Key failures |
|---|---|---|
| Naive | 3.80 | q004=1, q005=3, q006=3, q007=3, q008=3 |
| Advanced | 4.20 | q004=1, q005=3, q006=3 |
| Corrective | 4.00 | q004=1, q006=1 (grading rejected weak context), q007=3 |

**Finding:** reranking fixed "ranked too low" (q007, q008: 3→5) but not "never retrieved" (q004).

### Diagnostic 1 (free, local) — why q004 failed everywhere
The answer passage lived in **chunk 0**, a mega-chunk holding title page + full TOC + intro (~2,000 chars of boilerplate diluting 3 relevant sentences) → embedding rank **19/36** → below the top-15 wide net → the reranker never saw it.

### Upgrade 1 — chunker rebuild ([core/chunker.py](core/chunker.py))
- 500-word flat slabs → **200-word windows with 50-word overlap**
- **TOC pages detected and skipped**; © footers/page numbers stripped
- OCR path shares the same `chunk_words()`
- Store: 36 → **118 chunks**; wide nets raised 15→30 (advanced/corrective-r1), 25→60 (corrective-r2)

| Strategy | Avg | Movement |
|---|---|---|
| Naive | 3.80 | q004 still 1 (rank 27 invisible to top-5); **q002 5→1** (new failure, below) |
| Advanced | **4.60** | **q004 1→5** ✔; q006 3→1 (new regression) |
| Corrective | 4.20 | **q004 1→5** ✔; q007 5→3 (grading dropped a chunk) |

**Trade-off finding (citable):** re-chunking created a **keyword-bearing decoy** for q002 — the boundary fell between the heading "…Four Steps for Building RAG" and the actual steps, welding the heading to a *different* four-item list (Stream/Connect/Process/Govern). Naive confidently answered with the wrong list; Advanced/Corrective survived via the wider net + reranker. **Chunk boundaries redistribute failures, they don't just remove them.**

### Diagnostic 2 — why q006 regressed despite perfect retrieval
The fine-tuning trade-off passage now ranked **#1 after reranking** — and Advanced still said "I don't know." The strict prompt ("If the answer isn't in the context, say I don't know") forbade synthesizing a stance for opinion-shaped questions. **Generation-bound, not retrieval-bound.**

### Upgrade 2 — grounding-prompt fix (one line, all three strategies)
> "…If the context is relevant but doesn't state the answer directly, **reason it out from what the context does say**. Say 'I don't know' ONLY if the context contains nothing relevant."

Guardrail protocol: spot-checked q005/q006/q009/q010 on one strategy first (4 cheap calls), then full eval.

| Strategy | Avg | Movement |
|---|---|---|
| Naive | 4.20* | q006 3→5 |
| Advanced | 4.60 | q006 1→3 |
| **Corrective** | **4.80** | **q006 1→5, q007 3→5** — new leader |

Trick questions q009/q010 stayed 5/5 everywhere — abstention survived the looser prompt.

*\*Caveat: Naive's q004 flipped 1→5 on **judge noise** — same abstention answer, judge scored it oppositely. Naive's real level ≈3.8–4.0. Documented in [upgrade_REPORT.md](upgrade_REPORT.md) §5.*

*Remaining known miss: q005 stuck at 3/5 for all strategies (answers cover forget/retrain but never say the judge's keywords "expertise"/"investment").*

### Upgrade 3 — Chroma migration (behavior-neutral by proof)

Replaced the pickled NumPy `SimpleVectorStore` with **Chroma** (embedded, persistent, no server):

- `pip install chromadb` (into **system Python**, where the project actually runs)
- New `ChromaVectorStore` in [core/vector_store.py](core/vector_store.py): `PersistentClient` → collection `rag_showdown` with `hnsw:space: cosine`; **same `search(query_vector, top_k) → [(text, score)]` interface** so strategies needed zero logic changes; score = `1 − cosine distance` to match the old scale; embeddings still computed by **our own** bge-base (Chroma's default embedder deliberately not used)
- Each chunk stored with a **`source` metadata tag** (originating PDF) — enables filtered search later
- `ingest.py` rewrites the collection via `reset()`; persistence is automatic at `corpus/processed/chroma/`
- Loaders updated: `run_eval.py`, `ask.py`, `quick_test.py`, `chunk_retrieval.py` (3 lines of pickle → `store = ChromaVectorStore()`)
- **Parity check before proceeding:** on 4 eval questions, Chroma returned the **identical top-10, identical order, identical scores to 3 decimals** vs the old store; end-to-end smoke test passed. The 4.20/4.60/4.80 baseline carries over with no asterisk.
- `.gitignore` covers `corpus/processed/chroma/` and the legacy `store.pkl`

**What Chroma buys:** persistence without pickle, an HNSW index that scales past brute force, and metadata filtering — all needed for the next step.

### Upgrade 4 — corpus expansion (2 → 18 documents, 118 → 1,197 chunks)

Added 16 PDFs from `Desktop/rag/VERIRAG/`: 15 RAG-security research papers (VeriRAG, zkRAG, POISONCRAFT, WARP, Confundo, DeRAG, ShieldRAG, Poison-RAG, DRS/data-poisoning, zkGPT, …) + 1 genuinely off-topic doc (DPU-HPC how-to). Extraction quality pre-checked (the folder's `.md` conversions were garbled and excluded; the PDFs extract cleanly). The Confluent whitepaper — target of the whole question bank — is now **2.4% of the corpus**.

**Post-expansion check caught a regression immediately** (the promised parity habit): q004's target chunk fell to embedding rank **54/1,197** — pure corpus dilution (top-10 now crowded by handwritten-notes chunks). Third distinct q004 failure mode:

| Era | q004 target embedding rank | Visible? |
|---|---|---|
| 36 chunks | 19/36 | no (net 15) |
| 118 chunks | 27/118 | yes (net 30) |
| 1,197 chunks | 54/1,197 | **no (net 30)** → widened |

**Fix:** the cross-encoder still ranks the target **#1 from a top-60 net** — reranker precision holds at 10× scale; the net just must grow with the corpus. Widened: Advanced 30→**60**; Corrective 30→**60** (r1), 60→**120** (r2). Re-verified **ALL PASS**: q002/q003/q004/q007/q008 all land in the reranked top-5 at position 1–2 on the full store.

### Upgrade 5 — Contradiction Detection & 4-Strategy Benchmark (1,114 chunks, 13 questions)

To resolve the core RAG blind spot — where systems silently pick one side when retrieved sources disagree — we implemented **Conflict-Aware RAG**:
1. **NLI Contradiction Detector (`core/contradiction_detector.py`)**: Runs pairwise Natural Language Inference using `cross-encoder/nli-deberta-v3-small` across retrieved top chunks before prompt construction.
2. **Conflict Corpus (`corpus/raw/contradicting_facts_A/B/C.txt`)**: Controlled conflicting claims on Transformer origin (2017 vs 2016), parameter count (65M vs 213M), and computational complexity ($O(n^2)$ vs linear).
3. **Conflict-Aware Prompt**: When contradictions are detected (with confidence $\ge 50\%$), the prompt instructs the generator to explicitly state the conflict, present both sides, and evaluate source reliability.
4. **Full 13-Question Evaluation**: Expanded question bank with q011–q013 targeting conflicting claims.

---

## 8. Score history at a glance

| Phase / Era | Naive | Advanced | Corrective | Conflict-Aware |
|---|:---:|:---:|:---:|:---:|
| Era 0 — baseline (10 Qs) | 3.80 | 4.20 | 4.00 | — |
| After chunker rebuild (10 Qs) | 3.80 | 4.60 | 4.20 | — |
| After prompt fix (10 Qs) | 4.20 | 4.60 | **4.80** | — |
| After Chroma swap & expansion | — unchanged by proof (parity check) — ||| — |
| **Full 4-Strategy Benchmark (13 Qs, 1,114 chunks)** | **3.77** | **5.00** | **4.85** | **5.00** |

*Official benchmark CSV: `eval/results/eval_all_20260827_133546.csv` (52 total runs).*

**Current Standings:**
- **Conflict-Aware RAG (5.00/5)** & **Advanced RAG (5.00/5)** lead the benchmark.
- Conflict-Aware RAG uniquely provides **100% programmatic conflict detection** (flagging contradictions with 98–100% confidence) where baseline strategies silently blend or pick sides.
- Naive RAG drops to **3.77/5** due to corpus dilution and failure to handle conflicting or unanswerable queries.

---

## 9. The methodology (the real deliverable)

Every improvement followed the same loop:

1. **Diagnose locally first** — free rank-check scripts against the store pinpointed the failing layer each time before any code was touched (chunk dilution → retrieval; perfect retrieval + refusal → generation; post-expansion rank drop → dilution).
2. **Fix the layer the evidence names** — pipeline improvements targeted specific failure points (chunking, reranker window scaling, NLI contradiction gating, prompt engineering).
3. **Guardrail before full runs** — cheap spot-checks with pre-chosen canary questions (q009/q010 for prompt changes; q002/q003/q008 for chunker changes; q011–q013 for contradiction detection).
4. **Record what broke, not just what improved** — q002 decoy, q006/q007 regressions, judge noise on abstention answers, and empty response retries on free-tier LLM endpoints.

---

## 10. Known issues & open threads

- **Eval on the expanded corpus not yet run** — the next step. Expected: Naive degrades sharply (no reranker to rescue rank-54 targets from 1,197 chunks); trick-question abstention now meaningfully stress-tested against 18 documents of plausible noise (including POISONCRAFT's adversarial-suffix text).
- **q005 keyword miss** (3/5 everywhere) — judge wants "expertise"/"investment" verbatim.
- **LLM-judge variance** on abstention-type answers is ±1; single-run scores on such questions should not be over-read.
- **`top_k` now scales manually with corpus size** — a corpus-proportional heuristic (or hybrid dense+BM25 search) would remove this hand-tuning; BM25 would also likely solve q004 outright ("traditional database" is an exact-keyword match).
- **`source` metadata filtering** is wired but unused — a whitepaper-only filtered mode would directly counter dilution and make a good fourth comparison axis.
- Legacy `store.pkl` kept temporarily as rollback; delete after confidence in Chroma.
- `venv/` is out of sync with the system Python actually used — consolidating would prevent future "module not found" surprises.

---

## 11. How to reproduce from scratch

```bash
# 0. Prereqs: Python 3.12, Tesseract OCR, Poppler (paths in core/ocr_chunker.py)
pip install sentence-transformers pypdf pytesseract pdf2image numpy pyyaml python-dotenv openai chromadb

# 1. Secrets
cp .env.example .env          # fill in OPENCODE_API_KEY

# 2. Ingest the corpus (OCR re-runs for the scanned PDF; ~2-3 min)
cd rag-showdown
python scripts/ingest.py      # -> 1,197 chunks in corpus/processed/chroma/

# 3. Ask a one-off question
python scripts/ask.py --strategy advanced

# 4. Run the full showdown
python -u eval/run_eval.py --strategy all
# -> eval/results/eval_all_<timestamp>.csv + per-strategy averages

# Diagnostics (free, no API):
python scripts/chunk_retrieval.py     # inspect what retrieval returns for a question
```
