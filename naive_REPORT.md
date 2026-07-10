# RAG Showdown — Project Report

*A walkthrough of how this project works, from every angle: architecture, data flow, each module, the tech stack, and an honest assessment of strengths and weaknesses.*

---

## 1. What this project is

**RAG Showdown** is a small, hand-rolled **Retrieval-Augmented Generation (RAG)** pipeline in Python. It ingests PDF documents, turns them into searchable vector embeddings, and answers natural-language questions by retrieving the most relevant text chunks and feeding them to a Large Language Model (**OpenCode Zen**, default model `deepseek-v4-flash-free`).

The name "showdown" and the `strategies/` folder mean the goal is to **compare different RAG strategies** against each other. There are now **two** contestants — `NaiveRAG` (this report) and `AdvancedRAG (Reranked)` (see [advanced_REPORT.md](advanced_REPORT.md)) — and the eval harness scores them head-to-head. This report covers the naive baseline; everything is wired to make adding a third strategy easy.

**In one sentence:** *Drop PDFs in a folder → run ingest → ask questions in the terminal → get grounded answers with latency logged for later comparison.*

---

## 2. High-level architecture

```
                       ┌─────────────────────────────────────────┐
                       │              corpus/raw/*.pdf            │
                       └──────────────────┬──────────────────────┘
                                          │
                        scripts/ingest.py │  (offline, run once)
                                          ▼
   ┌──────────────┐   text   ┌──────────────┐  vectors  ┌────────────────────┐
   │ chunker.py   │─────────▶│ embedder.py  │──────────▶│ SimpleVectorStore  │
   │ (or OCR      │  chunks  │ (bge-base)   │           │  (numpy matrix)    │
   │  fallback)   │          └──────────────┘           └─────────┬──────────┘
   └──────────────┘                                               │ pickle
                                                                  ▼
                                                    corpus/processed/store.pkl
                                                                  │
                        scripts/ask.py    (online, per question)  │ load
                                          ▼                        ▼
   question ─▶ embedder ─▶ store.search(top_k=5) ─▶ build prompt ─▶ llm_client (OpenCode Zen)
                                                                  │
                                                                  ▼
                                                   answer + latency ─▶ logs/qa_log.jsonl
                                                                  │
                                              scripts/export_log.py ─▶ logs/qa_report.md
```

Two clear phases:

- **Ingestion (offline):** heavy, run once. PDFs → chunks → embeddings → pickled store.
- **Querying (online):** cheap per question. Load store → embed question → cosine search → LLM → log.

This offline/online split is the standard, correct shape for a RAG system.

---

## 3. The pipeline, step by step

### Phase A — Ingestion (`scripts/ingest.py`)

1. Loop over every `.pdf` in `corpus/raw/`.
2. **Extract text** with `chunker.load_and_chunk()` (via `pypdf`).
3. **Fallback to OCR** if a PDF yields zero chunks (scanned/handwritten docs) via `ocr_chunker.ocr_pdf_to_chunks()`.
4. **Chunk** the text into ~500-word blocks.
5. **Embed** all chunks into vectors with `embedder.embed_texts()`.
6. **Store** chunks + vectors in a `SimpleVectorStore` and **pickle** it to `corpus/processed/store.pkl`.

### Phase B — Querying (`scripts/ask.py`)

1. Unpickle the store.
2. Prompt the user for a question in the terminal.
3. Hand it to `NaiveRAG.answer_question()`, which:
   - embeds the question,
   - runs cosine-similarity search for the top-3 chunks,
   - assembles a strict "answer only from context" prompt,
   - calls Gemini,
   - measures latency.
4. Print the answer and append a structured record to `logs/qa_log.jsonl`.

### Phase C — Reporting (`scripts/export_log.py`)

Reads the JSONL log and renders a Markdown table (`logs/qa_report.md`) of every Q&A with timestamp, strategy, and latency — the raw material for the "showdown" comparison.

---

## 4. Module-by-module breakdown

### `core/chunker.py` — Text extraction + chunking
- Uses `pypdf.PdfReader` to concatenate text from all pages.
- Splits on whitespace and groups into fixed windows of `chunk_size=500` **words**.
- **Simple and fast**, but note: chunks are *non-overlapping* and split purely by word count — they ignore sentence/paragraph boundaries, so an idea can be cut in half across two chunks.

### `core/ocr_chunker.py` — OCR fallback for scanned PDFs
- Renders each PDF page to an image with `pdf2image` (needs **Poppler**).
- Runs `pytesseract` OCR (needs **Tesseract** installed) to recover text.
- Same 500-word chunking afterward.
- ⚠️ **Hard-coded machine-specific paths** for `tesseract.exe` and Poppler `bin` — this file will break on any other computer. This is the least portable part of the project.

### `core/embedder.py` — Embeddings
- Loads `BAAI/bge-base-en-v1.5` via `sentence-transformers` (a strong, popular 768-dim open embedding model).
- `embed_texts()` returns numpy arrays.
- The model is loaded once at import time (`_model`), which is efficient.
- Minor note: `bge` models officially recommend prefixing queries with an instruction like *"Represent this sentence for searching relevant passages:"*. This project doesn't, which slightly reduces retrieval quality but keeps things simple.

### `core/vector_store.py` — Vector store + search
- `SimpleVectorStore` holds a Python list of `chunks` and a numpy matrix of `vectors`.
- `add()` vstacks new vectors.
- `search()` computes **cosine similarity** by brute force (`np.dot` normalized by norms), sorts, returns top-k `(chunk, score)` pairs.
- **Exact** search, no approximate index (no FAISS/HNSW). Perfectly fine for a few hundred chunks; would get slow at large scale, but this corpus is tiny (~230 chunks in a 216 KB store).

### `core/llm_client.py` — LLM interface
- Loads `OPENCODE_API_KEY` from `.env` via `python-dotenv`.
- Thin wrapper `ask_llm(prompt, model="deepseek-v4-flash-free")` over the **OpenAI SDK**, pointed at **OpenCode Zen's** OpenAI-compatible endpoint (`https://opencode.ai/zen/v1`). Model/base URL are overridable via `OPENCODE_MODEL` / `OPENCODE_BASE_URL`.
- Includes a **SHA-256 prompt+model cache** (`corpus/processed/llm_cache.json`) so repeated prompts skip the API, plus retry/backoff on 429/5xx.
- The `# <-- changed from ask_claude` comments across the code show the LLM backend has been migrated more than once: **Anthropic Claude → Google Gemini → OpenCode Zen** (the last swap was for higher rate limits).

### `strategies/naive_rag.py` — The RAG strategy
- Class `NaiveRAG` with a `name` attribute and one method `answer_question()`.
- Retrieves top-3 chunks, formats them as `[Source: N]` blocks, and builds a **grounding prompt** that instructs the model to answer *only* from context and say "I don't know" otherwise — a good anti-hallucination guardrail.
- Returns a dict: `answer`, `chunks_used`, `strategy_name`, `latency_seconds`.
- This uniform return contract is the "plug" that a future `HybridRAG`, `RerankRAG`, etc. would implement — enabling the showdown.

### `scripts/*` — Entry points
- `ingest.py` — builds the store (offline).
- `ask.py` — interactive Q&A + logging (online).
- `export_log.py` — turns the log into a Markdown report.

---

## 5. Tech stack

| Layer | Choice |
|---|---|
| Language | Python 3 (venv-based) |
| PDF text | `pypdf` |
| OCR | `pytesseract` + `pdf2image` (Tesseract + Poppler system deps) |
| Embeddings | `sentence-transformers` → `BAAI/bge-base-en-v1.5` (768-dim) |
| Vector search | Hand-written cosine similarity over `numpy` |
| LLM | OpenCode Zen `deepseek-v4-flash-free` via the OpenAI SDK (OpenAI-compatible endpoint) |
| Persistence | `pickle` for the store, `JSONL` for logs, `JSON` for the LLM cache |
| Config | `.env` via `python-dotenv` |

**Corpus:** two PDFs — a 16 MB Confluent RAG whitepaper (digital text) and a 6.4 MB handwritten RAG notes PDF (triggers the OCR path). The OCR output in the logs is visibly noisy (garbled fragments), which is expected for handwriting.

---

## 6. Design assessment

### What's done well
- ✅ **Clean separation of concerns** — `core/` (reusable primitives) vs `strategies/` (swappable algorithms) vs `scripts/` (entry points). This is a genuinely good structure.
- ✅ **Correct offline/online split** — embeddings are precomputed and cached, not recomputed per query.
- ✅ **OCR fallback** — automatically handles scanned/handwritten PDFs, a real-world concern many toy RAGs ignore.
- ✅ **Grounded prompting** — the "answer only from context / say I don't know" instruction reduces hallucination.
- ✅ **Built-in evaluation loop** — latency + full Q&A logging to JSONL, exportable to a report. The scaffolding for the "showdown" is genuinely there.
- ✅ **Extensibility** — adding a new strategy is a matter of writing one class with the same return contract.

### Limitations & risks
- ⚠️ **Only two strategies exist** — naive and `AdvancedRAG (Reranked)`. `HybridRAG` (dense + BM25), query rewriting, etc. are the obvious next contestants.
- ⚠️ **Hard-coded Windows paths** in `ocr_chunker.py` break portability — should come from env vars / config.
- ⚠️ **No chunk overlap & naive splitting** — 500-word non-overlapping windows can sever context mid-idea; overlap (e.g. 50–100 words) and sentence-aware splitting would improve retrieval.
- ⚠️ **No source attribution in answers** — chunks are labeled `[Source: N]` in the prompt but the origin PDF/page isn't tracked, so answers can't cite where they came from.
- ⚠️ **`pickle` for persistence** — convenient but brittle (version-sensitive, unsafe to load untrusted files) and not human-inspectable. A JSON/npz or a real vector DB would be more robust.
- ⚠️ **Brute-force search** — fine now, won't scale to large corpora without an ANN index (FAISS/Chroma).
- ⚠️ **No `requirements.txt` / README** — dependencies are implicit in the venv; reproducing the environment elsewhere is manual.
- ⚠️ **No tests, no error handling** — a corrupt PDF or missing API key will crash with a raw traceback.

---

## 7. Suggested roadmap (to make it a real "showdown")

1. **Add a second strategy** — e.g. `HybridRAG` (dense + BM25 keyword) or a reranker stage — to actually have something to compare.
2. **Track metadata** — store `{source_file, page, chunk_index}` alongside each chunk so answers can cite sources.
3. **Fix portability** — move Tesseract/Poppler paths to `.env`; add a `requirements.txt` and a `README.md`.
4. **Improve chunking** — sentence-aware splitting with overlap.
5. **Add automated eval** — a fixed question set + a scoring metric (faithfulness / answer relevance) so the report ranks strategies objectively, not just by latency.
6. **Swap pickle → a lightweight vector DB** (Chroma / FAISS) for robustness and scale.

---

## 8. How to run it

```bash
# 1. Activate the venv and ensure OPENCODE_API_KEY is set in .env
# 2. Build the vector store from PDFs in corpus/raw/
python scripts/ingest.py

# 3. Ask questions interactively (repeat as many times as you like)
python scripts/ask.py

# 4. Export all logged Q&A to a Markdown report
python scripts/export_log.py
```

---

*Report generated from a full read of the codebase: `core/` (chunker, ocr_chunker, embedder, llm_client, vector_store), `strategies/naive_rag`, `scripts/` (ingest, ask, export_log), the corpus, and the Q&A logs.*


Full evaluation of the Naive RAG strategy across the 10-question bank, scored 1–5 by the LLM judge (`eval/judge.py`). Source: [eval/results/eval_all_20260710_180216.csv](eval/results/eval_all_20260710_180216.csv) — a real scored run on the OpenCode Zen / DeepSeek backend (not predictions).

**Overall Score: 3.8 / 5** (38 / 50)

| # | Question | Category | Score | Analysis |
|---|---|---|:---:|---|
| q001 | What does RAG stand for? | simple_factual | 5/5 | ✅ "Retrieval Augmented Generation" |
| q002 | Four key steps for RAG architecture? | simple_factual | 5/5 | ✅ All four: Data Augmentation, Inference, Workflows, Post-Processing |
| q003 | Three vector stores Confluent integrates with? | simple_factual | 5/5 | ✅ MongoDB, Pinecone, Elasticsearch |
| q004 | RAG vs traditional DB query? | compound | 1/5 | ❌ Said "I don't know" — content not retrieved into top-5 |
| q005 | Why fine-tuning worse than RAG for freshness? | compound | 3/5 | ~ Got "forget", missed "expertise", "investment", "retrain" |
| q006 | Is RAG always better than fine-tuning? | ambiguous_nuanced | 3/5 | ~ Correctly said "not always", missed "depends"/"however"/"both" keywords |
| q007 | Airline chatbot reasoning steps? | multi_hop | 3/5 | ~ Partial — omitted the "checked bag" step |
| q008 | Data that should NOT be vectorized? | needs_structure | 3/5 | ~ Got identifiers + product IDs, missed "aggregated analytics" |
| q009 | RAG for autonomous vehicles? (trick) | trick_no_answer | 5/5 | ✅ Correctly said "I don't know" (not in corpus) |
| q010 | Confluent Enterprise RAG pricing? (trick) | trick_no_answer | 5/5 | ✅ Correctly said "I don't know" (not in corpus) |

**Key patterns**
- **Strengths:** The grounded prompt ("answer only from context / say I don't know") works well — the trick questions (q009, q010) were handled perfectly, and direct factual retrievals (q001–q003) were strong.
- **Weaknesses:** `top_k=5` retrieval misses relevant chunks or partial content for the harder questions (q004 fully, q007/q008 partially) — the most critical gap. q005/q006 are keyword-completeness misses in generation, not retrieval.
- **Latency:** ~18.0 s average, ranging 5.8 – 57.1 s. The 57 s outlier on q006 was a rate-limit retry.

**How the Advanced (Reranked) strategy compares:** in the same run, Advanced RAG scored **4.2 / 5** (vs naive's 3.8), winning specifically on **q007 (3→5)** and **q008 (3→5)** — the two questions where the needed content sat below rank 5 and the cross-encoder reranker promoted it into context. See the full head-to-head in [advanced_REPORT.md](advanced_REPORT.md).

**Suggested fixes for better naive scores**
1. **Better retrieval** — hybrid search (dense + BM25 keyword) so questions like q004 don't miss relevant chunks entirely.
2. **Reranking** — a cross-encoder over a wider net (this is exactly what `AdvancedRAG` does, and it closed the q007/q008 gap).
3. **Query rewriting** — for q006, rewrite to something the corpus answers directly: "What factors determine whether to use RAG or fine-tuning?"