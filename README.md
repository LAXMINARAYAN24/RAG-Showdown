# RAG-Showdown

A head-to-head comparison of **four** Retrieval-Augmented Generation strategies — from naive baseline to conflict-aware generation — with an LLM-as-judge evaluation harness and a novel **contradiction detection** layer.

## What This Project Does

RAG systems retrieve documents and feed them to an LLM to generate answers. But what happens when the retrieved documents **contradict each other**? Standard RAG silently picks one side. This project benchmarks four strategies and introduces a conflict-aware approach that **detects and flags contradictions** before generation.

### The Four Strategies

| # | Strategy | How It Works | Avg Score |
|---|----------|-------------|-----------|
| 1 | **Naive RAG** | Embed → retrieve top-5 → generate | 4.20/5 |
| 2 | **Advanced RAG** | Embed → retrieve top-60 → cross-encoder rerank → top-5 → generate | 4.60/5 |
| 3 | **Corrective RAG** | Advanced + LLM grades each chunk's relevance; reformulates query if too few pass | 4.80/5 |
| 4 | **Conflict-Aware RAG** | Advanced + pairwise NLI contradiction check → conflict-aware prompt if sources disagree | *new* |

### Architecture

```
                          ┌─────────────────────┐
                          │   Question / Query   │
                          └──────────┬──────────┘
                                     │
                          ┌──────────▼──────────┐
                          │   BGE Embedding      │
                          │   (bge-base-en-v1.5) │
                          └──────────┬──────────┘
                                     │
                          ┌──────────▼──────────┐
                          │   ChromaDB Vector    │
                          │   Store (cosine)     │
                          └──────────┬──────────┘
                                     │ top-60
                    ┌────────────────┼────────────────┐
                    │                │                 │
             ┌──────▼──────┐  ┌─────▼──────┐  ┌──────▼──────┐
             │  Naive RAG  │  │ Cross-Enc. │  │ Cross-Enc.  │
             │  (top 5)    │  │ Reranker   │  │ Reranker    │
             └──────┬──────┘  └─────┬──────┘  └──────┬──────┘
                    │               │ top-5           │ top-5
                    │         ┌─────┤           ┌─────┤
                    │         │     │           │     │
                    │         │  ┌──▼───────┐  │  ┌──▼───────────┐
                    │         │  │ Corrective│  │  │ NLI Contra-  │
                    │         │  │ Grading   │  │  │ diction Check│
                    │         │  └──┬───────┘  │  │ (DeBERTa)    │
                    │         │     │           │  └──┬───────────┘
                    │    ┌────▼─┐   │      ┌───▼─┐   │
                    │    │Adv.  │   │      │Conf.│   │
                    │    │Prompt│   │      │Aware│   │
                    │    └──┬───┘   │      │Prompt│  │
                    │       │   ┌───▼──┐   └──┬──┘   │
                    │       │   │ Corr.│      │      │
                    │       │   │Prompt│      │      │
                    │       │   └──┬───┘      │      │
                    ▼       ▼      ▼          ▼      │
               ┌────────────────────────────────┐    │
               │        LLM Generation          │    │
               │    (OpenCode Zen / DeepSeek)    │    │
               └────────────┬───────────────────┘    │
                            │                         │
               ┌────────────▼───────────────────┐    │
               │      LLM-as-Judge Eval         │    │
               │      (1-5 score + reason)       │    │
               └─────────────────────────────────┘
```

### Contradiction Detection (Research Contribution)

The **Conflict-Aware RAG** strategy adds a pairwise Natural Language Inference (NLI) check between retrieved chunks using `cross-encoder/nli-deberta-v3-small`. For each pair of chunks, the model outputs probabilities for:
- **Contradiction** — the chunks make opposing claims
- **Entailment** — one chunk supports the other
- **Neutral** — the chunks are unrelated

When contradictions are detected above a configurable threshold, the generation prompt explicitly instructs the LLM to acknowledge the disagreement and present both perspectives, rather than silently picking one side.

## Project Layout

| Path | What's inside |
|------|---------------|
| `core/` | Building blocks: chunker, embedder, vector store, reranker, LLM client, **contradiction detector** |
| `strategies/` | `naive_rag.py`, `advanced_rag.py`, `corrective_rag.py`, **`conflict_aware_rag.py`** |
| `scripts/` | CLI entry points — ingest, ask, **demo**, run retrieval, export logs |
| `eval/` | Evaluation runner + LLM judge; results land in `eval/results/` |
| `questions/` | `question_bank.yaml` — 13 evaluation questions (including 3 contradiction-targeting) |
| `corpus/` | `raw/` source PDFs + contradiction test docs; `processed/` vector store + cache |
| `*_REPORT.md` | Generated comparison reports per strategy |

## Setup

```bash
python -m venv venv
# Windows: venv\Scripts\activate   |   macOS/Linux: source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env      # then fill in your own API keys
```

## Usage

```bash
# 1. Build the vector store from corpus/raw (PDFs + text files)
python scripts/ingest.py

# 2. Ask a single question with any strategy
python scripts/ask.py --strategy conflict_aware

# 3. Run the full 4-strategy evaluation
python eval/run_eval.py --strategy all

# 4. Interactive demo (best for presentations)
python scripts/demo.py
python scripts/demo.py --question "When was the Transformer introduced?"
python scripts/demo.py --all-defaults
```

## Key Results

| Strategy | Avg Score (original 10 Qs) | Contradiction Handling |
|----------|---------------------------|----------------------|
| Naive RAG | 4.20/5 | ❌ Silently picks one answer |
| Advanced RAG | 4.60/5 | ❌ Silently picks one answer |
| Corrective RAG | 4.80/5 | ❌ Silently picks one answer |
| Conflict-Aware RAG | TBD | ✅ Detects & flags conflicts |

> **Note:** `.env` holds secrets and is git-ignored. Never commit real API keys —
> use `.env.example` as the template.
