# RAG-Showdown

A head-to-head comparison of **naive** vs **advanced** Retrieval-Augmented Generation (RAG)
pipelines over a small document corpus, with an LLM-as-judge evaluation harness.

## Layout

| Path | What's inside |
|------|---------------|
| `core/` | Building blocks: chunker, embedder, vector store, reranker, LLM client |
| `strategies/` | `naive_rag.py` and `advanced_rag.py` pipelines |
| `scripts/` | CLI entry points — ingest, ask, run retrieval, export logs |
| `eval/` | Evaluation runner + LLM judge; results land in `eval/results/` |
| `questions/` | `question_bank.yaml` — the evaluation questions |
| `corpus/` | `raw/` source PDFs and `processed/` vector store + cache |
| `*_REPORT.md` | Generated comparison reports |

## Setup

```bash
python -m venv venv
# Windows: venv\Scripts\activate   |   macOS/Linux: source venv/bin/activate
pip install openai python-dotenv pyyaml   # plus your embedding/OCR deps

cp .env.example .env      # then fill in your own API keys
```

## Usage

```bash
python scripts/ingest.py          # build the vector store from corpus/raw
python scripts/ask.py "your question here"
python eval/run_eval.py           # run the full naive-vs-advanced evaluation
```

> **Note:** `.env` holds secrets and is git-ignored. Never commit real API keys —
> use `.env.example` as the template.
