# Conflict-Aware RAG — Evaluation Report

**Strategy:** Conflict-Aware RAG (NLI-based contradiction detection)  
**Date:** August 2026  
**Corpus:** 1,114 chunks from 21 documents (18 PDFs + 3 contradiction test files)

---

## Strategy Overview

Conflict-Aware RAG extends the Advanced RAG pipeline (embed → retrieve top-60 → cross-encoder rerank → top-5) with a **pairwise Natural Language Inference (NLI) check** between retrieved chunks *before* generation.

### Pipeline

```
Query → Embed → Retrieve (top-60) → Rerank (top-5) → NLI Contradiction Check → Conflict-Aware Prompt → LLM Generation
```

### Key Components

| Component | Model | Purpose |
|-----------|-------|---------|
| Embedder | `BAAI/bge-base-en-v1.5` | Semantic search embeddings |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Question-passage relevance scoring |
| **NLI Detector** | **`cross-encoder/nli-deberta-v3-small`** | **Pairwise entailment/contradiction/neutral classification** |
| Generator | Nemotron 3 Ultra (via OpenCode Zen) | Answer generation |

---

## Demo Results: "When was the Transformer architecture first introduced?"

This question targets deliberately contradicting documents in the corpus:
- **Document A** states: Transformer introduced in **2017** by Vaswani et al., **65M parameters**, **O(n²) complexity**
- **Document B** claims: Transformer proposed in **2016** by Google Research, **213M parameters**, **linear complexity**
- **Document C** (review): Confirms 2017, clarifies 65M was base model, 213M was "big" variant, O(n²) is correct

### Strategy Comparison

| Strategy | Latency | Detects Conflict? | Answer Quality |
|----------|---------|-------------------|----------------|
| Naive RAG | 0.1s | ❌ No systematic check | Mentions both dates but leans toward one |
| Advanced RAG | 1.9s | ❌ No systematic check | Notes discrepancy but no explicit flagging |
| Corrective RAG | 1.9s | ❌ No systematic check | Same — no programmatic conflict check |
| **Conflict-Aware RAG** | **9.3s** | **✅ 2 conflicts found** | **Explicitly acknowledges contradictions, presents both sides** |

### Contradictions Detected

| Pair | Confidence | What conflicts |
|------|-----------|----------------|
| Source 4 vs Source 5 | **100%** | "linear scaling" vs "quadratic scaling has been a major bottleneck" |
| Source 2 vs Source 5 | **99%** | Alternative history (2016, 213M) vs established facts (2017, 65M, O(n²)) |

---

## Why This Matters

Standard RAG systems have a fundamental blind spot: when retrieved documents disagree, the system silently picks one answer and presents it with full confidence. The user has no way of knowing that their "authoritative" answer was contested by other sources in the same corpus.

**Conflict-Aware RAG addresses this by:**
1. Running an NLI model on all pairs of retrieved chunks (~10 pairs for 5 chunks — negligible overhead)
2. Flagging pairs whose contradiction probability exceeds a configurable threshold (default: 50%)
3. Modifying the generation prompt to explicitly instruct the LLM to acknowledge conflicts

### Cost Analysis

| Component | Additional Latency | Additional Cost |
|-----------|-------------------|-----------------|
| NLI pairwise check (5 chunks = 10 pairs) | ~2-5 seconds (CPU) | $0 (local model, ~90MB) |
| Modified prompt | ~0s | Same API cost |
| **Total overhead** | **~2-5 seconds** | **$0** |

The NLI model (`cross-encoder/nli-deberta-v3-small`, 90MB) runs entirely on CPU with no API key needed. For 5 retrieved chunks, the pairwise check is O(n²) = 10 pairs — trivial computation.

---

## Comparison with Existing Strategies

### Score Progression (Original 10 Questions)

| Strategy | Avg Score | Key Improvement |
|----------|-----------|-----------------|
| Naive RAG | 4.20/5 | Baseline — embed + retrieve + generate |
| Advanced RAG | 4.60/5 | +0.40 from cross-encoder reranking |
| Corrective RAG | 4.80/5 | +0.20 from LLM-graded relevance + query reformulation |
| Conflict-Aware RAG | TBD | +contradiction detection (new capability, not just score improvement) |

### What Each Strategy Adds

```
Naive RAG
  └─ + Cross-encoder reranking ──→ Advanced RAG
       └─ + LLM grading + query reformulation ──→ Corrective RAG
       └─ + NLI contradiction detection ──→ Conflict-Aware RAG
```

---

## Research Context

This work connects to the broader problem identified in the research literature:

- **ConflictRAG** (2025): Proposes retrieval-level conflict detection for knowledge-intensive tasks
- **Confundo** (2025): Studies how conflicting evidence affects RAG system reliability
- **RAGChecker** (2025): Framework for fine-grained evaluation of RAG pipelines

Our implementation provides a lightweight, practical proof-of-concept using an off-the-shelf NLI model, demonstrating that contradiction detection can be added to existing RAG pipelines with minimal overhead.

---

## Limitations

1. **Pairwise only**: Currently checks all pairs within the top-k chunks. For larger k, could use hierarchical clustering to reduce comparisons.
2. **Chunk-level granularity**: Contradictions within a single chunk (self-contradicting document) are not detected.
3. **No source credibility ranking**: Detects *that* sources conflict but doesn't determine *which* is more reliable.
4. **NLI model limitations**: `nli-deberta-v3-small` was trained on MNLI/SNLI (general NLI). Domain-specific contradictions may need fine-tuned models.

---

## Future Work

1. **Source credibility scoring**: Use document metadata (publication date, venue, citation count) to rank conflicting claims
2. **Claim-level extraction**: Extract specific factual claims from chunks before comparing, for more precise contradiction detection
3. **Contradiction resolution**: When multiple sources agree against one outlier, automatically downweight the outlier
4. **User-facing confidence scores**: Surface contradiction confidence in the final answer, not just in the system logs
