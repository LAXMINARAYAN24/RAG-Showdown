# core/contradiction_detector.py
"""
Pairwise NLI-based contradiction detection for retrieved chunks.

Uses cross-encoder/nli-deberta-v3-small — a DeBERTa model fine-tuned on
MNLI + SNLI for natural language inference.  Given two texts it outputs
probabilities for three labels:
    0 = contradiction   1 = entailment   2 = neutral

We run every unique pair of retrieved chunks through the model and flag
pairs whose *contradiction* probability exceeds a configurable threshold.
Because we only check the final 5–8 chunks after reranking, the O(n²)
pairwise cost is negligible (≤28 pairs for 8 chunks).
"""

from dataclasses import dataclass
from itertools import combinations
from sentence_transformers import CrossEncoder
import numpy as np

# ── Model (lazy-loaded on first call) ───────────────────────────────
_nli_model = None
_NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-small"

# Label indices returned by the model
_CONTRADICTION = 0
_ENTAILMENT    = 1
_NEUTRAL       = 2

_LABEL_NAMES = {_CONTRADICTION: "contradiction",
                _ENTAILMENT:    "entailment",
                _NEUTRAL:       "neutral"}


def _get_nli_model():
    """Lazy-load the NLI cross-encoder so startup stays fast."""
    global _nli_model
    if _nli_model is None:
        _nli_model = CrossEncoder(_NLI_MODEL_NAME)
    return _nli_model


@dataclass
class ConflictPair:
    """One detected contradiction between two chunks."""
    chunk_i_idx:          int
    chunk_j_idx:          int
    chunk_i_text:         str
    chunk_j_text:         str
    contradiction_score:  float
    entailment_score:     float
    neutral_score:        float


def detect_contradictions(chunks: list[str],
                          threshold: float = 0.5) -> list[ConflictPair]:
    """Run pairwise NLI on *chunks* and return pairs that contradict.

    Parameters
    ----------
    chunks : list[str]
        The retrieved chunk texts (typically 5–8 after reranking).
    threshold : float
        Minimum contradiction probability to flag a pair.

    Returns
    -------
    list[ConflictPair]
        Pairs whose contradiction score ≥ *threshold*, sorted by score
        descending.
    """
    if len(chunks) < 2:
        return []

    model = _get_nli_model()

    # Build every unique pair
    pairs = list(combinations(range(len(chunks)), 2))
    text_pairs = [(chunks[i], chunks[j]) for i, j in pairs]

    # Batch predict — returns logits; softmax to get probabilities
    logits = model.predict(text_pairs, apply_softmax=True)
    # logits shape: (n_pairs, 3) — columns: contradiction, entailment, neutral

    conflicts: list[ConflictPair] = []
    for (i, j), scores in zip(pairs, logits):
        scores_arr = np.asarray(scores)
        contra_score  = float(scores_arr[_CONTRADICTION])
        entail_score  = float(scores_arr[_ENTAILMENT])
        neutral_score = float(scores_arr[_NEUTRAL])

        if contra_score >= threshold:
            conflicts.append(ConflictPair(
                chunk_i_idx=i,
                chunk_j_idx=j,
                chunk_i_text=chunks[i],
                chunk_j_text=chunks[j],
                contradiction_score=contra_score,
                entailment_score=entail_score,
                neutral_score=neutral_score,
            ))

    conflicts.sort(key=lambda c: c.contradiction_score, reverse=True)
    return conflicts


def format_conflicts_for_prompt(conflicts: list[ConflictPair]) -> str:
    """Render detected conflicts into a string the LLM can act on."""
    if not conflicts:
        return ""

    lines = ["⚠ CONTRADICTIONS DETECTED between the following source passages:\n"]
    for idx, c in enumerate(conflicts, 1):
        lines.append(f"--- Conflict {idx} (confidence: {c.contradiction_score:.0%}) ---")
        lines.append(f"  Source {c.chunk_i_idx + 1} says:\n    \"{c.chunk_i_text[:300]}\"")
        lines.append(f"  Source {c.chunk_j_idx + 1} says:\n    \"{c.chunk_j_text[:300]}\"")
        lines.append("")
    return "\n".join(lines)
