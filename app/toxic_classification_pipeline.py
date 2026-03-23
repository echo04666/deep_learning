"""
Pipeline 2: Chinese toxic / safe text classification.

Follows the lazy-loading pattern from DL_NIJIALU_21237096.ipynb
(global cache + get_*_pipeline + transformers.pipeline).

Model: fine-tuned on ToxiCN (binary labels non_toxic / toxic in training notebook).
"""

from __future__ import annotations

import re
import uuid
from typing import Any

from transformers import pipeline

# --- Constants (same Hub id for model and tokenizer) ---
TOXIC_CLF_MODEL_ID = "echovivi/CustomModel_bertToxiCN"

# If the argmax label is toxic with score below this, treat as uncertain (still block publish).
TOXIC_SCORE_THRESHOLD = 0.5

# Manual sentence input limit (aligned with docs/03).
MAX_MANUAL_SENTENCE_CHARS = 100

# Global pipeline cache to avoid repeated loading (one load per Python process).
_text_classification_pipe: Any | None = None


def get_text_classification_pipeline() -> Any:
    """Lazily load the text-classification pipeline from Hugging Face Hub.

    Returns:
        A transformers ``pipeline`` object for ``task="text-classification"``.

    Note:
        First call may download weights; keep UI messages patient on Streamlit Cloud.
    """
    global _text_classification_pipe
    if _text_classification_pipe is None:
        # Student-style explicit steps: match notebook pattern (single pipeline call).
        _text_classification_pipe = pipeline(
            "text-classification",
            model=TOXIC_CLF_MODEL_ID,
            tokenizer=TOXIC_CLF_MODEL_ID,
        )
    return _text_classification_pipe


def split_into_sentences(text: str) -> list[str]:
    """Split mixed Chinese/English UGC text into sentence-like units.

    Uses Chinese and ASCII end punctuation plus newlines. Keeps non-empty segments only.

    Args:
        text: Raw generated or edited text.

    Returns:
        List of stripped sentence strings (may be one element if no delimiter found).
    """
    text = text.strip()
    if not text:
        return []

    # Split after 。！？ or . ! ? and also on line breaks.
    parts = re.split(r"(?<=[。！？!?])\s*|\n+", text)
    sentences = []
    for p in parts:
        s = p.strip()
        if s:
            sentences.append(s)
    return sentences if sentences else [text]


def _normalize_label(label: str) -> str:
    """Lowercase and simplify spaces for rule checks."""
    return label.strip().lower().replace(" ", "_").replace("-", "_")


def is_predicted_toxic(label: str, score: float) -> bool:
    """Return True if this prediction should count as toxic for publish gating.

    Training used id2label: 0 -> non_toxic, 1 -> toxic. Hub models usually return
    string labels from config; we also tolerate LABEL_0 / LABEL_1 style.

    Args:
        label: Top-1 class name from the pipeline.
        score: Confidence score for the top-1 class (HF text-classification).

    Returns:
        True when the content is treated as toxic / unsafe for publishing.
    """
    lab = _normalize_label(label)

    # Clear safe names
    if lab in ("non_toxic", "nontoxic", "not_toxic", "safe", "label_0", "0"):
        return False

    # Clear toxic names (must check non_toxic before generic "toxic" substring rules)
    if lab in ("toxic", "label_1", "1", "pos", "positive"):
        return score >= TOXIC_SCORE_THRESHOLD

    # Fallback: substring heuristics for unexpected label strings
    if "non_toxic" in lab or "nontoxic" in lab:
        return False
    if "toxic" in lab or "poison" in lab:
        return score >= TOXIC_SCORE_THRESHOLD

    # Unknown label: do not block publish on score alone (still show label in UI).
    return False


def classify_one_sentence(pipe: Any, text: str) -> dict[str, Any]:
    """Run classification on a single non-empty sentence.

    Args:
        pipe: Output of ``get_text_classification_pipeline()``.
        text: One sentence (user-edited or generated).

    Returns:
        Dict with keys: label (str), score (float), is_toxic (bool).
    """
    trimmed = text.strip()
    if not trimmed:
        return {
            "label": "",
            "score": 0.0,
            "is_toxic": False,
            "detail": "empty_sentence",
        }

    # Pipeline handles tokenizer max_length internally; very long lines are still OK.
    outputs = pipe(trimmed)
    first = outputs[0]
    label = str(first["label"])
    score = float(first["score"])
    is_toxic = is_predicted_toxic(label, score)
    return {
        "label": label,
        "score": score,
        "is_toxic": is_toxic,
        "detail": "ok",
    }


def new_sentence_item(text: str) -> dict[str, Any]:
    """Build one editable row for the safety step (no classification yet)."""
    return {
        "id": str(uuid.uuid4()),
        "text": text.strip(),
        "label": None,
        "score": None,
        "is_toxic": None,
        "checked": False,
    }


def validate_manual_sentence(text: str) -> tuple[str, str | None]:
    """Validate user-typed sentence length for add/edit paths.

    Returns:
        (trimmed_text, error_message_or_None)
    """
    t = text.strip()
    if not t:
        return "", "Sentence must not be empty."
    if len(t) > MAX_MANUAL_SENTENCE_CHARS:
        return (
            t,
            f"Sentence is too long ({len(t)} chars). Limit is {MAX_MANUAL_SENTENCE_CHARS}.",
        )
    return t, None
