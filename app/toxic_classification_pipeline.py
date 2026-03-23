"""
Pipeline 2: Chinese toxic  text classification.
global cache + get_*_pipeline + transformers.pipeline
Model: fine-tuned on ToxiCN (binary labels non_toxic / toxic in training notebook).
"""

from __future__ import annotations

import re
import uuid # for generating unique sentence ids
from typing import Any

from transformers import pipeline

# model settings
TOXIC_CLF_MODEL_ID = "echovivi/CustomModel_bertToxiCN"

# Manual sentence input limit
MAX_MANUAL_SENTENCE_CHARS = 100

# Global pipeline cache to avoid repeated loading
_text_classification_pipe: Any | None = None


def get_text_classification_pipeline() -> Any:
    """Lazily load the text-classification pipeline from Hugging Face Hub.
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


# If a split fragment has no CJK, no Latin letters, and no digits, treat it as
# "punctuation / quotes / symbols only" and merge with neighbors. This covers
# ASCII ``"`` and curly quotes ``"\u201c\u201d"`` without maintaining a huge list.
_HAS_LETTER_OR_DIGIT_OR_CJK = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]")

# Line that is only ASCII/Unicode hyphens and dashes (dialogue bullets, separators).
_DASH_ONLY_LINE = re.compile(
    r"^[\-\u2010\u2011\u2012\u2013\u2014\u2015\u2212\uFE58\uFE63\uFF0D\s]+$"
)


def _merge_dash_only_segments(sentences: list[str]) -> list[str]:
    """Merge segments that are only hyphens/dashes/spaces into neighbors.
    Catches lone ``-`` lines that should not be a separate detection unit.
    """
    if not sentences:
        return sentences
    out: list[str] = []
    pending = ""
    for s in sentences:
        t = s.strip()
        if not t:
            continue
        if _DASH_ONLY_LINE.fullmatch(t):
            pending += s
            continue
        if pending:
            s = pending + s
            pending = ""
        out.append(s)
    if pending:
        if out:
            out[-1] = out[-1] + pending
        else:
            out.append(pending.strip())
    return out


def _is_punctuation_only_chunk(s: str) -> bool:
    """True if s has no real words (only punctuation, quotes, spaces, symbols)."""
    if not s or not s.strip():
        return True
    return _HAS_LETTER_OR_DIGIT_OR_CJK.search(s) is None


def split_into_sentences(text: str) -> list[str]:
    """Split mixed Chinese/English UGC text into sentence-like units.
    Uses Chinese and ASCII end punctuation plus newlines. Merges punctuation-only
    fragments (e.g. a lone ``。`` after split) onto the previous or next real segment.

    Args:
        text: Raw generated or edited text.

    Returns:
        List of stripped sentence strings.
    """
    text = text.strip()
    if not text:
        return []

    # Split after 。！？ or . ! ? and also on line breaks.
    parts = re.split(r"(?<=[。！？!?])\s*|\n+", text)
    raw: list[str] = []
    for p in parts:
        s = p.strip()
        if s:
            raw.append(s)
    if not raw:
        return [text]

    merged: list[str] = []
    pending_leading_punct = ""

    for s in raw:
        if _is_punctuation_only_chunk(s):
            if merged:
                merged[-1] = merged[-1] + s
            else:
                # No previous sentence: keep until we can prepend to real text.
                pending_leading_punct += s
            continue

        if pending_leading_punct:
            s = pending_leading_punct + s
            pending_leading_punct = ""
        merged.append(s)

    if pending_leading_punct:
        if merged:
            merged[-1] = merged[-1] + pending_leading_punct
        else:
            merged.append(pending_leading_punct.strip())

    merged = _merge_dash_only_segments(merged)
    if (
        len(merged) == 1
        and merged[0].strip()
        and _DASH_ONLY_LINE.fullmatch(merged[0].strip())
    ):
        return []
    return merged if merged else [text]


def _normalize_label(label: str) -> str:
    """Lowercase and simplify spaces for rule checks."""
    return label.strip().lower().replace(" ", "_").replace("-", "_")


def is_predicted_toxic(label: str) -> bool:
    """Binary toxic decision for Hub labels: non_toxic / toxic."""
    lab = _normalize_label(label)
    if lab == "toxic":
        return True
    if lab == "non_toxic":
        return False
    # Defensive default: if label is unexpected, block publish for safety.
    return True


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
    is_toxic = is_predicted_toxic(label)
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
