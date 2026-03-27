"""
Pipeline 2: Chinese toxic text classification + sensitive wordlist scan.
Global cache + get_*_pipeline + transformers.pipeline.
Model: fine-tuned on ToxiCN (binary labels non_toxic / toxic in training notebook).
Wordlist: fetched over HTTPS from cjh0613/tencent-sensitive-words (raw GitHub), cached in memory only.
"""

from __future__ import annotations

import re
import urllib.error
import urllib.request
import uuid  # for generating unique sentence ids
from typing import Any, Sequence

from transformers import pipeline

# model settings
TOXIC_CLF_MODEL_ID = "echovivi/CustomModel_bertToxiCN"

# Tencent-style list: one term per line
TENCENT_SENSITIVE_WORDS_URL = (
    "https://raw.githubusercontent.com/cjh0613/tencent-sensitive-words/main/sensitive_words_lines.txt"
)
# Large file (~650KB); allow slow links / Streamlit Cloud cold start
_SENSITIVE_WORDS_FETCH_TIMEOUT_SEC = 120
_SENSITIVE_WORDS_USER_AGENT = "GameUGC-Safety/1.0 (course project; +https://github.com/cjh0613/tencent-sensitive-words)"

# Cap stored hits per sentence; scanning still finds all for bool decision
MAX_DICT_HITS_STORED = 32

# Manual sentence input limit
MAX_MANUAL_SENTENCE_CHARS = 100

# Global pipeline cache to avoid repeated loading
_text_classification_pipe: Any | None = None
_sensitive_words_cache: list[str] | None = None


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


def load_sensitive_words() -> list[str]:
    """Download the Tencent-style word list once per process and cache in memory.
    """
    global _sensitive_words_cache
    if _sensitive_words_cache is not None:
        return _sensitive_words_cache

    req = urllib.request.Request(
        TENCENT_SENSITIVE_WORDS_URL,
        headers={"User-Agent": _SENSITIVE_WORDS_USER_AGENT},
    )
    try:
        with urllib.request.urlopen(req, timeout=_SENSITIVE_WORDS_FETCH_TIMEOUT_SEC) as resp:
            raw_bytes = resp.read()
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Could not download sensitive word list from GitHub "
            f"({TENCENT_SENSITIVE_WORDS_URL}). Check network access. "
            f"Original error: {exc}"
        ) from exc

    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError("Sensitive word list is not valid UTF-8.") from exc

    words: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        words.append(line)

    seen: set[str] = set()
    unique: list[str] = []
    for w in words:
        if w not in seen:
            seen.add(w)
            unique.append(w)

    if not unique:
        raise ValueError(
            "Downloaded sensitive word list is empty after parsing. "
            "Upstream file may have changed format."
        )

    _sensitive_words_cache = unique
    return _sensitive_words_cache


def get_sensitive_word_list() -> list[str]:
    """Return cached sensitive terms (downloads from GitHub on first call)."""
    return load_sensitive_words()


def _normalize_text_for_dict_scan(text: str) -> str:
    """Minimal normalization before substring scan (strip only for v1)."""
    return text.strip()


def scan_tencent_offline_hits(text: str, words: Sequence[str]) -> list[str]:
    """Find sensitive terms that appear as substrings in ``text``.
    """
    t = _normalize_text_for_dict_scan(text)
    if not t:
        return []

    hits: list[str] = []
    for w in words:
        if not w:
            continue
        if w in t:
            hits.append(w)
    return hits


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


def classify_one_sentence_with_wordlist(pipe: Any, text: str) -> dict[str, Any]:
    """Run sensitive wordlist scan (GitHub-fetched, in-memory) plus HF classification; merge into one gate.
    A sentence fails if **either** the model predicts toxic **or** any word hits.
    """
    trimmed = text.strip()
    if not trimmed:
        return {
            "label": "",
            "score": 0.0,
            "is_toxic": False,
            "dict_hits": [],
            "is_sensitive_hit": False,
            "model_is_toxic": False,
            "detail": "empty_sentence",
        }

    word_list = load_sensitive_words()
    all_hits = scan_tencent_offline_hits(trimmed, word_list)
    is_sensitive_hit = len(all_hits) > 0
    dict_hits_stored = all_hits[:MAX_DICT_HITS_STORED]

    model_out = classify_one_sentence(pipe, trimmed)
    model_toxic = bool(model_out["is_toxic"])
    combined_toxic = model_toxic or is_sensitive_hit

    return {
        "label": model_out["label"],
        "score": float(model_out["score"]),
        "is_toxic": combined_toxic,
        "dict_hits": dict_hits_stored,
        "is_sensitive_hit": is_sensitive_hit,
        "model_is_toxic": model_toxic,
        "detail": str(model_out.get("detail", "ok")),
    }


def new_sentence_item(text: str) -> dict[str, Any]:
    """Build one editable row for the safety step (no classification yet)."""
    return {
        "id": str(uuid.uuid4()),
        "text": text.strip(),
        "label": None,
        "score": None,
        "is_toxic": None,
        "dict_hits": None,
        "is_sensitive_hit": None,
        "model_is_toxic": None,
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
