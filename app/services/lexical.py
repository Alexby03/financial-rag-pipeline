"""Local BM25 over a small candidate set (late lexical fusion).

Scores are computed only among the dense-retrieved candidates — no collection-wide
index. That is enough to promote exact terms like account names that embeddings miss.
"""

from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN = re.compile(r"[0-9A-Za-zÀ-ÖØ-öø-ÿ]+", re.UNICODE)
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "of",
        "in",
        "on",
        "for",
        "to",
        "is",
        "was",
        "what",
        "vad",
        "är",
        "var",
        "och",
        "i",
        "på",
        "av",
        "för",
        "en",
        "ett",
        "som",
        "att",
        "med",
        "om",
        "den",
        "det",
        "de",
        "hur",
        "vilken",
        "vilket",
        "vilka",
        "kr",
        "sek",
    }
)


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN.findall(text)]


def content_terms(query: str) -> list[str]:
    """Query tokens with Swedish/English function words removed."""
    return [token for token in tokenize(query) if token not in _STOPWORDS and len(token) > 1]


def content_phrases(query: str, *, min_len: int = 2, max_len: int = 4) -> list[str]:
    """Contiguous content-term phrases from the query (not the whole question).

    Stopwords and bare years/numbers break runs so
    "vad var Balanserat resultat i kr 2024 för RTP Radical …" yields
    ``balanserat resultat`` and company-name n-grams — never one giant join of
    every content word (which never matches any chunk).
    """
    runs: list[list[str]] = []
    current: list[str] = []
    for token in tokenize(query):
        if token in _STOPWORDS or len(token) <= 1 or token.isdigit():
            if current:
                runs.append(current)
                current = []
            continue
        current.append(token)
    if current:
        runs.append(current)

    phrases: list[str] = []
    seen: set[str] = set()
    for run in runs:
        windows: list[list[str]] = []
        if min_len <= len(run) <= max_len:
            windows.append(run)
        for size in range(min(max_len, len(run)), min_len - 1, -1):
            for start in range(0, len(run) - size + 1):
                windows.append(run[start : start + size])
        for window in windows:
            phrase = " ".join(window)
            if phrase not in seen:
                seen.add(phrase)
                phrases.append(phrase)
    return phrases


def query_phrase(query: str) -> str | None:
    """Best single phrase for simple callers; prefer ``content_phrases``."""
    phrases = content_phrases(query)
    return phrases[0] if phrases else None


def bm25_rank_indices(
    query: str,
    documents: list[str],
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[int]:
    """Return document indices sorted best→worst by BM25 against ``query``."""
    if not documents:
        return []

    query_terms = tokenize(query)
    if not query_terms:
        return list(range(len(documents)))

    doc_tokens = [tokenize(document) for document in documents]
    doc_freqs = [Counter(tokens) for tokens in doc_tokens]
    lengths = [len(tokens) for tokens in doc_tokens]
    avgdl = sum(lengths) / len(lengths) if lengths else 0.0
    doc_count = len(documents)

    df: Counter[str] = Counter()
    for freq in doc_freqs:
        for term in freq:
            df[term] += 1

    scores = [0.0] * doc_count
    query_tf = Counter(query_terms)
    for term, q_weight in query_tf.items():
        n_q = df.get(term, 0)
        if n_q == 0:
            continue
        idf = math.log(1.0 + (doc_count - n_q + 0.5) / (n_q + 0.5))
        for index, freq in enumerate(doc_freqs):
            tf = freq.get(term, 0)
            if tf == 0 or lengths[index] == 0:
                continue
            denom = tf + k1 * (1.0 - b + b * lengths[index] / avgdl)
            scores[index] += idf * (tf * (k1 + 1.0) / denom) * q_weight

    # Phrase bonus: only rare contiguous matches. Summing every company-name n-gram
    # otherwise drowns a unique account line that appears in a single candidate.
    phrases = content_phrases(query)
    if phrases:
        joined = [" ".join(tokens) for tokens in doc_tokens]
        phrase_df = {
            phrase: sum(1 for text in joined if phrase in text) for phrase in phrases
        }
        rarity_cap = max(1, doc_count // 4)
        for index, text in enumerate(joined):
            best = 0.0
            for phrase, n_docs in phrase_df.items():
                if n_docs == 0 or n_docs > rarity_cap or phrase not in text:
                    continue
                best = max(best, (8.0 if n_docs == 1 else 2.0) * (len(phrase.split()) / n_docs))
            scores[index] += best

    return sorted(range(doc_count), key=lambda i: (-scores[i], i))
