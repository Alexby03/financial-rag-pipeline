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
    }
)


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN.findall(text)]


def content_terms(query: str) -> list[str]:
    """Query tokens with Swedish/English function words removed."""
    return [token for token in tokenize(query) if token not in _STOPWORDS and len(token) > 1]


def query_phrase(query: str) -> str | None:
    terms = content_terms(query)
    if len(terms) < 2:
        return None
    return " ".join(terms)


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

    # Exact phrase bonus: significant query terms appear contiguously in the document.
    phrase = query_phrase(query)
    if phrase:
        for index, tokens in enumerate(doc_tokens):
            if phrase in " ".join(tokens):
                scores[index] += 2.0

    return sorted(range(doc_count), key=lambda i: (-scores[i], i))
