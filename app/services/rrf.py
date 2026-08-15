"""Reciprocal Rank Fusion over multiple ranked lists.

Classic form from Cormack, Clarke & Buettcher — the default fusion used by
Elasticsearch, Azure AI Search, and most hybrid RAG stacks:

    score(d) = Σ 1 / (k + rank_i(d))
"""

from __future__ import annotations


def reciprocal_rank_fusion(
    rankings: list[list[int]],
    *,
    k: int = 60,
    tie_break_ranks: dict[int, int] | None = None,
) -> list[tuple[int, float]]:
    """Fuse ranked index lists into ``(index, rrf_score)`` best→worst.

    Each ``rankings[i]`` is a permutation of candidate indices (best first).
    Candidates missing from a list simply get no contribution from that list.
    ``tie_break_ranks`` maps index → preferred rank (lower wins) when RRF ties.
    """
    if not rankings:
        return []

    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, index in enumerate(ranking, start=1):
            scores[index] = scores.get(index, 0.0) + 1.0 / (k + rank)

    breakers = tie_break_ranks or {}
    return sorted(
        scores.items(),
        key=lambda item: (-item[1], breakers.get(item[0], 10**9), item[0]),
    )
