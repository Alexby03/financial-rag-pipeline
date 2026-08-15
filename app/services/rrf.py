"""Reciprocal Rank Fusion over multiple ranked lists.

Classic form from Cormack, Clarke & Buettcher — the default fusion used by
Elasticsearch, Azure AI Search, and most hybrid RAG stacks:

    score(d) = Σ w_i / (k + rank_i(d))
"""

from __future__ import annotations


def reciprocal_rank_fusion(
    rankings: list[list[int]],
    *,
    k: int = 60,
    weights: list[float] | None = None,
    tie_break_ranks: dict[int, int] | None = None,
) -> list[tuple[int, float]]:
    """Fuse ranked index lists into ``(index, rrf_score)`` best→worst.

    Each ``rankings[i]`` is a permutation of candidate indices (best first).
    Candidates missing from a list simply get no contribution from that list.
    ``weights`` scales each list (default 1.0); use >1 for an exact-phrase list.
    ``tie_break_ranks`` maps index → preferred rank (lower wins) when RRF ties.
    """
    if not rankings:
        return []

    if weights is None:
        list_weights = [1.0] * len(rankings)
    elif len(weights) != len(rankings):
        raise ValueError("weights must match rankings length")
    else:
        list_weights = weights

    scores: dict[int, float] = {}
    for ranking, weight in zip(rankings, list_weights, strict=True):
        for rank, index in enumerate(ranking, start=1):
            scores[index] = scores.get(index, 0.0) + weight / (k + rank)

    breakers = tie_break_ranks or {}
    return sorted(
        scores.items(),
        key=lambda item: (-item[1], breakers.get(item[0], 10**9), item[0]),
    )
