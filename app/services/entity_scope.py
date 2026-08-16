"""Scope retrieval to companies / periods named in the question.

Dense search alone happily returns every similar SIE ledger in the tenant. When the
user names a company (or Q3 2024), matching ``metadata.company_name`` / year / quarter
must dominate — and non-matches must be demoted or dropped from the answer context.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from app.models.metadata import ChunkMetadata

_COMPANY_SUFFIXES = frozenset(
    {
        "ab",
        "aktiebolag",
        "hb",
        "kb",
        "abpubl",
        "plc",
        "ltd",
        "inc",
        "corp",
        "corporation",
        "group",
        "as",
        "oy",
        "gmbh",
    }
)

_QUARTER = re.compile(
    r"(?i)\b(?:q\s*([1-4])|kv(?:artal)?\s*([1-4])|([1-4]):e\s*kvartalet)\b"
)
_YEAR = re.compile(r"\b((?:19|20)\d{2})\b")


@dataclass(frozen=True)
class EntityScope:
    """Constraints inferred from the question and/or explicit request filters."""

    company_names: tuple[str, ...] = ()
    year: int | None = None
    quarter: int | None = None

    @property
    def active(self) -> bool:
        return bool(self.company_names) or self.year is not None or self.quarter is not None


def normalize_company(name: str) -> str:
    text = unicodedata.normalize("NFKC", name).lower().replace("&", " ")
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    tokens = [token for token in text.split() if token and token not in _COMPANY_SUFFIXES]
    return " ".join(tokens)


def company_mentioned_in_query(query: str, company_name: str) -> bool:
    """True when the query clearly refers to this indexed company."""
    needle = normalize_company(company_name)
    if len(needle) < 3:
        return False
    haystack = normalize_company(query)
    if needle in haystack:
        return True
    tokens = needle.split()
    # Require the distinctive name tokens (skip ultra-short crumbs).
    significant = [token for token in tokens if len(token) >= 4]
    if not significant:
        significant = tokens
    return bool(significant) and all(token in haystack for token in significant)


def parse_period_from_query(query: str) -> tuple[int | None, int | None]:
    quarter: int | None = None
    match = _QUARTER.search(query)
    if match:
        for group in match.groups():
            if group:
                quarter = int(group)
                break
    years = [int(value) for value in _YEAR.findall(query)]
    year = years[-1] if years else None
    return year, quarter


def companies_mentioned_in_text(
    text: str, candidate_metadata: list[ChunkMetadata]
) -> list[str]:
    """Indexed company names that appear in ``text``, first occurrence wins."""
    seen: set[str] = set()
    companies: list[str] = []
    for meta in candidate_metadata:
        key = normalize_company(meta.company_name)
        if not key or key in seen:
            continue
        if company_mentioned_in_query(text, meta.company_name):
            seen.add(key)
            companies.append(meta.company_name)
    return companies


def resolve_entity_scope(
    question: str,
    candidate_metadata: list[ChunkMetadata],
    *,
    explicit_company: str | None = None,
    explicit_year: int | None = None,
    explicit_quarter: int | None = None,
    history_text: str = "",
) -> EntityScope:
    """Build scope from explicit filters, else from question ∩ candidate companies.

    Follow-ups often use pronouns ("de", "they") and omit the company. When the
    current question names none, ``history_text`` (prior user turns) fills the
    company list. Year/quarter still come only from the current question so a
    Q4 thread does not hard-drop later "full year" questions.
    """
    companies: list[str] = []
    if explicit_company and explicit_company.strip():
        companies.append(explicit_company.strip())
    else:
        companies = companies_mentioned_in_text(question, candidate_metadata)
        if not companies and history_text:
            companies = companies_mentioned_in_text(history_text, candidate_metadata)

    year = explicit_year
    quarter = explicit_quarter
    if year is None or quarter is None:
        parsed_year, parsed_quarter = parse_period_from_query(question)
        if year is None:
            year = parsed_year
        if quarter is None:
            quarter = parsed_quarter

    return EntityScope(
        company_names=tuple(companies),
        year=year,
        quarter=quarter,
    )


def metadata_matches_scope(meta: ChunkMetadata, scope: EntityScope) -> bool:
    if not scope.active:
        return True
    if scope.company_names:
        allowed = {normalize_company(name) for name in scope.company_names}
        if normalize_company(meta.company_name) not in allowed:
            return False
    if scope.year is not None and meta.year != scope.year:
        return False
    if scope.quarter is not None and meta.quarter != scope.quarter:
        return False
    return True


def entity_rank_indices(
    question: str,
    metadatas: list[ChunkMetadata],
    *,
    explicit_company: str | None = None,
    explicit_year: int | None = None,
    explicit_quarter: int | None = None,
    history_text: str = "",
) -> tuple[EntityScope, list[int] | None]:
    """RRF list: in-scope first (best), out-of-scope last. ``None`` if nothing to scope."""
    scope = resolve_entity_scope(
        question,
        metadatas,
        explicit_company=explicit_company,
        explicit_year=explicit_year,
        explicit_quarter=explicit_quarter,
        history_text=history_text,
    )
    if not scope.active:
        return scope, None

    in_scope = [
        index
        for index, meta in enumerate(metadatas)
        if metadata_matches_scope(meta, scope)
    ]
    # Named company + mismatched year/quarter should still prefer that company
    # over a different firm's matching year.
    if not in_scope and scope.company_names:
        relaxed = EntityScope(company_names=scope.company_names)
        in_scope = [
            index
            for index, meta in enumerate(metadatas)
            if metadata_matches_scope(meta, relaxed)
        ]
        scope = relaxed
    if not in_scope:
        # Named entity not present in the pool — do not empty the results.
        return scope, None

    out_scope = [index for index in range(len(metadatas)) if index not in set(in_scope)]
    return scope, in_scope + out_scope


def filter_indices_to_scope(
    indices: list[int],
    metadatas: list[ChunkMetadata],
    scope: EntityScope,
) -> list[int]:
    """Drop out-of-scope hits when the question named an entity that exists in the pool."""
    if not scope.active:
        return indices
    matched = [
        index
        for index, meta in enumerate(metadatas)
        if metadata_matches_scope(meta, scope)
    ]
    if not matched:
        return indices
    allowed = set(matched)
    return [index for index in indices if index in allowed]
