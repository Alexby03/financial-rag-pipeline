# Ranking

How recall and precision turn a question into `top_k` chunks. The map is
[rag_strategy.md](rag_strategy.md). What happens after `top_k` is
[answering.md](answering.md).

Code: `RagService.query` → `_hits_for_entity_scope` → `_select_candidates`.

## 1. Conversational query

The question is embedded with `text-embedding-3-small` — the same model as ingest.
Query and documents must share an embedding space, so changing the model requires
re-embedding every chunk.

Follow-ups often drop the company ("berätta om de är väl positionerade…").
`chat_history` is not only for generation: the last **two user turns** are
concatenated onto the current question for embed, BM25, Cohere, and vision. The
reasoner still sees the current question plus the full turn list separately.
The string actually searched is `trace.retrieval_question`.

Company names for entity scope are taken from the current question first; if none
match candidate metadata, **all** prior user turns are scanned. Year and quarter
still come only from the current question (or explicit filters), so a Q4 thread
does not lock a later "full year" question.

## 2. Filtered vector search

Qdrant is queried with `query_points` (`client.search()` was removed in
qdrant-client 1.16). Results live on `response.points`.

`must` clauses: `metadata.tenant_id`, plus any of `company_name`, `doc_type`,
`category`, `year`, `quarter` supplied on the request. Sidebar filters win here.
Tenant isolation is enforced at the database.

The number of candidates is **not** `top_k`. When Cohere **or** hybrid BM25 is
active it is `top_k * RERANK_CANDIDATE_MULTIPLIER` (default 4), because both can
only promote chunks vector search already returned. With defaults that is a pool
of 20. Hybrid and Cohere off: the limit is `top_k`.

### Inferred company re-query

When the question (or prior user turns) **names** a company that appears on
candidate metadata, and the sidebar did not already filter `company_name`,
retrieval infers that entity and **re-queries Qdrant** with
`metadata.company_name` so the pool is that company.

Asking for Testföretaget AB must not surface Datakonsulterna AB just because both
are SIE ledgers — and `top_k=5` must still mean five Testföretaget chunks, not
one leftover after a hard-drop on a mixed pool of 20. If the company-filtered
search returns fewer hits than the first in-scope set, the first set is kept.

## 3. BM25, phrases, RRF

Embedding similarity ranks by how alike two texts look, which buries chunks whose
wording matches the question exactly (account names, "Balanserat resultat").
Industry default is **hybrid retrieval**: dense rank + lexical rank, fused with
**Reciprocal Rank Fusion** (Cormack et al.):

```
score(d) = Σ wᵢ / (k + rankᵢ(d))   with k = RRF_K = 60
```

Lexical rank is **local BM25 over the dense candidate pool** (no extra Qdrant
index). A third list (weight **2**) ranks hits for **contiguous content phrases**
extracted from the question, using only phrases that are rare in that pool so
cover-page company names do not drown out a unique account line.

A fourth list (weight **3**) ranks in-scope entity chunks first when the pool is
still mixed company or period. Stronger than the phrase boost: wrong-company SIE
ledgers must lose.

`HYBRID_LEXICAL=false` skips the BM25 leg. When hybrid is off and Cohere is on,
final order is Cohere alone. When both are off, final order is vector order.

RRF uses **rank, not score**. A 0.96 vs 0.65 Cohere gap is a small rank delta;
fused equally with BM25, a generic "growing market" hit that BM25 ranks #1 used
to bury the chunk Cohere scored 0.96. That is why Cohere is no longer an RRF
peer when it applied.

## 4. Cohere rerank

When Cohere Rerank is on, it is the **final ranking**: `top_k` is taken from
Cohere order after entity hard-drop. BM25 + RRF still run and are recorded
(`lexical_rank`, `rrf_score`) so you can see when recall disagreed.

Cohere is optional at three levels:

1. Inactive unless both `COHERE_API_BASE` and `COHERE_API_KEY` are set.
2. `QueryRequest.rerank` turns it off per query.
3. Any failure falls back to vector+BM25 fusion rather than failing the request.

The fallback reason is `trace.rerank.fallback_reason` and a UI warning.
`trace.fusion.method` is `cohere` or `rrf`.

Operational traps (also in [operations.md](operations.md)):

- The model name must match the Foundry **deployment name** exactly, or the
  endpoint answers 404 `DeploymentNotFound`.
- `build_rerank_url` normalizes the base: already ending in `/rerank` is used as
  is; ending in `/providers/cohere` gets `/v2/rerank`; anything else gets
  `/rerank`.

## 5. Hard-drop and fill `top_k`

After the chosen order (Cohere or RRF), chunks that are not the named
company/period are dropped. The selection is then topped up from in-scope
candidates so `top_k` stays filled when those chunks exist in the pool (or in
the company re-query). Named entity not present in the pool: results are not
emptied.

Period mismatch with a named company relaxes to company-only so a Q4 question
can still prefer that issuer over another firm's matching year.
