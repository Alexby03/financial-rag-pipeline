# Retrieval strategy

How a question becomes an answer with citations. The write side is
[ingestion_pipeline.md](ingestion_pipeline.md). Ranking detail is
[ranking.md](ranking.md). Neighbours, vision, generation, and sources are
[answering.md](answering.md). Failure modes are [operations.md](operations.md).

Entry point: `POST /query` → `RagService.query`.

```
question (+ prior user turns) → embed → Qdrant → (company re-query) → Cohere + BM25
    → pick top_k → neighbours → vision → answer
```

Each step is recorded on a `RetrievalTrace` so the UI shows what the agent did.

## Stages

| Stage | Job | Where |
|-------|-----|--------|
| **0. Conversational query** | Last two user turns concatenated onto the current question (`trace.retrieval_question`) | [ranking.md](ranking.md) §1 |
| **1. Recall** | Embed, Qdrant vector search, optional second search for the inferred company. Pool size is `top_k × RERANK_CANDIDATE_MULTIPLIER` (default 5 × 4 = 20) | [ranking.md](ranking.md) §2 |
| **2. Precision** | Cohere rerank is the final order when it runs. BM25 + RRF still run and fill the hits table. Entity hard-drop then fills `top_k` | [ranking.md](ranking.md) §3–5 |
| **3. Context** | Neighbour chunks, figure vision, reasoning model | [answering.md](answering.md) |

Sidebar filters (`company_name`, `doc_type`, `category`, `year`, `quarter`) are Qdrant
`must` clauses on the first search. Tenant isolation is `metadata.tenant_id` at the
database, not an application-side filter after the fact.

## Hits-table scores

These columns are **different scales**. Do not threshold them against each other.

| Column | Field | Produced by | Meaning |
|--------|--------|-------------|---------|
| Vector | `vector_rank` | Qdrant cosine order | 1 = closest in the candidate pool |
| Score | `vector_score` | Cosine similarity | ~0–1; not comparable to Rerank |
| Lexical | `lexical_rank` | BM25 over the pool | 1 = best exact-term match |
| RRF | `rrf_score` | `Σ wᵢ / (k + rankᵢ)`, `k=60` | Typical ~0.02–0.08; rank fusion, not magnitude |
| Rerank | `rerank_score` | Cohere cross-encoder | 0–1; **row order** when `fusion.method` is `cohere` |

`fusion.method` is `cohere` when rerank applied, else `rrf`. The UI caption
"Final ranking is Cohere rerank order" is that rule — by design, not a warning.

RRF list weights, when hybrid is on: vector or Cohere **1**, BM25 **1**, rare
phrases **2**, entity **3**. Phrase and entity lists are omitted when they have
nothing to add.

## Follow-up in one pass

1. User asks about INOD Q4, then "berätta om de är väl positionerade…".
2. Retrieval embeds the concatenated questions, so entity scope still sees INOD.
3. If the mixed vector pool only has one INOD chunk, a second Qdrant search with
   `metadata.company_name` fills the window.
4. Cohere scores the INOD pages ~0.96; those rows are `top_k`, not the BM25
   leaders from other issuers.
5. The reasoner gets `top_k` INOD chunks plus any neighbours and figure readings.
