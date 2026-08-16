# Retrieval strategy

How a question becomes an answer with citations. The write side is described in
[ingestion_pipeline.md](ingestion_pipeline.md).

Entry point: `POST /query` → `RagService.query`.

```
question (+ prior user turns) → embed → filtered vector search → (Cohere) → BM25 + RRF (trace) → neighbours → vision → answer
```

Each step is recorded in a `RetrievalTrace` returned with the answer, so the UI can show what
the agent actually did rather than asserting that it worked. The ranking hops and score
columns are drawn in the Cursor canvas **RAG ranking process** (open beside chat). Failure
modes for the same path are in [`operations.md`](operations.md).

## 1. Embed the question

The question is embedded with the same model used at ingest, `text-embedding-3-small`. Query and
documents must share an embedding space, so changing the model requires re-embedding every
chunk.

Follow-ups in the same chat often drop the company ("berätta om de är väl positionerade…").
`chat_history` is not only for generation: the last two user turns are concatenated onto
the current question for embed, BM25, Cohere, and vision. The reasoner still sees the
current question plus the full turn list separately. The concatenated string is on
`trace.retrieval_question`.

## 2. Filtered vector search

Qdrant is queried with a `must` filter on `metadata.tenant_id`, plus any of `company_name`,
`doc_type`, `category`, `year` and `quarter` supplied in the request. Tenant isolation is
enforced here, at the database, not in application code after the fact.

When the question **names** a company (or Q3/year) that appears on candidate metadata,
retrieval infers that entity and **re-queries Qdrant** with `metadata.company_name` so
the candidate pool is that company, then **hard-drops** any remaining out-of-scope
chunks. Asking for Testföretaget AB must not surface Datakonsulterna AB just because
both are SIE ledgers — and `top_k=5` must still mean five Testföretaget chunks, not
one leftover after the drop. Explicit sidebar filters still win as Qdrant `must`
clauses on the first search (no second hop). Pronoun follow-ups inherit the company
from prior **user** turns in `chat_history`; year and quarter still come only from
the current question (or explicit filters), so a Q4 thread does not lock later
"full year" questions.

The number of candidates fetched is **not** `top_k`. When Cohere rerank **or** hybrid lexical
fusion is active it is `top_k * RERANK_CANDIDATE_MULTIPLIER`, because both the reranker and
BM25 can only promote chunks that vector search already returned. Raising `top_k` alone does
not widen the pool.

Note that `client.search()` was removed in qdrant-client 1.16; the current call is
`client.query_points()`, whose results live on `response.points`.

## 3. Hybrid fusion (BM25 + RRF) and Cohere rerank

Embedding similarity ranks by how alike two texts look, which buries chunks whose wording
matches the question exactly (account names, "Balanserat resultat") when nearby prose scores
higher. Industry default is **hybrid retrieval**: dense rank + lexical rank, fused with
**Reciprocal Rank Fusion** (RRF, Cormack et al.):

```
score(d) = Σ 1 / (k + rank_i(d))   with k = 60 by default
```

Here lexical rank is **local BM25 over the dense candidate pool** (no extra Qdrant index),
plus a bonus for **contiguous content phrases** extracted from the question (e.g.
`balanserat resultat`), using only phrases that are rare in that pool so cover-page
company names do not drown out a unique account line. A third RRF list (weight 2) ranks
the same rare phrase hits. That lifts exact terms without re-embedding or retuning Cohere.

When Cohere Rerank is on, it is the **final ranking**: `top_k` is taken from Cohere
order after entity hard-drop. BM25 + RRF still run and are recorded on the trace
(`lexical_rank`, `rrf_score`) so you can see when first-stage recall disagreed,
but they do not veto a 0.96 vs 0.65 relevance gap. RRF uses rank, not score —
fusing Cohere rank equally with BM25 lets a generic "growing market" hit that
BM25 ranks #1 bury the chunk Cohere scored 0.96. When Cohere is off or fails
open, RRF fuses **vector + BM25** as before.
`HYBRID_LEXICAL=false` disables the BM25 leg; `RRF_K` tunes the fusion constant.

Cohere remains optional at three levels:

1. It is inactive unless both `COHERE_API_BASE` and `COHERE_API_KEY` are set.
2. `QueryRequest.rerank` turns it off per query.
3. Any failure falls back to vector+BM25 fusion rather than failing the request.

The Cohere fallback reason is always reported in `trace.rerank.fallback_reason` and rendered
as a warning in the UI. Fusion details land on `trace.fusion` (method, lexical/cohere flags,
`rrf_k`). Hit rows also expose `lexical_rank` and `rrf_score`.

Two operational traps are worth repeating:

- **The model name must match the deployment name exactly** on project endpoints. Otherwise the
  endpoint answers 404 `DeploymentNotFound` even though the URL and key are correct.
- **The base URL comes in several shapes** and the Foundry portal displays the wrong one for
  Cohere deployments. `build_rerank_url` normalizes them: a base already ending in `/rerank` is
  used as is, one ending in `/providers/cohere` gets `/v2/rerank`, anything else gets `/rerank`.

`score` (cosine), `rerank_score` (Cohere), and `rrf_score` are different scales — do not
threshold them against each other.

## 4. Neighbour expansion

Chunks form a double-linked list within each document. When `expand_neighbors` is set, the
`previous_chunk_id` and `next_chunk_id` of every selected chunk are fetched and merged into the
context, then ordered back into reading order per source file.

This runs *after* the final ranking, so neighbours are fetched for the chunks that were actually
selected rather than for candidates that were about to be discarded.

## 5. Figure vision, then generation

Context is formatted with a header line per chunk carrying the source filename, page number and
link ids, so the model can both cite sources and tell when a chunk continues elsewhere.

Figure chunks still retrieve on their OCR caption. That caption is a hint, not a reading of the
chart. After retrieval, the API downloads up to four stored crops from MinIO
(`figure_object_key`) and asks `OPENAI_VISION_MODEL` (default `gpt-4o`) to answer the
user's question from the image. Those readings are placed first in the reasoner prompt as
authoritative figure evidence, then repeated on the figure chunks. The reasoning model never
sees the pixels — it synthesizes from captions plus vision readings.

A failed crop download or vision call is recorded on the trace and skipped; the query still
answers from whatever text remains. Set `OPENAI_VISION_MODEL` blank to disable the pass.

Answers are generated by a reasoning model (`OPENAI_ANSWER_MODEL`) through the Responses API
with `reasoning={"effort": ..., "summary": ...}`. Ingest parsing uses Mistral OCR 4, not this
model. Optional `chat_history` on the request carries prior user/assistant turns from the
browser tab so follow-ups resolve against the same conversation; nothing is stored on the
server. Retrieval concatenates recent user turns onto the current question (see §1);
generation still receives the current question plus the turn list.

The system prompt restricts the answer to the provided chunks, tells it to prefer vision
readings over captions for plotted numbers, instructs it to say when context is insufficient,
and asks it to cite filenames and page numbers.

Reasoning summaries come back empty for trivial lookups even though the model reasoned
internally. Treat an empty summary as "no extended reasoning to report", not as an error.

## 6. Sources and transparency

Every answer returns its source chunks with `score`, `rerank_score` and full metadata.

The frontend resolves `metadata.object_key` through `GET /sources/url` into a presigned MinIO
URL, including for figure chunks. The browser opens the original PDF on `page_number`; cropped
figure bytes stay in MinIO for the vision pass, not for the Open button. PDFs and images are
served `inline`; other formats are served `attachment` and download. The endpoint rejects any
object key that does not begin with the requesting tenant's id.

The trace carries the filters applied, the retrieval question actually embedded, each hit with
its vector rank, lexical rank, RRF score and rerank score, `fusion.method` (`cohere` or `rrf`),
the neighbour chunk ids, any figure vision readings, context size, timings, and the reasoning
summary.
