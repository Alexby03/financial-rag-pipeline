# Answering

What happens after `top_k` is chosen. Ranking is [ranking.md](ranking.md). The
map is [rag_strategy.md](rag_strategy.md).

## 1. Neighbour expansion

Chunks form a double-linked list within each document (`previous_chunk_id` /
`next_chunk_id`). When `expand_neighbors` is set, those neighbours of every
**selected** hit are fetched and merged into the context, then ordered back into
reading order per source file.

This runs *after* ranking, so neighbours are fetched for the chunks that were
actually selected rather than for candidates about to be discarded.

## 2. Context format

Each chunk is a header line plus body:

```
[source.pdf p.12 text chunk=<id> prev=<id> next=<id>]
```

The model can cite filename and page, and tell when a table continues on the
next chunk. Figure chunks are tagged `figure` and still carry the parent
`object_key` and `page_number`.

Financial tables carry period labels as markdown header cells (`Post | date1 | date2`)
and may include a `Rows:` narrative block below spreadsheet tables. Those dates are column
headers left→right after the row label. See [document_formats.md](document_formats.md) and
[ingestion_pipeline.md](ingestion_pipeline.md) §3.

## 3. Figure vision

Figure chunks retrieve on their OCR caption. That caption is a hint, not a
reading of the chart. After retrieval, the API downloads up to four stored
crops from MinIO (`figure_object_key`) and asks `OPENAI_VISION_MODEL` (default
`gpt-4o`) to answer the user's question from the image. Those readings are
placed first in the reasoner prompt as authoritative figure evidence, then
repeated on the figure chunks. The reasoning model never sees the pixels.

A failed crop download or vision call is recorded on the trace and skipped; the
query still answers from remaining text. Set `OPENAI_VISION_MODEL` blank to
disable the pass.

## 4. Generation

Answers come from `OPENAI_ANSWER_MODEL` through the Responses API with
`reasoning={"effort": ..., "summary": ...}`. Ingest parsing uses Mistral OCR 4,
not this model.

`chat_history` on the request is prior user/assistant turns from the browser
tab. Nothing is stored on the server. Retrieval uses recent user turns as in
[ranking.md](ranking.md) §1; generation receives the current question plus the
turn list.

The system prompt restricts the answer to this turn's context chunks, prefers
vision readings over captions for plotted numbers, says so when context is
insufficient, and cites filenames and page numbers. When the **conversation** is
about a named company, year, or quarter, it must not blend figures from another
entity that happens to appear in context.

Reasoning summaries come back empty for trivial lookups even though the model
reasoned internally. Treat an empty summary as "no extended reasoning to
report", not as an error.

## 5. Sources and the trace

Every answer returns source chunks with `score`, `rerank_score`, and full
metadata.

The frontend resolves `metadata.object_key` through `GET /sources/url` into a
presigned MinIO URL, including for figure chunks. The browser opens the original
PDF on `page_number`; cropped figure bytes stay in MinIO for vision, not for
Open. PDFs and images are served `inline`; other formats `attachment`. The
endpoint rejects any object key that does not begin with the requesting
tenant's id.

The trace carries the filters applied, `retrieval_question`, each hit with
vector rank, lexical rank, RRF score and rerank score, `fusion.method`,
neighbour chunk ids, figure vision readings, context size, timings, and the
reasoning summary.
