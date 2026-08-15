# Ingestion pipeline

How a document becomes searchable. The retrieval side is described in
[rag_strategy.md](rag_strategy.md).

Entry point: `POST /ingest` accepts the upload and returns `202` with a `job_id`. A lifespan
worker drains an in-memory queue and runs `IngestService.ingest` off the event loop. Poll
`GET /ingest/jobs/{job_id}` for `queued` → `running` → `completed` | `duplicate` | `failed`.
Duplicate detection still happens inside the worker so batch uploads can share the same path.

```
file bytes → enqueue → (worker) hash → MinIO → parse → chunk → link → embed → Qdrant
                └─ 202 job_id
```

Jobs are not durable across process restarts; re-upload after a crash. That is an accepted MVP
trade-off until a real queue is introduced.

## 1. Identity, then storage

The first thing the worker does is hash the bytes. A sha256 digest of the file is the document's
identity, and it is checked against the index before anything expensive happens: if the same
tenant already has chunks carrying that hash, the job finishes with
`status="duplicate"`, the existing `object_key` and the existing chunk count. Nothing is
uploaded, parsed, embedded or written.

This matters because parsing is by far the most expensive step, and filenames are a terrible
identity. `report.pdf`, `report (1).pdf` and `@report.pdf` are three names for one document, and
before hashing each one produced its own complete set of chunks — paid for twice and retrieved
twice, competing for slots in the same answer.

Once a document is known to be new, it is stored before it is parsed so every later step has
an `object_key`. If parse, embed, or index fails, those MinIO objects (the original file and
any figures) are deleted before the job is marked failed. A blob without an indexed document
is not left behind.

- `document_id` is a fresh UUID per ingest.
- `object_key` is `{tenant_id}/{document_id}/{filename}` and is written into every chunk's
  metadata. Without it there is no path from a retrieved chunk back to the source file.
- `content_hash` is written into every chunk's metadata alongside it.
- Content type is guessed from the filename so MinIO serves the file correctly when the
  frontend later opens it.

## 2. Mistral OCR 4 parsing

`DocumentParser` sends PDF, Office (DOCX/PPTX), and images to Mistral OCR 4 on Azure AI
Foundry. The OCR response is mapped to Markdown pages plus optional figures. Plain `.txt` /
`.md` stay local (verbatim UTF-8) because they are already text. Swedish SIE accounting
files (`.sie` / `.si` / `.se`) are parsed locally into Markdown summaries without OCR.
Bolagsverket-style XHTML is printed to a multi-page PDF (page containers / CSS breaks), then
OCR'd in the same 30-page slices as a normal PDF. No OpenAI model reads a document. Foundry
rejects PDFs over 30 pages (`document_parser_too_many_pages`), so a long PDF is sliced with
pypdf and each slice is OCR'd separately; page numbers are mapped back onto the original
document.

| Input | Strategy | Output |
|-------|----------|--------|
| PDF | Mistral OCR 4, split into 30-page slices (Foundry cap) | Markdown pages + figures |
| DOCX, PPTX, images | Mistral OCR 4 (`/ocr`, one call) | Markdown pages + figures |
| Plain text / Markdown | Verbatim decode | Single Markdown page |
| SIE (`.sie`, `.si`, `.se`) | Lean vendored SIE 4 parser → LLM summaries (no OCR) | Markdown summary pages |
| XHTML / HTML | Headless Edge/Chrome print → PDF → OCR (30-page slices) | Markdown pages + figures |

**XHTML annual reports.** Bolagsverket iXBRL files are one long scrollable document. Ingest
detects print pages via containers such as `rb-page`, `ar-page`, and `pagebreak_before` (or
falls back to the browser honouring CSS `@page` / `page-break-*`). Each page is printed to
PDF with headless Edge or Chrome (`CHROME_PATH` / `EDGE_PATH` override), the pages are merged,
and the result reuses the PDF OCR path — including 30-page Foundry batches. The original
`.xhtml` bytes stay in MinIO; Open still points at that source file.

**SIE accounting files.** Swedish SIE 4 exports are tagged plain text (usually CP437). The
raw bytes are stored in MinIO like any other upload. Parsing never calls OCR: a lean
adaptation of [magnusfroste/sie-parser](https://github.com/magnusfroste/sie-parser)
(see [NOTICE.md](../NOTICE.md)) builds balance-sheet / income-statement / transaction
summaries, rendered as Markdown tables. Those summaries are what the chunker embeds into
Qdrant — not every voucher line. `doc_type` is set to `accounting_data` and `category`
defaults to `financing` (they are separate filters — accounting_data is not a category).
Company name and year are filled from `#FNAMN` / `#RAR` when the upload form leaves them
blank.

**OCR → Markdown.** Each `pages[].markdown` becomes a `ParsedPage`. Page identity comes from
`pages[].index`, normalized to 1-based `page_number`. Empty pages are dropped and **not**
renumbered, so citations follow the source document. `table_format` is left unset so tables stay
inline as Markdown tables (the chunker treats a table as one unsplittable chunk). Relative
image placeholders such as `![img-0.jpeg](img-0.jpeg)` are stripped so they never reach the
index.

**Figures are separate chunks.** OCR returns figure bytes on `pages[].images[]` when
`include_image_base64` is on, plus a caption from bbox annotation (`short_description` /
`summary`). Each figure is stored in MinIO under
`{tenant_id}/{document_id}/figures/{uuid}.png` (or `.jpg` if the crop is JPEG). Ingest then
appends one retrieval chunk whose `raw_text` is the caption and whose metadata carries
`figure_object_key`. Text/table chunks leave that field null. Every chunk still carries the
parent `object_key` and `page_number` — that is the 1→N link back to the original file. The
crop stays in MinIO so query-time vision can fetch it; the UI Open button still uses the parent
PDF.

**Document metadata can be filled by OCR.** The same `/ocr` call includes `document_annotation`
with our `doc_type` / `category` enums plus `company_name`, `year`, and `quarter`. Form values
win; blanks are filled from the annotation. After merge, `company_name`, `doc_type`,
`category`, and `year` are required — if still missing, the job fails rather than inventing
values. The annotation window is the first 8 pages; the whole document is still OCR'd.

**Heading depth is normalized locally.** Numbered sections are re-leveled from the decimal
outline (`3.` → `##`, `2.1` → `###`) with no extra model calls. Unnumbered headings are left
unchanged.

## 3. Chunking

`Chunker.split_pages` walks the pages in order and is structure-aware rather than length-driven.
The rules exist because naive splitting destroyed exactly the content we most needed to
retrieve.

- **Code fences are stripped** if a model ever wraps Markdown (kept as a safety net).
- **Headings never become chunks of their own.** A heading-only chunk carries no information and
  wastes a retrieval slot. Instead headings accumulate into a breadcrumb
  (`Kvartalsrapport Q3 2024 > 2. Finansiell översikt`) that is prefixed onto every chunk beneath
  them. This is what makes an otherwise context-free table of numbers findable.
- **Markdown tables are never split.** A table block becomes exactly one chunk, with its
  breadcrumb prefix.
- **Paragraphs under the same heading are merged** until `chunk_size` is reached, so chunks are
  meaningful units instead of single sentences. Oversized blocks fall back to character
  splitting that prefers a paragraph break past the halfway point, with `chunk_overlap`
  characters carried into the next chunk.

- **Headings survive the page break.** The breadcrumb is threaded from one page into the next, so
  a table that starts on page 4 under a heading printed on page 3 still carries that heading.
  Chunking per page without this would trade a correct page number for lost context.

- **Figure chunks are not split.** Each figure is one chunk with `figure_object_key` set.

Each chunk records the page it came from. Defaults are `chunk_size=800` and `chunk_overlap=100`.

## 4. Linking

`build_payloads` assigns each chunk a UUID and wires `previous_chunk_id` and `next_chunk_id`
into a double-linked list across the document. Retrieval uses these links to pull adjacent
context for tables and clauses that continue past a chunk boundary.

## 5. Qdrant payload schema

Every vector point carries this payload. The `metadata.*` fields are what filtering depends on.

```json
{
    "raw_text": "...the actual text chunk...",
    "metadata": {
        "tenant_id": "default_tenant_test_corp_123",
        "company_name": "SME Corp AB",
        "doc_type": "financial_report",
        "category": "financing",
        "year": 2024,
        "quarter": 3,
        "source_filename": "q3_report_2024.pdf",
        "object_key": "default_tenant_test_corp_123/uuid-doc/q3_report_2024.pdf",
        "content_hash": "9dacf0e31040b708...",
        "figure_object_key": null,
        "page_number": 14,
        "chunk_id": "uuid123",
        "previous_chunk_id": "uuid122",
        "next_chunk_id": "uuid124",
        "language": "sv",
        "processed_timestamp": "2026-08-09T14:30:00Z"
    }
}
```

Valid enums:

- `doc_type`: `agreement`, `financial_report`, `minutes`, `org_chart`, `accounting_data`,
  `presentation`, `report_analysis`, `policy_regulation`, `transaction_doc`, `other`
- `category`: `hr`, `financing`, `legal`, `operations`, `compliance_governance`, `m_and_a`,
  `strategy_market`, `other`

`app/scripts/init_qdrant.py` creates the collection with cosine distance and payload indexes on
`tenant_id`, `doc_type`, `category`, `year`, `quarter`, `company_name` and `source_filename`.

## 6. Embedding and upsert

Chunk texts are embedded in one batch with `text-embedding-3-small` (1536 dimensions) and
upserted with the chunk UUID as the point id.

Before upserting, the previous version of the document is deleted. The delete matches on
`tenant_id` and then on *either* the filename or the content hash, so a document that was edited
keeps its name and replaces itself, and a document that was renamed is still recognized by its
content. Ingest is therefore idempotent per `(tenant_id, source_filename)` and per
`(tenant_id, content_hash)` rather than appending duplicates.

## Schema changes require re-ingestion

`ChunkMetadata` is validated on read. Adding a required field makes every previously stored
chunk fail validation and `/query` returns 500 until the affected documents are ingested again.
Either re-ingest, or add the field as optional.

`content_hash` was added as optional for exactly this reason. Chunks written before hashing keep
validating, they simply carry `null` and therefore never match a duplicate check — the first
re-ingest gives them a hash.

`figure_object_key` is optional for the same reason. Text and table chunks leave it null.
