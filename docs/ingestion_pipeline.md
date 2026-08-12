# Ingestion pipeline

How a document becomes searchable. The retrieval side is described in
[rag_strategy.md](rag_strategy.md).

Entry point: `POST /ingest` → `IngestService.ingest`.

```
file bytes → hash → MinIO → parse → chunk → link → embed → Qdrant
```

## 1. Identity, then storage

The first thing ingest does is hash the bytes. A sha256 digest of the file is the document's
identity, and it is checked against the index before anything expensive happens: if the same
tenant already has chunks carrying that hash, ingest returns immediately with
`status="duplicate"`, the existing `object_key` and the existing chunk count. Nothing is
uploaded, parsed, embedded or written.

This matters because parsing is by far the most expensive step, and filenames are a terrible
identity. `report.pdf`, `report (1).pdf` and `@report.pdf` are three names for one document, and
before hashing each one produced its own complete set of chunks — paid for twice and retrieved
twice, competing for slots in the same answer.

Once a document is known to be new, it is stored before it is parsed, so the original is never
lost if parsing fails.

- `document_id` is a fresh UUID per ingest.
- `object_key` is `{tenant_id}/{document_id}/{filename}` and is written into every chunk's
  metadata. Without it there is no path from a retrieved chunk back to the source file.
- `content_hash` is written into every chunk's metadata alongside it.
- Content type is guessed from the filename so MinIO serves the file correctly when the
  frontend later opens it.

## 2. Multimodal parsing

`DocumentParser` picks a strategy from the MIME type and `doc_type`. Output is normalized to
`ParsedContent`: a list of `ParsedPage`, each with its text and its 1-based page number, in
either Markdown or JSON. Inputs that have no pagination — images, plain text, org charts —
produce a single page numbered 1.

| Input | Strategy | Output |
|-------|----------|--------|
| `org_chart` (any image) | gpt-4o Vision, described as structured JSON | JSON |
| PDF | rasterized per page, then gpt-4o Vision | Markdown |
| Images | gpt-4o Vision | Markdown |
| Plain text, financial doc types | gpt-4o text completion | Markdown |
| Plain text, other doc types | passed through verbatim | Markdown |

Financial doc types are `financial_report`, `agreement` and `minutes`. These skip the verbatim
path because their tables need to be reconstructed as Markdown tables rather than kept as
whitespace-aligned text.

**PDFs must be rasterized.** The chat completions endpoint rejects `application/pdf` as image
input and answers 400. `OpenAIService.pdf_pages_to_png_base64` opens the PDF with PyMuPDF and
renders every page to PNG at 150 DPI.

**Each page is its own request.** The page number is only knowable while the pages are still
separate; merging them into one Markdown document destroys it, and a citation that always reads
"p.1" is worse than useless because it looks trustworthy. So the parser sends one image per
request and keeps the index. The cost is latency — the requests are sequential, so page count
drives ingest time as directly as it drives token cost. A page that comes back empty is dropped
rather than stored, which leaves a gap in the numbering instead of shifting every later page.

The parsing prompt asks for faithful extraction rather than summarization, and requires
financial tables to use Markdown table syntax with correct columns and rows.

**Refusals must never reach the index.** Vision models decline signature pages, which are
standard in agreements and reports. The prompt therefore tells the model to transcribe the
printed text and mark a handwritten signature as `[signatur]` without identifying the signatory.
As a backstop, `is_refusal` drops any page whose output is short and apologetic — otherwise
"I'm sorry, I can't help with that." is stored as an ordinary chunk, where it is
indistinguishable from real content to the embedder and to the reader.

**Heading depth is normalized locally.** Each page is parsed in isolation, so the model often
mis-guesses `#` depth (`### 3.` under an earlier `2.3`). After parse, `heading_normalizer`
re-levels numbered sections from the decimal outline (`3.` → `##`, `2.1` → `###`) with no extra
model calls. The per-page prompt also tells the model to reserve `#` for the title and to treat
the upload as page N of M. Unnumbered headings are left unchanged.

## 3. Chunking

`Chunker.split_pages` walks the pages in order and is structure-aware rather than length-driven.
The rules exist because naive splitting destroyed exactly the content we most needed to
retrieve.

- **Code fences are stripped.** The vision model wraps its answer in ` ```markdown `, which
  otherwise leaks into the first and last chunk.
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

Each chunk records the page it came from. Defaults are `chunk_size=800` and `chunk_overlap=100`.

JSON output (org charts) is kept whole when it fits inside `chunk_size`.

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
        "category": "Financing",
        "year": 2024,
        "quarter": 3,
        "source_filename": "q3_report_2024.pdf",
        "object_key": "default_tenant_test_corp_123/uuid-doc/q3_report_2024.pdf",
        "content_hash": "9dacf0e31040b708...",
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

- `doc_type`: `agreement`, `financial_report`, `minutes`, `org_chart`, `other`
- `category`: `HR`, `Financing`, `Legal`, `Operations`, `other`

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
