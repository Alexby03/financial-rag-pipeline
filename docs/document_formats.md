# Document formats

How each upload becomes Markdown pages (and optional figures) before chunking. The write-path
stages around parse are described in [ingestion_pipeline.md](ingestion_pipeline.md).

`DocumentParser` is the entry point. No OpenAI model reads a document at ingest time.

| Input | Strategy | Output |
|-------|----------|--------|
| PDF | Mistral OCR 4, split into 30-page slices (Foundry cap) | Markdown pages + figures |
| DOCX, PPTX, images | Mistral OCR 4 (`/ocr`, one call) | Markdown pages + figures |
| Excel (`.xlsx`, `.xlsm`) | openpyxl → Markdown tables (no OCR) | One Markdown page per sheet |
| Plain text / Markdown | Verbatim decode | Single Markdown page |
| SIE (`.sie`, `.si`, `.se`) | Lean vendored SIE 4 parser → LLM summaries (no OCR) | Markdown summary pages |
| XHTML / HTML | Headless Edge/Chrome print → PDF → OCR (30-page slices) | Markdown pages + figures |

## Mistral OCR 4

PDF, Office (DOCX/PPTX), and images go to Mistral OCR 4 on Azure AI Foundry. The OCR response
is mapped to Markdown pages plus optional figures. Plain `.txt` / `.md` stay local (verbatim
UTF-8) because they are already text.

Foundry rejects PDFs over 30 pages (`document_parser_too_many_pages`), so a long PDF is sliced
with pypdf and each slice is OCR'd separately; page numbers are mapped back onto the original
document.

## XHTML annual reports

Bolagsverket iXBRL files are one long scrollable document. Ingest detects print pages via
containers such as `rb-page`, `ar-page`, and `pagebreak_before` (or falls back to the browser
honouring CSS `@page` / `page-break-*`). Each page is printed to PDF with headless Edge or
Chrome (`CHROME_PATH` / `EDGE_PATH` override), the pages are merged, and the result reuses the
PDF OCR path — including 30-page Foundry batches. The original `.xhtml` bytes stay in MinIO;
Open still points at that source file.

## SIE accounting files

Swedish SIE 4 exports are tagged plain text (usually CP437). The raw bytes are stored in MinIO
like any other upload. Parsing never calls OCR: a lean adaptation of
[magnusfroste/sie-parser](https://github.com/magnusfroste/sie-parser)
(see [NOTICE.md](../NOTICE.md)) builds balance-sheet / income-statement / transaction
summaries, rendered as Markdown tables. Those summaries are what the chunker embeds into
Qdrant — not every voucher line. `doc_type` is set to `accounting_data` and `category`
defaults to `financing` (they are separate filters — accounting_data is not a category).
Company name and year are filled from `#FNAMN` / `#RAR` when the upload form leaves them
blank.

## Excel workbooks

`.xlsx` / `.xlsm` bypass OCR (Foundry's Office path can explode a small grid past the 128k
context). `openpyxl` reads cached values, splits each sheet on blank-row gaps (stacked tables)
and blank-column gaps (side-by-side parallel tables), and emits Markdown with year/period
columns in the header row — the same breadcrumb pattern as Bolagsverket balance sheets
(`Kolumner …` via the chunker, `Sheet > Table` headings). Note strips without their own year
grid stay attached to the table on their left. Raw bytes stay in MinIO.

## OCR → Markdown

Each `pages[].markdown` becomes a `ParsedPage`. Page identity comes from `pages[].index`,
normalized to 1-based `page_number`. Empty pages are dropped and **not** renumbered, so
citations follow the source document. `table_format` is left unset so tables stay inline as
Markdown tables (the chunker treats a table as one unsplittable chunk). Relative image
placeholders such as `![img-0.jpeg](img-0.jpeg)` are stripped so they never reach the index.

## Figures are separate chunks

OCR returns figure bytes on `pages[].images[]` when `include_image_base64` is on, plus a
caption from bbox annotation (`short_description` / `summary`). Each figure is stored in MinIO
under `{tenant_id}/{document_id}/figures/{uuid}.png` (or `.jpg` if the crop is JPEG). Ingest
then appends one retrieval chunk whose `raw_text` is the caption and whose metadata carries
`figure_object_key`. Text/table chunks leave that field null. Every chunk still carries the
parent `object_key` and `page_number` — that is the 1→N link back to the original file. The
crop stays in MinIO so query-time vision can fetch it; the UI Open button still uses the parent
PDF.

## Document metadata from OCR

The same `/ocr` call includes `document_annotation` with our `doc_type` / `category` enums
plus `company_name`, `year`, and `quarter`. Form values win; blanks are filled from the
annotation. After merge, `company_name`, `doc_type`, `category`, and `year` are required — if
still missing, the job fails rather than inventing values. The annotation window is the first
8 pages; the whole document is still OCR'd.

## Heading depth

Numbered sections are re-leveled from the decimal outline (`3.` → `##`, `2.1` → `###`) with no
extra model calls. Unnumbered headings are left unchanged.
