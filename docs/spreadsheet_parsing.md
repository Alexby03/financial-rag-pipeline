# Spreadsheet parsing

How `.xlsx` / `.xlsm` workbooks become Markdown for chunking and retrieval. This is
local-only (no OCR, no Unstructured/Azure Document Intelligence). The general ingest
path is in [ingestion_pipeline.md](ingestion_pipeline.md); per-format overview is in
[document_formats.md](document_formats.md).

Entry point: `app/services/spreadsheet/` via `parse_xlsx_bytes` (re-exported from
`app/services/xlsx.py`).

## Why Excel is special

Financial models are not “one table per sheet.” A typical DCF sheet has:

- A **year grid** on the left (`Aberlyn (USDm) | 1993 | … | 2004`)
- **General Assumptions** as a narrow key/value block under a banner a few rows above
- **Equity Value Sensitivity** as a multi-cube matrix on the right (same top row as the year grid)
- Blank rows that visually group sections but **split connected components**
- Display formats (`#,##0`) that round values Excel shows as `393` while the cached float is `392.679…`

Similarity search cannot invent sheet inventory or fix a smashed table. Structure has to
be recovered at parse time.

## Pipeline

```
xlsx bytes
  → loader (openpyxl, data_only + formula fallback)
  → string grid per sheet (format-aware floats)
  → connected components (4-connected populated cells)
  → vertical stitch (blank-row continuations)
  → absorb contained crumbs into host (paint, then drop)
  → side-by-side matrix merge (sensitivity cubes + Terminal growth)
  → title merge / orphan banner attach (General Assumptions, Equity Value Sensitivity)
  → note-strip glue (comments to the right of a table)
  → split adjacent KV columns (value|label|value glue)
  → row-local KV fill (wide-gap label→value + ! notes)
  → render (fiscal Post|year tables + literal grids + Rows: narratives)
  → one ParsedPage per sheet (carry_page_context=False)
```

### Loader

- Prefer `data_only=True` so formulas become cached values.
- If the workbook has no cached values, reopen without `data_only` (formulas are skipped as amounts).
- Floats use the cell’s **Excel number format** for display precision (see below).

### Structure detection

| Step | Purpose |
|------|---------|
| Connected components | Split islands separated by blank rows/columns |
| Vertical stitch | Rejoin year-grid slices (revenue then D&A) and sensitivity header→values |
| Absorb contained panels | Paint interior scalars (e.g. `12.8%` next to `Discount Rate:`) into the host cube, then drop crumbs so hstack still works |
| Side-by-side merge | Join three Discount Rate cubes + Terminal growth label strip |
| Absorb value crumbs | Rejoin labels with `14.32` / `Yes`/`No` when blank columns break 4-connectivity |
| Drop label-less dumps | Do not emit answer-only islands (`Yes`/`No`/`351.77`…) as their own tables |
| Split adjacent KV | Break glued `value \| label \| value` from two side-by-side form columns (Options warrants) |
| Row-local KV fill | Same-row label → nearest value within 8 cols (+ `!` note); matrix hstack gaps stay tight |
| Orphan titles | Attach `General Assumptions` / `Equity Value Sensitivity` to the right table band |
| Note strips | Glue weak right-hand comments onto the left table only |

**Key/value vs matrix:** Damodaran input forms are repeating `label → value → !note` columns.
Wide blank gaps and touching dual columns are handled by the KV passes above. Fiscal year
grids and Equity Value Sensitivity still use the **tight** matrix merge rules — do not raise
`_MAX_TITLE_COL_GAP` to “fix” form layouts.

**Hard rules that bite in production:**

1. A banner must not attach to a table **entirely to its left** (Equity title ≠ D&A rows).
2. Prefer the **largest** matrix under a banner (not a header-only stub).
3. Do not vertically stitch two full year-header tables stacked with a blank row (P&L vs FCF).
4. Do not hstack a year grid onto a sensitivity matrix.
5. Chunker must not stamp one markdown table’s year headers onto the next table on the same sheet.

### Rendering

- **Fiscal year grids** (period labels in the header): `Post | 1993 | …` plus optional
  `Rows:` narratives when period→column mapping is confident.
- **Everything else** (sensitivity cubes, assumptions): literal cell grid — no fake `Col2` headers.
- Section titles come from banners (`## Equity Value Sensitivity`), never from the first data row (`Terminal tax`).

## Display floats vs stored floats

Not a pandas bug. Excel stores IEEE floats; the UI rounds via `number_format`.

| | |
|--|--|
| Cached value | `392.6792663085469` |
| `number_format` | `#,##0` |
| What Excel shows | `393` |
| Old parser output | `392.679266` |

`loader._format_float` rounds using the format’s decimal places (`#,##0` → 0 places → `393`).
General / unknown formats still trim trailing zeros with a modest default.

## Accepted extensions

| Extension | Supported? |
|-----------|------------|
| `.xlsx` | Yes |
| `.xlsm` | Yes (macros ignored; values/cached results only) |
| `.xls` (legacy BIFF) | **No** — not in `XLSX_EXTENSIONS`; Streamlit uploader only lists `xlsx` |

Adding `.xls` would need another engine (e.g. `xlrd` or Calamine) and a separate path;
keep it out of the MVP unless a real corpus requires it.

## Chunking / retrieval notes

- One Markdown **page per sheet**; `carry_page_context=False` so sheet N cannot inherit
  `Financials > …` from sheet N−1.
- Within a sheet, markdown tables use **local** period headers only — a DCF year grid must
  not inject `1993|1994|…` into Equity Value Sensitivity on the same page.
- Inventory questions (“list every DCF case”) still need a workbook outline artifact;
  similarity search over table chunks will not guarantee full sheet coverage. See the
  product discussion on ingest-time sheet inventory.

## Tests worth keeping green

- Side-by-side Income / Balance sheets
- Banner + assumptions with a vertical gap (`General Assumptions`)
- Equity Value Sensitivity must not contain D&A / year-grid rows
- Stacked P&L vs FCF stay separate tables
- Note strips without a year grid stay with the left table

## Module map

| File | Role |
|------|------|
| `app/services/spreadsheet/loader.py` | Bytes → grids, format-aware floats |
| `app/services/spreadsheet/structure.py` | Components, merges, titles |
| `app/services/spreadsheet/kv.py` | KV column roles, dual-block split, row-local fill |
| `app/services/spreadsheet/render.py` | Markdown + row narratives |
| `app/services/spreadsheet/periods.py` | Year / ISO period tokens |
| `app/services/spreadsheet/__init__.py` | `parse_xlsx_bytes` |
| `app/services/xlsx.py` | Thin re-export shim |
