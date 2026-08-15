"""SIE / SI / SE accounting files → Markdown summaries for chunking.

Raw bytes stay in MinIO; only these summaries are embedded into Qdrant.
Parsing uses the lean vendored magnusfroste sie-parser (see NOTICE.md).
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import Any

from app.models.enums import Category, DocType
from app.models.ocr import OcrDocumentAnnotation
from app.models.parsed import ParsedContent, ParsedPage
from third_party.magnusfroste_sie_parser import SIEParser, process_for_llm

SIE_EXTENSIONS = {".sie", ".si", ".se"}


def is_sie_filename(filename: str) -> bool:
    return Path(filename).suffix.lower() in SIE_EXTENSIONS


def parse_sie_bytes(file_bytes: bytes) -> ParsedContent:
    """Parse SIE4 bytes into Markdown summary pages plus metadata annotation."""
    if not file_bytes:
        raise ValueError("SIE file is empty")

    with tempfile.NamedTemporaryFile(suffix=".se", delete=False) as handle:
        handle.write(file_bytes)
        temp_path = handle.name

    try:
        try:
            sie_data = SIEParser(temp_path).parse()
        except Exception as error:
            raise ValueError(f"Failed to parse SIE file: {error}") from error
        if sie_data is None:
            raise ValueError(
                "Failed to parse SIE file. The file may be corrupted or unsupported."
            )

        try:
            llm_data = process_for_llm(sie_data)
        except Exception as error:
            raise ValueError(f"Failed to summarize SIE file: {error}") from error

        pages = _markdown_pages(llm_data)
        if not pages:
            raise ValueError("SIE parse produced no summary content")

        return ParsedContent(
            pages=pages,
            content_format="markdown",
            annotation=_annotation_from_sie(llm_data),
        )
    finally:
        Path(temp_path).unlink(missing_ok=True)


def _annotation_from_sie(llm_data: dict[str, Any]) -> OcrDocumentAnnotation:
    """Fill ingest blanks from SIE tags.

    `accounting_data` is a doc_type, not a category. SIE defaults category to
    financing so uploads do not require the form when the ledger parses cleanly.
    """
    metadata = llm_data.get("metadata") or {}
    summary = llm_data.get("summary") or {}
    company = _filled(metadata.get("company_name")) or _filled(summary.get("company_name"))
    # Never use #PROGRAM version (e.g. Norstedts "2011.2.1") as the fiscal year.
    year = (
        _year_from_metadata(metadata)
        or _year_from_summary(summary)
        or _year_from_transactions(llm_data)
    )
    return OcrDocumentAnnotation(
        company_name=company,
        doc_type=DocType.ACCOUNTING_DATA,
        category=Category.FINANCING,
        year=year,
    )


def _year_from_metadata(metadata: dict[str, Any]) -> int | None:
    for key in (
        "rar",
        "taxar",
        "omfattn",
        "current_fiscal_year_start_year",
        "current_fiscal_year_end_year",
    ):
        parsed = _year_from_value(metadata.get(key))
        if parsed is not None:
            return parsed

    for key in (
        "financial_year_start",
        "financial_year_end",
        "fiscal_year",
        "generation_date",
        "gen_date",
    ):
        parsed = _year_from_value(metadata.get(key))
        if parsed is not None:
            return parsed

    fiscal = metadata.get("current_fiscal_year") or {}
    if isinstance(fiscal, dict):
        for key in ("start_date", "end_date"):
            parsed = _year_from_value(fiscal.get(key))
            if parsed is not None:
                return parsed

    years = metadata.get("fiscal_years") or {}
    if isinstance(years, dict):
        # Prefer current year id "0", then any other #RAR block.
        ordered = []
        if "0" in years:
            ordered.append(years["0"])
        ordered.extend(value for key, value in years.items() if key != "0")
        for year_data in ordered:
            if not isinstance(year_data, dict):
                continue
            for key in ("end_date", "start_date"):
                parsed = _year_from_value(year_data.get(key))
                if parsed is not None:
                    return parsed
    return None


def _year_from_summary(summary: dict[str, Any]) -> int | None:
    return _year_from_value(summary.get("period"))


def _year_from_transactions(llm_data: dict[str, Any]) -> int | None:
    """Fall back to the latest YYYYMMDD date on vouchers / samples."""
    years: list[int] = []
    for sample in llm_data.get("transaction_samples") or []:
        parsed = _year_from_value(sample.get("date") if isinstance(sample, dict) else None)
        if parsed is not None:
            years.append(parsed)
    for verification in llm_data.get("all_transactions") or llm_data.get("verifications") or []:
        if not isinstance(verification, dict):
            continue
        parsed = _year_from_value(verification.get("date"))
        if parsed is not None:
            years.append(parsed)
        for transaction in verification.get("transactions") or []:
            if not isinstance(transaction, dict):
                continue
            parsed = _year_from_value(transaction.get("date"))
            if parsed is not None:
                years.append(parsed)
    return max(years) if years else None


def _year_from_value(value: Any) -> int | None:
    """Pick a calendar year from SIE dates like 20241231 or 2024-12-31.

    Ignores dotted software versions such as 2011.2.1 from #PROGRAM.
    """
    if value is None:
        return None
    text = str(value).strip()
    if re.fullmatch(r"\d+\.\d+(\.\d+)*", text):
        return None
    for index in range(len(text) - 3):
        chunk = text[index : index + 4]
        if chunk.isdigit():
            year = int(chunk)
            if 1900 <= year <= 2100:
                return year
    return None


def _markdown_pages(llm_data: dict[str, Any]) -> list[ParsedPage]:
    sections: list[str] = []
    summary = _render_summary(llm_data)
    if summary:
        sections.append(summary)
    balance = _render_balance_sheet(llm_data.get("balance_sheet") or {})
    if balance:
        sections.append(balance)
    income = _render_income_statement(llm_data.get("income_statement") or {})
    if income:
        sections.append(income)
    transactions = _render_transactions(llm_data)
    if transactions:
        sections.append(transactions)

    pages: list[ParsedPage] = []
    for index, section in enumerate(sections, start=1):
        pages.append(ParsedPage(text=section, page_number=index))
    return pages


def _render_summary(llm_data: dict[str, Any]) -> str:
    metadata = llm_data.get("metadata") or {}
    summary = llm_data.get("summary") or {}
    company = summary.get("company_name") or metadata.get("company_name") or "Unknown"
    lines = [
        "# Accounting summary",
        "",
        f"- Company: {company}",
    ]
    org = metadata.get("organization_number") or metadata.get("org_number")
    if org:
        lines.append(f"- Organization number: {org}")
    period = summary.get("period") or "Unknown period"
    lines.append(f"- Period: {period}")
    if summary.get("total_accounts") is not None:
        lines.append(f"- Accounts: {summary['total_accounts']}")
    if summary.get("total_verifications") is not None:
        lines.append(f"- Verifications: {summary['total_verifications']}")
    if summary.get("total_transactions") is not None:
        lines.append(f"- Transactions: {summary['total_transactions']}")
    if summary.get("total_income") is not None:
        lines.append(f"- Total income: {_fmt_amount(summary['total_income'])}")
    if summary.get("total_expenses") is not None:
        lines.append(f"- Total expenses: {_fmt_amount(summary['total_expenses'])}")
    if summary.get("net_result") is not None:
        lines.append(f"- Net result: {_fmt_amount(summary['net_result'])}")
    return "\n".join(lines)


def _render_balance_sheet(balance_sheet: dict[str, Any]) -> str:
    assets = balance_sheet.get("assets") or {}
    liabilities = balance_sheet.get("liabilities") or {}
    equity = balance_sheet.get("equity") or {}
    if not assets and not liabilities and not equity:
        return ""

    lines = ["# Balance sheet", ""]
    lines.extend(_account_table("Assets", assets, amount_keys=("amount", "balance")))
    lines.append("")
    lines.extend(
        _account_table("Liabilities", liabilities, amount_keys=("amount", "balance"))
    )
    lines.append("")
    lines.extend(_account_table("Equity", equity, amount_keys=("amount", "balance")))

    total_assets = balance_sheet.get("total_assets")
    if total_assets is not None:
        lines.extend(["", f"Total assets: {_fmt_amount(total_assets)}"])
    total_le = balance_sheet.get("total_liabilities_equity")
    if total_le is None:
        total_liabilities = balance_sheet.get("total_liabilities")
        total_equity = balance_sheet.get("total_equity")
        if total_liabilities is not None or total_equity is not None:
            total_le = (total_liabilities or 0) + (total_equity or 0)
    if total_le is not None:
        lines.append(f"Total liabilities and equity: {_fmt_amount(total_le)}")
    return "\n".join(lines).strip()


def _render_income_statement(income_statement: dict[str, Any]) -> str:
    income = income_statement.get("income") or {}
    expenses = income_statement.get("expenses") or {}
    if not income and not expenses:
        return ""

    lines = ["# Income statement", ""]
    lines.extend(_account_table("Income", income, amount_keys=("amount", "balance")))
    lines.append("")
    lines.extend(_account_table("Expenses", expenses, amount_keys=("amount", "balance")))

    total_income = income_statement.get("total_income")
    total_expenses = income_statement.get("total_expenses")
    net = income_statement.get("net_result")
    if net is None:
        net = income_statement.get("net_income")
    if total_income is not None:
        lines.extend(["", f"Total income: {_fmt_amount(total_income)}"])
    if total_expenses is not None:
        lines.append(f"Total expenses: {_fmt_amount(total_expenses)}")
    if net is not None:
        lines.append(f"Net result: {_fmt_amount(net)}")
    return "\n".join(lines).strip()


def _render_transactions(llm_data: dict[str, Any]) -> str:
    aggregates = llm_data.get("transaction_aggregates")
    if aggregates:
        return _render_aggregates(aggregates)

    samples = llm_data.get("transaction_samples") or []
    if not samples:
        return ""

    lines = [
        "# Transaction samples",
        "",
        "| Verification | Date | Account | Name | Amount | Text |",
        "| --- | --- | --- | --- | ---: | --- |",
    ]
    for sample in samples:
        lines.append(
            "| {ver} | {date} | {account} | {name} | {amount} | {text} |".format(
                ver=_cell(sample.get("verification")),
                date=_cell(sample.get("date")),
                account=_cell(sample.get("account")),
                name=_cell(sample.get("account_name")),
                amount=_fmt_amount(sample.get("amount")),
                text=_cell(sample.get("text")),
            )
        )
    count = llm_data.get("transaction_count")
    if count is not None:
        lines.extend(["", f"Total transactions in file: {count}"])
    return "\n".join(lines)


def _render_aggregates(aggregates: dict[str, Any]) -> str:
    lines = [
        "# Transaction aggregates by account",
        "",
        "| Account | Name | Total |",
        "| --- | --- | ---: |",
    ]
    for account, payload in aggregates.items():
        if isinstance(payload, dict):
            name = payload.get("name", "")
            total = payload.get("total", 0)
        else:
            name = ""
            total = payload
        lines.append(
            f"| {_cell(account)} | {_cell(name)} | {_fmt_amount(total)} |"
        )
    return "\n".join(lines)


def _account_table(
    title: str,
    accounts: dict[str, Any],
    *,
    amount_keys: tuple[str, ...],
) -> list[str]:
    lines = [f"## {title}", "", "| Account | Name | Amount |", "| --- | --- | ---: |"]
    if not accounts:
        lines.append("| — | — | — |")
        return lines
    for account, payload in accounts.items():
        if isinstance(payload, dict):
            name = payload.get("name", "")
            amount = next(
                (payload[key] for key in amount_keys if key in payload),
                0,
            )
        else:
            name = ""
            amount = payload
        lines.append(f"| {_cell(account)} | {_cell(name)} | {_fmt_amount(amount)} |")
    return lines


def _fmt_amount(value: Any) -> str:
    try:
        return f"{float(value):,.2f}"
    except (TypeError, ValueError):
        return _cell(value)


def _cell(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


def _filled(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
