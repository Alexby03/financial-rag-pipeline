# Adapted from https://github.com/magnusfroste/sie-parser
# Copyright (c) 2025 Magnus Froste — MIT License. See LICENSE in this directory.
# Lean adaptation for rag-system-mvp (debug prints and demo UI helpers removed).

from __future__ import annotations

from collections import defaultdict
from typing import Any


def process_for_llm(sie_data: dict[str, Any]) -> dict[str, Any]:
    """
    Process SIE data for LLM analysis.

    Builds a summary, balance sheet, income statement, and transaction
    samples or aggregates when the full set is large.
    """
    transaction_count = count_transactions(sie_data)
    processed: dict[str, Any] = {
        "metadata": sie_data.get("metadata", {}),
        "summary": create_summary(sie_data),
        "accounts": sie_data.get("accounts", {}),
        "balance_sheet": sie_data.get("balance_sheet")
        or create_balance_sheet(sie_data),
        "income_statement": sie_data.get("income_statement")
        or create_income_statement(sie_data),
        "transaction_samples": sample_transactions(sie_data, max_samples=20),
        "transaction_count": transaction_count,
        "opening_balances": sie_data.get("opening_balances")
        or sie_data.get("ib", {}),
        "closing_balances": sie_data.get("closing_balances")
        or sie_data.get("ub", {}),
        "results": sie_data.get("results") or sie_data.get("res", {}),
    }

    if transaction_count <= 500:
        processed["all_transactions"] = sie_data.get("verifications", [])
    else:
        processed["transaction_aggregates"] = aggregate_transactions(sie_data)

    return processed


def create_summary(sie_data: dict[str, Any]) -> dict[str, Any]:
    """Create a summary of the SIE data."""
    metadata = sie_data.get("metadata", {})
    income_total, expense_total = calculate_income_expense_totals(sie_data)
    return {
        "company_name": metadata.get("company_name", "Unknown"),
        "period": get_period_string(sie_data),
        "total_accounts": len(sie_data.get("accounts", {})),
        "total_verifications": len(sie_data.get("verifications", [])),
        "total_transactions": count_transactions(sie_data),
        "account_types": count_account_types(sie_data),
        "total_income": income_total,
        "total_expenses": expense_total,
        "net_result": income_total - expense_total,
    }


def get_period_string(sie_data: dict[str, Any]) -> str:
    """Get a string representation of the accounting period."""
    metadata = sie_data.get("metadata", {})
    start = metadata.get("financial_year_start", "")
    end = metadata.get("financial_year_end", "")
    if start and end:
        return f"{start} to {end}"

    years = metadata.get("fiscal_years", {})
    if years:
        year_data = next(iter(years.values()))
        start = year_data.get("start_date", "Unknown")
        end = year_data.get("end_date", "Unknown")
        return f"{start} to {end}"

    return "Unknown period"


def count_transactions(sie_data: dict[str, Any]) -> int:
    """Count the total number of transactions."""
    count = 0
    for ver in sie_data.get("verifications", []):
        transactions = (
            ver.get("transactions", [])
            if isinstance(ver, dict)
            else getattr(ver, "transactions", [])
        )
        count += len(transactions or [])
    return count


def count_account_types(sie_data: dict[str, Any]) -> dict[str, int]:
    """Count accounts by type."""
    type_counts: dict[str, int] = defaultdict(int)
    for account_data in sie_data.get("accounts", {}).values():
        if isinstance(account_data, dict):
            account_type = account_data.get("type", "unknown")
        else:
            account_type = getattr(account_data, "type", "unknown")
        type_counts[account_type or "unknown"] += 1
    return dict(type_counts)


def calculate_income_expense_totals(sie_data: dict[str, Any]) -> tuple[float, float]:
    """Calculate total income and expenses from RES or income statement."""
    income_statement = sie_data.get("income_statement")
    if income_statement:
        return (
            float(income_statement.get("total_income", 0.0) or 0.0),
            float(income_statement.get("total_expenses", 0.0) or 0.0),
        )

    income_total = 0.0
    expense_total = 0.0
    results = sie_data.get("results") or sie_data.get("res", {})
    accounts = sie_data.get("accounts", {})

    year = next(iter(results), None)
    if year is None:
        return income_total, expense_total

    for account, entry in results[year].items():
        amount = entry.get("amount", entry) if isinstance(entry, dict) else entry
        try:
            amount_f = float(amount)
        except (TypeError, ValueError):
            continue
        try:
            account_int = int(account)
        except ValueError:
            continue

        if account not in accounts:
            continue
        if 3000 <= account_int < 4000:
            income_total -= amount_f
        elif 4000 <= account_int < 8000:
            expense_total += amount_f

    return income_total, expense_total


def create_balance_sheet(sie_data: dict[str, Any]) -> dict[str, Any]:
    """Create a balance sheet from closing balances (UB)."""
    balance_sheet: dict[str, Any] = {
        "assets": {},
        "liabilities": {},
        "equity": {},
    }

    closing = sie_data.get("closing_balances") or sie_data.get("ub", {})
    accounts = sie_data.get("accounts", {})
    year = next(iter(closing), None)

    if year is not None:
        for account, entry in closing[year].items():
            if account not in accounts:
                continue
            amount = entry.get("amount", entry) if isinstance(entry, dict) else entry
            try:
                amount_f = float(amount)
            except (TypeError, ValueError):
                continue

            account_info = accounts[account]
            account_type = (
                account_info.get("type", "")
                if isinstance(account_info, dict)
                else getattr(account_info, "type", "")
            )
            account_name = (
                account_info.get("name", "Unknown")
                if isinstance(account_info, dict)
                else getattr(account_info, "name", "Unknown")
            )

            item = {"name": account_name, "amount": amount_f}
            if account_type in ("asset", "Asset"):
                balance_sheet["assets"][account] = item
            elif account_type in (
                "liability",
                "equity_or_liability",
                "Liability/Equity",
            ):
                try:
                    account_int = int(account)
                except ValueError:
                    account_int = 0
                if account_int < 2900:
                    balance_sheet["liabilities"][account] = item
                else:
                    balance_sheet["equity"][account] = item

    balance_sheet["total_assets"] = sum(
        item["amount"] for item in balance_sheet["assets"].values()
    )
    balance_sheet["total_liabilities"] = sum(
        item["amount"] for item in balance_sheet["liabilities"].values()
    )
    balance_sheet["total_equity"] = sum(
        item["amount"] for item in balance_sheet["equity"].values()
    )
    return balance_sheet


def create_income_statement(sie_data: dict[str, Any]) -> dict[str, Any]:
    """Create an income statement from result (RES) data."""
    income_statement: dict[str, Any] = {"income": {}, "expenses": {}}
    results = sie_data.get("results") or sie_data.get("res", {})
    accounts = sie_data.get("accounts", {})
    year = next(iter(results), None)

    if year is not None:
        for account, entry in results[year].items():
            if account not in accounts:
                continue
            amount = entry.get("amount", entry) if isinstance(entry, dict) else entry
            try:
                amount_f = float(amount)
                account_int = int(account)
            except (TypeError, ValueError):
                continue

            account_info = accounts[account]
            account_name = (
                account_info.get("name", "Unknown")
                if isinstance(account_info, dict)
                else getattr(account_info, "name", "Unknown")
            )

            if 3000 <= account_int < 4000:
                income_statement["income"][account] = {
                    "name": account_name,
                    "amount": -amount_f,
                }
            elif 4000 <= account_int < 8000:
                income_statement["expenses"][account] = {
                    "name": account_name,
                    "amount": amount_f,
                }

    income_statement["total_income"] = sum(
        item["amount"] for item in income_statement["income"].values()
    )
    income_statement["total_expenses"] = sum(
        item["amount"] for item in income_statement["expenses"].values()
    )
    income_statement["net_result"] = (
        income_statement["total_income"] - income_statement["total_expenses"]
    )
    return income_statement


def sample_transactions(
    sie_data: dict[str, Any], max_samples: int = 20
) -> list[dict[str, Any]]:
    """Sample a limited number of transactions for LLM analysis."""
    all_transactions: list[dict[str, Any]] = []

    for ver in sie_data.get("verifications", []):
        if isinstance(ver, dict):
            series = ver.get("series", "")
            number = ver.get("number", "")
            ver_text = ver.get("text", "")
            transactions = ver.get("transactions", [])
        else:
            series = getattr(ver, "series", "")
            number = getattr(ver, "number", "")
            ver_text = getattr(ver, "text", "")
            transactions = getattr(ver, "transactions", [])

        for trans in transactions or []:
            if isinstance(trans, dict):
                all_transactions.append(
                    {
                        "verification": f"{series}{number}",
                        "date": trans.get("date", ""),
                        "account": trans.get("account", ""),
                        "account_name": trans.get("account_name", "Unknown"),
                        "amount": trans.get("amount", 0.0),
                        "text": trans.get("text") or ver_text,
                    }
                )
            else:
                all_transactions.append(
                    {
                        "verification": f"{series}{number}",
                        "date": getattr(trans, "date", ""),
                        "account": getattr(trans, "account", ""),
                        "account_name": getattr(trans, "account_name", "Unknown"),
                        "amount": getattr(trans, "amount", 0.0),
                        "text": getattr(trans, "text", "") or ver_text,
                    }
                )

    if len(all_transactions) <= max_samples:
        return all_transactions

    sampled: list[dict[str, Any]] = []
    step = max(1, len(all_transactions) // max_samples)
    for i in range(0, len(all_transactions), step):
        if len(sampled) >= max_samples:
            break
        sampled.append(all_transactions[i])
    return sampled


def aggregate_transactions(sie_data: dict[str, Any]) -> dict[str, Any]:
    """Aggregate transactions by account and month for large datasets."""
    aggregates: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    accounts = sie_data.get("accounts", {})

    for ver in sie_data.get("verifications", []):
        transactions = (
            ver.get("transactions", [])
            if isinstance(ver, dict)
            else getattr(ver, "transactions", [])
        )
        for trans in transactions or []:
            if isinstance(trans, dict):
                account = trans.get("account", "")
                amount = float(trans.get("amount", 0.0) or 0.0)
                date = trans.get("date", "") or ""
            else:
                account = getattr(trans, "account", "")
                amount = float(getattr(trans, "amount", 0.0) or 0.0)
                date = getattr(trans, "date", "") or ""

            month_key = date[:6] if len(date) >= 6 else "unknown"
            aggregates[account][month_key] += amount

    result: dict[str, Any] = {}
    for account, months in aggregates.items():
        account_name = "Unknown"
        if account in accounts:
            info = accounts[account]
            account_name = (
                info.get("name", "Unknown")
                if isinstance(info, dict)
                else getattr(info, "name", "Unknown")
            )
        result[account] = {
            "name": account_name,
            "monthly_totals": dict(months),
            "total": sum(months.values()),
        }
    return result
