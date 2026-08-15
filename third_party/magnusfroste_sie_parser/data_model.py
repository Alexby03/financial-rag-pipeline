# Adapted from https://github.com/magnusfroste/sie-parser
# Copyright (c) 2025 Magnus Froste — MIT License. See LICENSE in this directory.
# Lean adaptation for rag-system-mvp (debug prints and demo UI helpers removed).

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Account:
    """Represents an account in the chart of accounts."""

    number: str
    name: str
    type: str = ""
    balance: float = 0.0
    transactions_amount: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Transaction:
    """Represents a single transaction line."""

    account: str
    amount: float
    date: str = ""
    text: str = ""
    account_name: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Verification:
    """Represents a verification (group of transactions)."""

    series: str = ""
    number: str = ""
    date: str = ""
    text: str = ""
    transactions: list[Transaction] = field(default_factory=list)
    original_number: str = ""
    original_date: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "series": self.series,
            "number": self.number,
            "date": self.date,
            "text": self.text,
            "original_number": self.original_number,
            "original_date": self.original_date,
            "transactions": [t.to_dict() for t in self.transactions],
        }


@dataclass
class BalanceEntry:
    """Represents a balance entry (opening or closing)."""

    account: str
    amount: float
    year: str = ""
    had_transactions: bool = False
    transaction_amount: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Metadata:
    """Represents metadata about the SIE file."""

    company_name: str = ""
    organization_number: str = ""
    financial_year_start: str = ""
    financial_year_end: str = ""
    generation_date: str = ""
    program: str = ""
    program_version: str = ""
    currency: str = "SEK"
    fiscal_years: dict[str, dict[str, str]] = field(default_factory=dict)
    current_fiscal_year: dict[str, str] = field(default_factory=dict)
    current_fiscal_year_start_year: str = ""
    current_fiscal_year_end_year: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SIEDataModel:
    """Standardized SIE data across bookkeeping systems."""

    def __init__(self) -> None:
        self.metadata = Metadata()
        self.accounts: dict[str, Account] = {}
        self.verifications: list[Verification] = []
        self.opening_balances: dict[str, dict[str, BalanceEntry]] = {}
        self.closing_balances: dict[str, dict[str, BalanceEntry]] = {}
        self.results: dict[str, dict[str, BalanceEntry]] = {}

    def from_parser_data(self, parser_data: dict[str, Any]) -> SIEDataModel:
        """Convert parser data to the standardized data model."""
        metadata = parser_data.get("metadata", {})
        self.metadata = Metadata(
            company_name=metadata.get("company_name", ""),
            organization_number=metadata.get("organization_number", "")
            or metadata.get("org_number", ""),
            financial_year_start=metadata.get("financial_year_start", ""),
            financial_year_end=metadata.get("financial_year_end", ""),
            generation_date=metadata.get("gen_date", "") or metadata.get("date", ""),
            program=metadata.get("program", ""),
            program_version=metadata.get("program_version", ""),
            currency=metadata.get("currency", "SEK"),
            fiscal_years=metadata.get("fiscal_years", {}),
            current_fiscal_year=metadata.get("current_fiscal_year", {}),
            current_fiscal_year_start_year=metadata.get(
                "current_fiscal_year_start_year", ""
            ),
            current_fiscal_year_end_year=metadata.get(
                "current_fiscal_year_end_year", ""
            ),
        )

        for acc_num, acc_data in parser_data.get("accounts", {}).items():
            self.accounts[acc_num] = Account(
                number=acc_num,
                name=acc_data.get("name", ""),
                type=self._determine_account_type(acc_num),
            )

        for ver_data in parser_data.get("verifications", []):
            if hasattr(ver_data, "series"):
                verification = Verification(
                    series=getattr(ver_data, "series", ""),
                    number=getattr(ver_data, "number", ""),
                    date=getattr(ver_data, "date", ""),
                    text=getattr(ver_data, "text", ""),
                    original_number=getattr(ver_data, "original_number", ""),
                    original_date=getattr(ver_data, "original_date", ""),
                )
                for trans_data in getattr(ver_data, "transactions", []) or []:
                    verification.transactions.append(
                        self._transaction_from_any(trans_data, verification.date)
                    )
            else:
                verification = Verification(
                    series=ver_data.get("series", ""),
                    number=ver_data.get("number", ""),
                    date=ver_data.get("date", ""),
                    text=ver_data.get("text", ""),
                    original_number=ver_data.get("original_number", ""),
                    original_date=ver_data.get("original_date", ""),
                )
                for trans_data in ver_data.get("transactions", []):
                    verification.transactions.append(
                        self._transaction_from_any(trans_data, verification.date)
                    )
            self.verifications.append(verification)

        for year, balances in parser_data.get("ib", {}).items():
            self.opening_balances.setdefault(year, {})
            for acc_num, amount in balances.items():
                self.opening_balances[year][acc_num] = BalanceEntry(
                    account=acc_num, amount=float(amount), year=year
                )

        for year, balances in parser_data.get("ub", {}).items():
            self.closing_balances.setdefault(year, {})
            for acc_num, amount in balances.items():
                self.closing_balances[year][acc_num] = BalanceEntry(
                    account=acc_num, amount=float(amount), year=year
                )

        for year, results in parser_data.get("res", {}).items():
            self.results.setdefault(year, {})
            if not isinstance(results, dict):
                continue
            for acc_num, amount in results.items():
                actual_amount: float
                if isinstance(amount, dict) and "amount" in amount:
                    actual_amount = float(amount["amount"])
                else:
                    try:
                        actual_amount = float(amount)
                    except (ValueError, TypeError):
                        actual_amount = 0.0
                self.results[year][acc_num] = BalanceEntry(
                    account=acc_num, amount=actual_amount, year=year
                )

        return self

    def _transaction_from_any(
        self, trans_data: Any, default_date: str
    ) -> Transaction:
        if hasattr(trans_data, "account"):
            account = getattr(trans_data, "account", "")
            transaction = Transaction(
                account=account,
                amount=float(getattr(trans_data, "amount", 0.0) or 0.0),
                date=getattr(trans_data, "date", "") or default_date,
                text=getattr(trans_data, "text", "") or "",
            )
            account_name = getattr(trans_data, "account_name", "") or ""
            if account_name:
                transaction.account_name = account_name
            elif account in self.accounts:
                transaction.account_name = self.accounts[account].name
            return transaction

        account = trans_data.get("account", "")
        transaction = Transaction(
            account=account,
            amount=float(trans_data.get("amount", 0.0) or 0.0),
            date=trans_data.get("date", "") or default_date,
            text=trans_data.get("text", "") or "",
        )
        account_name = trans_data.get("account_name", "") or ""
        if account_name:
            transaction.account_name = account_name
        elif account in self.accounts:
            transaction.account_name = self.accounts[account].name
        return transaction

    def _determine_account_type(self, account_number: str) -> str:
        """Map account number to BAS account type."""
        if not account_number or not account_number[0].isdigit():
            return ""

        first_digit = int(account_number[0])
        if first_digit == 1:
            return "Asset"
        if first_digit == 2:
            return "Liability/Equity"
        if first_digit == 3:
            return "Income"
        if 4 <= first_digit <= 8:
            return "Expense"
        return "Other"

    def calculate_account_balances(self) -> None:
        """Compute balances from opening balances and current-year transactions."""
        current_year = (
            self.metadata.financial_year_start[:4]
            if self.metadata.financial_year_start
            else ""
        )

        for account in self.accounts.values():
            account.balance = 0.0
            account.transactions_amount = 0.0

        opening_balance_year = next(iter(self.opening_balances), None)
        if opening_balance_year is not None:
            for acc_num, balance_entry in self.opening_balances[
                opening_balance_year
            ].items():
                if acc_num in self.accounts:
                    self.accounts[acc_num].balance = balance_entry.amount

        accounts_with_transactions: set[str] = set()
        for verification in self.verifications:
            if current_year and verification.date and not verification.date.startswith(
                current_year
            ):
                continue
            for transaction in verification.transactions:
                acc_num = transaction.account
                if acc_num in self.accounts:
                    self.accounts[acc_num].balance += transaction.amount
                    self.accounts[acc_num].transactions_amount += transaction.amount
                    accounts_with_transactions.add(acc_num)

        year_key = current_year or "0"
        self.closing_balances.setdefault(year_key, {})
        for acc_num, account in self.accounts.items():
            if account.balance != 0:
                balance_entry = BalanceEntry(
                    account=acc_num, amount=account.balance, year=year_key
                )
                balance_entry.had_transactions = acc_num in accounts_with_transactions
                balance_entry.transaction_amount = account.transactions_amount
                self.closing_balances[year_key][acc_num] = balance_entry

    def get_balance_sheet(self) -> dict[str, Any]:
        """Generate a balance sheet from the data model."""
        self.calculate_account_balances()

        balance_sheet: dict[str, Any] = {
            "assets": {},
            "liabilities": {},
            "equity": {},
            "total_assets": 0.0,
            "total_liabilities_equity": 0.0,
        }

        for acc_num, account in self.accounts.items():
            if account.type == "Asset" and account.balance != 0:
                balance_sheet["assets"][acc_num] = {
                    "name": account.name,
                    "balance": account.balance,
                }
                balance_sheet["total_assets"] += account.balance
            elif account.type == "Liability/Equity" and account.balance != 0:
                bucket = (
                    "equity"
                    if acc_num.startswith("20") or acc_num.startswith("21")
                    else "liabilities"
                )
                balance_sheet[bucket][acc_num] = {
                    "name": account.name,
                    "balance": account.balance,
                }
                balance_sheet["total_liabilities_equity"] += account.balance

        return balance_sheet

    def get_income_statement(self) -> dict[str, Any]:
        """Generate an income statement from the data model."""
        self.calculate_account_balances()

        income_statement: dict[str, Any] = {
            "income": {},
            "expenses": {},
            "total_income": 0.0,
            "total_expenses": 0.0,
            "net_income": 0.0,
        }

        for acc_num, account in self.accounts.items():
            if account.type == "Income" and account.balance != 0:
                income_statement["income"][acc_num] = {
                    "name": account.name,
                    "amount": account.balance,
                }
                income_statement["total_income"] += account.balance
            elif account.type == "Expense" and account.balance != 0:
                income_statement["expenses"][acc_num] = {
                    "name": account.name,
                    "amount": account.balance,
                }
                income_statement["total_expenses"] += account.balance

        income_statement["net_income"] = (
            income_statement["total_income"] - income_statement["total_expenses"]
        )
        return income_statement

    def to_dict(self) -> dict[str, Any]:
        """Convert the data model to a JSON-serializable dictionary."""
        result: dict[str, Any] = {
            "metadata": self.metadata.to_dict(),
            "accounts": {
                acc_num: acc.to_dict() for acc_num, acc in self.accounts.items()
            },
            "verifications": [ver.to_dict() for ver in self.verifications],
            "opening_balances": {},
            "closing_balances": {},
            "results": {},
            "balance_sheet": self.get_balance_sheet(),
            "income_statement": self.get_income_statement(),
        }

        for year, balances in self.opening_balances.items():
            result["opening_balances"][year] = {
                acc_num: balance.to_dict() for acc_num, balance in balances.items()
            }

        for year, balances in self.closing_balances.items():
            result["closing_balances"][year] = {
                acc_num: balance.to_dict() for acc_num, balance in balances.items()
            }

        for year, balances in self.results.items():
            result["results"][year] = {
                acc_num: balance.to_dict() for acc_num, balance in balances.items()
            }

        return result

    def to_json(self) -> str:
        """Convert the data model to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
