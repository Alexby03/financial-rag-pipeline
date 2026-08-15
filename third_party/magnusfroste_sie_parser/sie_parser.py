# Adapted from https://github.com/magnusfroste/sie-parser
# Copyright (c) 2025 Magnus Froste — MIT License. See LICENSE in this directory.
# Lean adaptation for rag-system-mvp (debug prints and demo UI helpers removed).

from __future__ import annotations

import re
from typing import Any

from .data_model import SIEDataModel, Transaction, Verification


class SIEParser:
    """
    Parser for Swedish SIE 4 files.
    Prefers CP437 encoding with latin-1 / utf-8 fallbacks.
    """

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.data: dict[str, Any] = {
            "metadata": {},
            "accounts": {},
            "verifications": [],
            "ib": {},
            "ub": {},
            "res": {},
        }
        self.program_info: dict[str, str | None] = {"name": None, "version": None}
        self.data_model = SIEDataModel()

    def parse(self) -> dict[str, Any] | None:
        """Parse the SIE file and return structured data, or None on hard failure."""
        try:
            lines = self._read_lines()
            if lines is None:
                return None

            current_ver: Verification | None = None
            in_verification_block = False

            for raw_line in lines:
                line = raw_line.strip()
                if not line:
                    continue

                if line.startswith("#FLAGGA"):
                    self._parse_flagga(line)
                elif line.startswith("#PROGRAM"):
                    self._parse_program(line)
                elif line.startswith("#FORMAT"):
                    self._parse_format(line)
                elif line.startswith("#GEN"):
                    self._parse_gen(line)
                elif line.startswith("#SIETYP"):
                    self._parse_sietyp(line)
                elif line.startswith("#RAR"):
                    self._parse_rar(line)
                elif line.startswith("#OMFATTN"):
                    self._parse_omfattn(line)
                elif line.startswith("#TAXAR"):
                    self._parse_taxar(line)
                elif line.startswith("#FNAMN"):
                    self._parse_fnamn(line)
                elif line.startswith("#ORGNR"):
                    self._parse_orgnr(line)
                elif line.startswith("#ADRESS"):
                    self._parse_adress(line)
                elif line.startswith("#KPTYP"):
                    self._parse_kptyp(line)
                elif line.startswith("#KONTO"):
                    self._parse_konto(line)
                elif line.startswith("#SRU"):
                    self._parse_sru(line)
                elif line.startswith("#IB"):
                    self._parse_ib(line)
                elif line.startswith("#UB"):
                    self._parse_ub(line)
                elif line.startswith("#RES"):
                    self._parse_res(line)
                elif line.startswith("#VER") or line.startswith("VER "):
                    if current_ver and not in_verification_block:
                        self.data["verifications"].append(current_ver)
                    current_ver = self._parse_ver(line)
                elif line.startswith("#TRANS") and current_ver:
                    self._parse_trans(line, current_ver)
                elif line.startswith("#RTRANS") and current_ver:
                    self._parse_rtrans(line, current_ver)
                elif line.startswith("#BTRANS"):
                    self._parse_btrans(line)
                elif line.startswith("{"):
                    in_verification_block = True
                elif line.startswith("}"):
                    if current_ver:
                        self.data["verifications"].append(current_ver)
                        current_ver = None
                    in_verification_block = False

            if current_ver and not in_verification_block:
                self.data["verifications"].append(current_ver)

            self._calculate_account_balances()
            self._process_data()
            self.data_model.from_parser_data(self.data)
            return self.data_model.to_dict()
        except Exception:
            return None

    def _read_lines(self) -> list[str] | None:
        """Read file lines, preferring CP437 with encoding fallbacks."""
        for encoding in ("cp437", "latin-1", "utf-8"):
            try:
                with open(self.file_path, "r", encoding=encoding) as file:
                    return file.readlines()
            except UnicodeDecodeError:
                continue
            except OSError:
                return None
        return None

    def _extract_quoted_string(self, text: str) -> str:
        """Extract the first string enclosed in quotes."""
        match = re.search(r'"([^"]*)"', text)
        return match.group(1) if match else ""

    def _extract_values(self, line: str) -> list[str]:
        """Split a SIE line into tokens; tabs and spaces are both delimiters."""
        values: list[str] = []
        for match in re.finditer(r'"([^"]*)"|(\S+)', line.strip()):
            quoted, bare = match.groups()
            values.append(quoted if quoted is not None else bare)
        return values

    def _parse_flagga(self, line: str) -> None:
        parts = self._extract_values(line)
        if len(parts) > 1:
            self.data["metadata"]["flagga"] = parts[1]

    def _parse_program(self, line: str) -> None:
        values = self._extract_values(line)
        # values[0] is #PROGRAM; remaining are name / version (version is NOT a fiscal year).
        name = values[1] if len(values) > 1 else self._extract_quoted_string(line)
        version = values[2] if len(values) > 2 else None
        self.data["metadata"]["program"] = name
        if version:
            self.data["metadata"]["program_version"] = version
        self.program_info["name"] = name or None
        self.program_info["version"] = version

    def _parse_format(self, line: str) -> None:
        parts = self._extract_values(line)
        if len(parts) > 1:
            self.data["metadata"]["format"] = parts[1]

    def _parse_gen(self, line: str) -> None:
        parts = self._extract_values(line)
        if len(parts) > 1:
            self.data["metadata"]["gen_date"] = parts[1]
            self.data["metadata"]["generation_date"] = parts[1]

    def _parse_sietyp(self, line: str) -> None:
        parts = self._extract_values(line)
        if len(parts) > 1:
            self.data["metadata"]["sie_type"] = parts[1]

    def _parse_fnamn(self, line: str) -> None:
        self.data["metadata"]["company_name"] = self._extract_quoted_string(line)

    def _parse_orgnr(self, line: str) -> None:
        parts = self._extract_values(line)
        if len(parts) > 1:
            org_number = parts[1]
            self.data["metadata"]["organization_number"] = org_number
            self.data["metadata"]["org_number"] = org_number

    def _parse_omfattn(self, line: str) -> None:
        """Parse #OMFATTN — period end date the balances cover."""
        parts = self._extract_values(line)
        if len(parts) > 1:
            self.data["metadata"]["omfattn"] = parts[1]

    def _parse_taxar(self, line: str) -> None:
        """Parse #TAXAR — taxation year."""
        parts = self._extract_values(line)
        if len(parts) > 1:
            self.data["metadata"]["taxar"] = parts[1]

    def _parse_rar(self, line: str) -> None:
        parts = self._extract_values(line)
        # Expect: #RAR year_id start end
        if len(parts) < 3:
            return

        year_id = parts[1]
        start_date = parts[2]
        end_date = parts[3] if len(parts) > 3 else None

        if "fiscal_years" not in self.data["metadata"]:
            self.data["metadata"]["fiscal_years"] = {}

        self.data["metadata"]["fiscal_years"][year_id] = {
            "start_date": start_date,
            "end_date": end_date,
        }

        if start_date and end_date and len(start_date) >= 8 and len(end_date) >= 8:
            formatted_start = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]}"
            formatted_end = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:8]}"
            self.data["metadata"]["fiscal_year"] = f"{formatted_start} to {formatted_end}"

        # Current year is year_id 0; if that is missing, still record the first RAR seen.
        is_current = str(year_id) == "0"
        already_set = bool(self.data["metadata"].get("financial_year_start"))
        if is_current or not already_set:
            self.data["metadata"]["current_fiscal_year"] = {
                "start_date": start_date,
                "end_date": end_date,
            }
            if start_date and len(start_date) >= 4:
                self.data["metadata"]["rar"] = start_date[:4]
            if start_date and len(start_date) >= 8:
                self.data["metadata"]["financial_year_start"] = (
                    f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]}"
                )
                self.data["metadata"]["current_fiscal_year_start_year"] = start_date[:4]
            if end_date and len(end_date) >= 8:
                self.data["metadata"]["financial_year_end"] = (
                    f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:8]}"
                )
                self.data["metadata"]["current_fiscal_year_end_year"] = end_date[:4]

    def _parse_konto(self, line: str) -> None:
        parts = self._extract_values(line)
        if len(parts) >= 2:
            account_number = parts[1]
            account_name = parts[2] if len(parts) > 2 else ""
            self.data["accounts"][account_number] = {
                "name": account_name,
                "type": self._determine_account_type(account_number),
            }

    def _parse_ib(self, line: str) -> None:
        parts = self._extract_values(line)
        if len(parts) >= 3:
            year = parts[1]
            account = parts[2]
            amount = float(parts[3]) if len(parts) > 3 else 0.0
            self.data["ib"].setdefault(year, {})
            self.data["ib"][year][account] = amount

    def _parse_ub(self, line: str) -> None:
        parts = self._extract_values(line)
        if len(parts) >= 3:
            year = parts[1]
            account = parts[2]
            amount = float(parts[3]) if len(parts) > 3 else 0.0
            self.data["ub"].setdefault(year, {})
            self.data["ub"][year][account] = amount

    def _parse_res(self, line: str) -> None:
        """Parse #RES year account amount."""
        original_line = line
        if line.startswith("#"):
            line = line[1:].strip()

        parts = self._extract_values(line)
        if len(parts) >= 3 and parts[0].upper() == "RES":
            try:
                year_key = str(parts[1])
                account_key = str(parts[2])
                amount = float(parts[3].replace(",", ".")) if len(parts) > 3 else 0.0
                self.data["res"].setdefault(year_key, {})
                self.data["res"][year_key][account_key] = amount
                return
            except (ValueError, IndexError):
                pass

        match = re.search(
            r"RES\s+(-?\d+)\s+(\d+)\s+(-?[\d,\.]+)",
            original_line,
            re.IGNORECASE,
        )
        if match:
            year, account, amount_str = match.groups()
            try:
                amount = float(amount_str.replace(",", "."))
            except ValueError:
                return
            year_key = str(year)
            account_key = str(account)
            self.data["res"].setdefault(year_key, {})
            self.data["res"][year_key][account_key] = amount

    def _parse_ver(self, line: str) -> Verification:
        if line.startswith("#"):
            line = line[1:]

        parts = self._extract_values(line)
        series = parts[1] if len(parts) >= 2 else ""
        ver_number = parts[2] if len(parts) >= 3 else ""
        ver_date = parts[3] if len(parts) >= 4 else ""
        ver_text = parts[4] if len(parts) >= 5 else ""
        ver_reg_date = parts[5] if len(parts) >= 6 else ""

        if ver_date and len(ver_date) == 8 and ver_date.isdigit():
            ver_date = f"{ver_date[:4]}-{ver_date[4:6]}-{ver_date[6:8]}"

        verification = Verification(
            series=series,
            number=ver_number,
            date=ver_date,
            text=ver_text,
        )
        if ver_reg_date:
            setattr(verification, "reg_date", ver_reg_date)
        verification.transactions = []
        return verification

    def _apply_object_info(self, parts: list[str]) -> tuple[str, list[str]]:
        """
        Parse object_info at parts[2].

        Single-token {} / {..} keeps amount at parts[3]. Multi-token braces
        collapse into object_info and rebuild parts so amount stays at [3].
        """
        if len(parts) < 3:
            return "", parts

        if parts[2] == "{}" or (
            parts[2].startswith("{") and parts[2].endswith("}")
        ):
            return parts[2], parts

        if parts[2].startswith("{") and not parts[2].endswith("}"):
            object_parts = [parts[2]]
            i = 3
            while i < len(parts) and not parts[i].endswith("}"):
                object_parts.append(parts[i])
                i += 1
            if i < len(parts):
                object_parts.append(parts[i])
                rebuilt = parts[:2] + parts[i + 1 :]
                # Re-insert placeholder so amount remains at index 3
                rebuilt = rebuilt[:2] + [""] + rebuilt[2:]
                return " ".join(object_parts), rebuilt

        return "", parts

    def _parse_trans_fields(
        self, line: str, *, negate_amount: bool = False
    ) -> tuple[str, float, str, str, str, float, str]:
        if line.startswith("#"):
            line = line[1:]

        parts = self._extract_values(line)
        account = parts[1] if len(parts) >= 2 else ""
        object_info, parts = self._apply_object_info(parts)

        amount = 0.0
        if len(parts) >= 4:
            try:
                amount = float(parts[3].replace(",", "."))
                if negate_amount:
                    amount = -amount
            except (ValueError, TypeError):
                pass

        trans_date = ""
        trans_text = ""
        if len(parts) >= 5:
            if len(parts[4]) == 8 and parts[4].isdigit():
                trans_date = f"{parts[4][:4]}-{parts[4][4:6]}-{parts[4][6:8]}"
            elif parts[4] and parts[4] != '""':
                trans_text = parts[4]

        if len(parts) >= 6 and not trans_text:
            trans_text = parts[5]

        quantity = 0.0
        if len(parts) >= 7:
            try:
                quantity = float(parts[6])
            except (ValueError, TypeError):
                pass

        sign = parts[7] if len(parts) >= 8 else ""
        return account, amount, trans_date, trans_text, object_info, quantity, sign

    def _append_transaction(
        self,
        current_ver: Verification,
        account: str,
        amount: float,
        trans_date: str,
        trans_text: str,
        object_info: str,
        quantity: float,
        sign: str,
    ) -> Transaction:
        if not trans_date:
            trans_date = current_ver.date

        transaction = Transaction(
            account=account,
            amount=amount,
            date=trans_date,
            text=trans_text,
        )
        if object_info:
            setattr(transaction, "object_info", object_info)
        if quantity != 0.0:
            setattr(transaction, "quantity", quantity)
        if sign:
            setattr(transaction, "sign", sign)
        if account in self.data["accounts"]:
            transaction.account_name = self.data["accounts"][account].get("name", "")

        current_ver.transactions.append(transaction)
        return transaction

    def _parse_trans(self, line: str, current_ver: Verification) -> Transaction:
        account, amount, trans_date, trans_text, object_info, quantity, sign = (
            self._parse_trans_fields(line)
        )
        return self._append_transaction(
            current_ver,
            account,
            amount,
            trans_date,
            trans_text,
            object_info,
            quantity,
            sign,
        )

    def _parse_rtrans(self, line: str, current_ver: Verification) -> Transaction:
        """Parse #RTRANS (same layout as TRANS; amount is negated)."""
        account, amount, trans_date, trans_text, object_info, quantity, sign = (
            self._parse_trans_fields(line, negate_amount=True)
        )
        return self._append_transaction(
            current_ver,
            account,
            amount,
            trans_date,
            trans_text,
            object_info,
            quantity,
            sign,
        )

    def _parse_btrans(self, line: str) -> None:
        """Budget transactions are accepted but unused in the data model."""
        del line

    def _determine_account_type(self, account_number: str) -> str:
        """Determine account type based on Swedish BAS ranges."""
        if not account_number:
            return "unknown"
        try:
            account_int = int(account_number)
        except ValueError:
            return "unknown"

        if 1000 <= account_int < 2000:
            return "asset"
        if 2000 <= account_int < 3000:
            return "liability"
        if 3000 <= account_int < 4000:
            return "equity_or_liability"
        if 4000 <= account_int < 8000:
            return "income_statement"
        return "other"

    def _process_data(self) -> None:
        """Compute per-account transaction totals and attach account names."""
        account_totals: dict[str, float] = {}

        for ver in self.data["verifications"]:
            transactions = (
                ver.transactions
                if hasattr(ver, "transactions")
                else ver.get("transactions", [])
            )
            for trans in transactions:
                if hasattr(trans, "account"):
                    account = trans.account
                    amount = trans.amount
                else:
                    account = trans.get("account", "")
                    amount = trans.get("amount", 0)

                account_totals[account] = account_totals.get(account, 0) + amount

                if hasattr(trans, "account"):
                    if account in self.data["accounts"]:
                        name = self.data["accounts"][account].get("name", "Unknown")
                    else:
                        name = "Unknown"
                    setattr(trans, "account_name", name)
                else:
                    if account in self.data["accounts"]:
                        trans["account_name"] = self.data["accounts"][account].get(
                            "name", "Unknown"
                        )
                    else:
                        trans["account_name"] = "Unknown"

        self.data["account_totals"] = account_totals

    def _calculate_account_balances(self) -> None:
        """Calculate account balances from opening balances and transactions."""
        account_balances: dict[str, float] = {}

        for balances in self.data.get("ib", {}).values():
            for account, amount in balances.items():
                account_balances[account] = account_balances.get(account, 0) + amount

        for ver in self.data["verifications"]:
            transactions = (
                ver.transactions
                if hasattr(ver, "transactions")
                else ver.get("transactions", [])
            )
            for trans in transactions:
                if hasattr(trans, "account"):
                    account = trans.account
                    amount = trans.amount
                else:
                    account = trans.get("account", "")
                    amount = trans.get("amount", 0)
                account_balances[account] = account_balances.get(account, 0) + amount

        self.data["account_balances"] = account_balances

    def _parse_adress(self, line: str) -> None:
        parts = self._extract_values(line)
        if len(parts) >= 2:
            self.data["metadata"]["address"] = parts[1]
        if len(parts) >= 3:
            self.data["metadata"]["postal_code"] = parts[2]
        if len(parts) >= 4:
            self.data["metadata"]["city"] = parts[3]

    def _parse_kptyp(self, line: str) -> None:
        parts = self._extract_values(line)
        if len(parts) >= 2:
            self.data["metadata"]["account_type"] = parts[1]

    def _parse_sru(self, line: str) -> None:
        parts = self._extract_values(line)
        if len(parts) >= 3:
            account = parts[1]
            sru_code = parts[2]
            if account in self.data["accounts"]:
                self.data["accounts"][account]["sru_code"] = sru_code
