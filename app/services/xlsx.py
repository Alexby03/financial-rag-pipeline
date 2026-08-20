"""Backward-compatible re-exports for Excel parsing."""

from app.services.spreadsheet import is_xlsx_filename, parse_xlsx_bytes

__all__ = ["is_xlsx_filename", "parse_xlsx_bytes"]
