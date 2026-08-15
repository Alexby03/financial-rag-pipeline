"""Convert Bolagsverket-style XHTML/HTML annual reports into a multi-page PDF.

Page boundaries come from known container classes / CSS page-break markers when
present. Each page is printed alone via headless Edge/Chrome so A4 layout is
preserved. The resulting PDF is then OCR'd with the existing Mistral 30-page
slicer.

If no page containers are found, the whole document is printed once and the
browser's CSS ``@page`` / ``page-break-*`` rules define the pages.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader, PdfWriter

XHTML_EXTENSIONS = {".xhtml", ".html", ".htm"}

# Bolagsverket / annual-report generators mark print pages with these classes.
_PAGE_CLASS_RE = re.compile(
    r"(?:^|\s)(?:"
    r"rb-page"
    r"|ar-page(?:\s|$)"  # ar-page but not ar-page-hdr
    r"|pagebreak_before"
    r"|normalhardpagebreak"
    r"|rubrik\w*hardpagebreak"
    r")",
    re.IGNORECASE,
)

_HEAD_RE = re.compile(r"<head[^>]*>(.*?)</head>", re.IGNORECASE | re.DOTALL)
_BODY_OPEN_RE = re.compile(r"<body[^>]*>", re.IGNORECASE)
_HTML_OPEN_RE = re.compile(r"<html[^>]*>", re.IGNORECASE)


class XhtmlConversionError(RuntimeError):
    """Raised when XHTML cannot be turned into a printable PDF."""


def is_xhtml_filename(filename: str) -> bool:
    return Path(filename).suffix.lower() in XHTML_EXTENSIONS


def xhtml_to_pdf(file_bytes: bytes, filename: str = "document.xhtml") -> bytes:
    """Render XHTML bytes to a multi-page PDF ready for Mistral OCR."""
    if not file_bytes:
        raise XhtmlConversionError("XHTML file is empty")

    html = _decode_html(file_bytes)
    pages = split_xhtml_pages(html)
    browser = find_print_browser()
    if browser is None:
        raise XhtmlConversionError(
            "No Chrome/Edge browser found for XHTML print-to-PDF. "
            "Install Microsoft Edge or Google Chrome, or set CHROME_PATH / EDGE_PATH."
        )

    with tempfile.TemporaryDirectory(prefix="xhtml_pdf_") as temp_dir:
        work = Path(temp_dir)
        pdf_parts: list[bytes] = []
        for index, page_html in enumerate(pages, start=1):
            source = work / f"page_{index:04d}.html"
            source.write_text(page_html, encoding="utf-8")
            target = work / f"page_{index:04d}.pdf"
            _print_html_to_pdf(browser, source, target)
            pdf_parts.append(target.read_bytes())

    if len(pdf_parts) == 1:
        return pdf_parts[0]
    return _merge_pdfs(pdf_parts)


def split_xhtml_pages(html: str) -> list[str]:
    """Return one printable HTML document per detected page (or the whole doc)."""
    roots = _find_page_roots(html)
    if len(roots) < 2:
        return [_as_printable_document(html, html)]

    head = _extract_head(html)
    html_open = _HTML_OPEN_RE.search(html)
    html_open_tag = html_open.group(0) if html_open else "<html>"
    return [_wrap_page(html_open_tag, head, fragment) for _, fragment in roots]


def find_print_browser() -> str | None:
    """Resolve a Chromium-based browser binary for headless print-to-PDF."""
    for env_name in ("CHROME_PATH", "EDGE_PATH", "BROWSER_PATH"):
        configured = os.environ.get(env_name, "").strip()
        if configured and Path(configured).is_file():
            return configured

    candidates = [
        shutil.which("msedge"),
        shutil.which("chrome"),
        shutil.which("google-chrome"),
        shutil.which("chromium"),
        shutil.which("chromium-browser"),
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        "/usr/bin/google-chrome",
        "/usr/bin/chromium",
        "/usr/bin/chromium-browser",
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    ]
    for path in candidates:
        if path and Path(path).is_file():
            return path
    return None


def _decode_html(file_bytes: bytes) -> str:
    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return file_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    return file_bytes.decode("utf-8", errors="replace")


def _extract_head(html: str) -> str:
    match = _HEAD_RE.search(html)
    if match:
        return match.group(1)
    return ""


def _wrap_page(html_open_tag: str, head_inner: str, body_fragment: str) -> str:
    # Force one physical page when printing a single extracted container.
    page_css = """
    <style type="text/css">
      @page { size: A4 portrait; margin: 10mm; }
      html, body { margin: 0; padding: 0; }
      .rb-page, .ar-page, [class*="pagebreak_before"] {
        page-break-after: avoid !important;
        page-break-before: avoid !important;
        break-after: avoid !important;
        break-before: avoid !important;
        box-shadow: none !important;
        border: none !important;
        margin: 0 !important;
      }
    </style>
    """
    return (
        f'<?xml version="1.0" encoding="UTF-8"?>\n'
        f"{html_open_tag}\n<head>\n<meta charset=\"utf-8\"/>\n{head_inner}\n{page_css}\n"
        f"</head>\n<body>\n{body_fragment}\n</body>\n</html>\n"
    )


def _as_printable_document(original_html: str, body_source: str) -> str:
    """Use the original document; inject a small print hint if none exists."""
    if "@page" in original_html.lower() or "page-break" in original_html.lower():
        return original_html
    injection = (
        "<style type=\"text/css\">@page { size: A4 portrait; margin: 12mm; }</style>"
    )
    if "</head>" in original_html.lower():
        return re.sub(r"</head>", injection + "</head>", original_html, count=1, flags=re.I)
    body = _BODY_OPEN_RE.search(original_html)
    if body:
        return original_html[: body.end()] + injection + original_html[body.end() :]
    return injection + original_html


def _find_page_roots(html: str) -> list[tuple[int, str]]:
    """Return (start_index, outer_html) for top-level page containers."""
    parser = _PageRootFinder()
    parser.feed(html)
    parser.close()
    return parser.pages


class _PageRootFinder(HTMLParser):
    """Collect outer HTML for elements whose class marks a print page."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.pages: list[tuple[int, str]] = []
        self._stack: list[tuple[str, int, bool, int]] = []
        # tag, startpos, is_page, depth_within_page
        self._capture: list[str] | None = None
        self._capture_depth = 0
        self._capture_start = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._handle_start(tag, attrs, self.get_starttag_text() or f"<{tag}>")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        text = self.get_starttag_text() or f"<{tag}/>"
        if self._capture is not None:
            self._capture.append(text)
            return
        # Self-closing page roots are ignored (not useful as pages).

    def _handle_start(self, tag: str, attrs: list[tuple[str, str | None]], raw: str) -> None:
        class_attr = ""
        for name, value in attrs:
            if name.lower() == "class" and value:
                class_attr = value
                break
        is_page = bool(_PAGE_CLASS_RE.search(class_attr)) and not re.search(
            r"(?:^|\s)ar-page-hdr(?:\s|$)", class_attr, re.I
        )

        if self._capture is not None:
            self._capture.append(raw)
            self._capture_depth += 1
            self._stack.append((tag, 0, False, self._capture_depth))
            return

        if is_page:
            self._capture = [raw]
            self._capture_depth = 1
            self._capture_start = self.getpos()[0]  # approximate; unused for merge
            self._stack.append((tag, 0, True, 1))
            return

        self._stack.append((tag, 0, False, 0))

    def handle_endtag(self, tag: str) -> None:
        raw = f"</{tag}>"
        if self._capture is not None:
            self._capture.append(raw)
            self._capture_depth -= 1
            if self._capture_depth <= 0:
                fragment = "".join(self._capture)
                self.pages.append((self._capture_start, fragment))
                self._capture = None
                self._capture_depth = 0
            return

    def handle_data(self, data: str) -> None:
        if self._capture is not None:
            self._capture.append(data)

    def handle_entityref(self, name: str) -> None:
        if self._capture is not None:
            self._capture.append(f"&{name};")

    def handle_charref(self, name: str) -> None:
        if self._capture is not None:
            self._capture.append(f"&#{name};")

    def handle_comment(self, data: str) -> None:
        if self._capture is not None:
            self._capture.append(f"<!--{data}-->")


def _print_html_to_pdf(browser: str, source: Path, target: Path) -> None:
    uri = source.resolve().as_uri()
    command = [
        browser,
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={target}",
        uri,
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired as error:
        raise XhtmlConversionError(f"Timed out printing {source.name} to PDF") from error
    except OSError as error:
        raise XhtmlConversionError(f"Failed to launch browser for PDF print: {error}") from error

    if not target.is_file() or target.stat().st_size == 0:
        detail = (completed.stderr or completed.stdout or "").strip()[:300]
        raise XhtmlConversionError(
            f"Browser did not write a PDF for {source.name}. {detail}"
        )


def _merge_pdfs(parts: list[bytes]) -> bytes:
    writer = PdfWriter()
    for part in parts:
        reader = PdfReader(BytesIO(part))
        for page in reader.pages:
            writer.add_page(page)
    buffer = BytesIO()
    writer.write(buffer)
    return buffer.getvalue()
