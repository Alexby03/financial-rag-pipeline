from io import BytesIO

from pypdf import PdfWriter

from app.services.mistral_ocr import OcrPage, OcrResult
from app.services.parser import DocumentParser
from app.services.xhtml_pdf import is_xhtml_filename, split_xhtml_pages, xhtml_to_pdf

TWO_PAGE_XHTML = """<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml">
<head><title>Test</title>
<style type="text/css">@page { size: A4; }</style>
</head>
<body>
<div class="rb-page general-page-layout" id="first-page"><p>Page one content</p></div>
<div class="rb-page general-page-layout"><p>Page two content</p></div>
</body>
</html>
"""


class FakeOcrClient:
    def __init__(self) -> None:
        self.calls: list[tuple[bytes, str]] = []

    def process(self, file_bytes: bytes, filename: str) -> OcrResult:
        self.calls.append((file_bytes, filename))
        return OcrResult(
            pages=[
                OcrPage(index=1, markdown="# Page one content"),
                OcrPage(index=2, markdown="# Page two content"),
            ]
        )


def _one_page_pdf() -> bytes:
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buffer = BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def test_is_xhtml_filename() -> None:
    assert is_xhtml_filename("report.xhtml")
    assert is_xhtml_filename("report.HTML")
    assert is_xhtml_filename("a.htm")
    assert not is_xhtml_filename("report.pdf")


def test_split_xhtml_pages_uses_rb_page_containers() -> None:
    pages = split_xhtml_pages(TWO_PAGE_XHTML)
    assert len(pages) == 2
    assert "Page one content" in pages[0]
    assert "Page two content" in pages[1]
    assert "Page two content" not in pages[0]


def test_split_xhtml_pages_falls_back_to_whole_document() -> None:
    html = "<html><body><p>No page markers here</p></body></html>"
    pages = split_xhtml_pages(html)
    assert len(pages) == 1
    assert "No page markers here" in pages[0]


def test_document_parser_converts_xhtml_then_ocrs_pdf(
    monkeypatch,
) -> None:
    client = FakeOcrClient()
    parser = DocumentParser(ocr_client=client)
    pdf_bytes = _one_page_pdf()

    def fake_xhtml_to_pdf(file_bytes: bytes, filename: str = "document.xhtml") -> bytes:
        assert b"rb-page" in file_bytes
        assert filename.endswith(".xhtml")
        return pdf_bytes

    monkeypatch.setattr("app.services.parser.xhtml_to_pdf", fake_xhtml_to_pdf)

    parsed = parser.parse(TWO_PAGE_XHTML.encode("utf-8"), "bolagsverket.xhtml")

    assert client.calls == [(pdf_bytes, "bolagsverket.pdf")]
    assert [page.page_number for page in parsed.pages] == [1, 2]
    assert "Page one content" in parsed.pages[0].text


def test_xhtml_to_pdf_requires_browser(monkeypatch) -> None:
    monkeypatch.setattr("app.services.xhtml_pdf.find_print_browser", lambda: None)
    try:
        xhtml_to_pdf(TWO_PAGE_XHTML.encode("utf-8"), "t.xhtml")
        assert False, "expected XhtmlConversionError"
    except Exception as error:
        assert "Chrome/Edge" in str(error)
