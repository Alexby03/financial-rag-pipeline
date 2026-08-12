from app.services.heading_normalizer import (
    markdown_level_for_section,
    normalize_heading_line,
    normalize_page_markdown,
    numbered_section_depth,
)


def test_outline_depth_follows_decimal_sections() -> None:
    assert numbered_section_depth("3") == 1
    assert numbered_section_depth("2.3") == 2
    assert markdown_level_for_section("3") == 2
    assert markdown_level_for_section("2.3") == 3


def test_top_level_section_is_demoted_from_h3_to_h2() -> None:
    assert normalize_heading_line("### 3. Analys av finansiering") == (
        "## 3. Analys av finansiering"
    )


def test_subsections_keep_deeper_levels() -> None:
    assert normalize_heading_line("## 2.1 Resultaträkning") == "### 2.1 Resultaträkning"
    assert normalize_heading_line("2.1 Resultaträkning") == "### 2.1 Resultaträkning"
    assert normalize_heading_line("#### 2.1.1 Detaljer") == "#### 2.1.1 Detaljer"


def test_unnumbered_document_title_is_unchanged() -> None:
    assert normalize_heading_line("# Kvartalsrapport Q3 2024") == "# Kvartalsrapport Q3 2024"


def test_plain_numbered_lines_become_markdown_headings() -> None:
    markdown = normalize_page_markdown("3. Analys av finansiering\n\nBrödtext.")
    assert markdown.startswith("## 3. Analys av finansiering")
