from app.services.minio_client import is_viewable_in_browser


def test_browser_viewable_types_open_inline() -> None:
    assert is_viewable_in_browser("q3_report_2024.pdf")
    assert is_viewable_in_browser("org_chart.png")
    assert is_viewable_in_browser("notes.txt")


def test_office_and_unknown_types_are_downloaded() -> None:
    assert not is_viewable_in_browser("budget.xlsx")
    assert not is_viewable_in_browser("agreement.docx")
    assert not is_viewable_in_browser("archive.bin")
