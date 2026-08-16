from pydantic import BaseModel, ConfigDict, Field

from app.models.enums import Category, DocType, Language


class FigureBBoxAnnotation(BaseModel):
    """Per-figure caption produced by OCR bbox annotation."""

    model_config = ConfigDict(extra="ignore")

    short_description: str = Field(
        default="",
        description="One-line description of the figure or chart.",
    )
    summary: str = Field(
        default="",
        description="What the figure shows, including trends, axes, and notable values.",
    )


class OcrDocumentAnnotation(BaseModel):
    """Document-level fields extracted by OCR document_annotation."""

    model_config = ConfigDict(extra="ignore")

    company_name: str | None = Field(
        default=None,
        description="Legal or trading name of the company the document is about.",
    )
    doc_type: DocType | None = Field(
        default=None,
        description="Document type. Must be one of the allowed enum values.",
    )
    category: Category | None = Field(
        default=None,
        description="Business category. Must be one of the allowed enum values.",
    )
    year: int | None = Field(
        default=None,
        description="Fiscal or calendar year the document refers to.",
    )
    quarter: int | None = Field(
        default=None,
        ge=1,
        le=4,
        description="Fiscal quarter 1-4, or omit if the document is not quarterly.",
    )
    language: Language | None = Field(
        default=None,
        description=(
            "Primary language of the document body as an allowed enum code "
            "(sv, en, dk, no, fi, de, fr, other)."
        ),
    )
