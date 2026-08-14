from pydantic import BaseModel


class ReasonedAnswer(BaseModel):
    answer: str
    reasoning_summary: str = ""


class FigureReading(BaseModel):
    """Vision pass over a stored figure crop, fed back to the reasoning model as text."""

    chunk_id: str
    source_filename: str
    page_number: int
    reading: str = ""
    error: str = ""
