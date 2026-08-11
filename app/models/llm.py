from pydantic import BaseModel


class ReasonedAnswer(BaseModel):
    answer: str
    reasoning_summary: str = ""
