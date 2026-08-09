from enum import StrEnum


class DocType(StrEnum):
    AGREEMENT = "agreement"
    FINANCIAL_REPORT = "financial_report"
    MINUTES = "minutes"
    ORG_CHART = "org_chart"
    OTHER = "other"


class Category(StrEnum):
    HR = "HR"
    FINANCING = "Financing"
    LEGAL = "Legal"
    OPERATIONS = "Operations"
    OTHER = "other"
