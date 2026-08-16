from enum import StrEnum


class DocType(StrEnum):
    AGREEMENT = "agreement"
    FINANCIAL_REPORT = "financial_report"
    MINUTES = "minutes"
    ORG_CHART = "org_chart"
    ACCOUNTING_DATA = "accounting_data"  # .se and .si files
    PRESENTATION = "presentation"
    REPORT_ANALYSIS = "report_analysis"
    POLICY_REGULATION = "policy_regulation"
    TRANSACTION_DOC = "transaction_doc"  # term sheets, prospectuses
    OTHER = "other"


class Category(StrEnum):
    HR = "hr"
    FINANCING = "financing"
    LEGAL = "legal"
    OPERATIONS = "operations"
    COMPLIANCE_GOVERNANCE = "compliance_governance"
    M_AND_A = "m_and_a"
    STRATEGY_MARKET = "strategy_market"
    OTHER = "other"


class Language(StrEnum):
    """ISO-ish language codes stored on every chunk. Danish uses `dk` (not `da`)."""

    SV = "sv"
    EN = "en"
    DK = "dk"
    NO = "no"
    FI = "fi"
    DE = "de"
    FR = "fr"
    OTHER = "other"
