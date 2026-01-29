"""
Data models for deed validation.
"""

from dataclasses import dataclass
from datetime import datetime


@dataclass
class ExtractedDeed:
    """Structured representation of a deed extracted from OCR text"""
    doc_number: str
    county: str
    state: str
    date_signed: datetime
    date_recorded: datetime
    grantor: str
    grantee: str
    amount_numeric: float
    amount_written: str
    apn: str
    status: str
