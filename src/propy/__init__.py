"""
Propy Deed Validator Package

A professional solution for validating OCR-scanned property deeds using
LLM extraction followed by rigorous code-based validation.
"""

from .exceptions import (
    ValidationError,
    DateLogicError,
    MoneyMismatchError,
    CountyNotFoundError
)
from .models import ExtractedDeed
from .validator import DeedValidator

__version__ = "1.0.0"
__all__ = [
    "ValidationError",
    "DateLogicError",
    "MoneyMismatchError",
    "CountyNotFoundError",
    "ExtractedDeed",
    "DeedValidator",
]
