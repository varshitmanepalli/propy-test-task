"""
Custom exceptions for deed validation.
"""


class ValidationError(Exception):
    """Base exception for validation errors"""
    pass


class DateLogicError(ValidationError):
    """Raised when recorded date is before signed date"""
    pass


class MoneyMismatchError(ValidationError):
    """Raised when written amount doesn't match numeric amount"""
    pass


class CountyNotFoundError(ValidationError):
    """Raised when county cannot be matched"""
    pass
