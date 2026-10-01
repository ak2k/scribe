"""Domain errors.

All errors raised across module boundaries should inherit from `AppError`.
Never catch and re-raise external library exceptions unchanged — wrap them
in a domain error so callers can match on intent, not library identity.
"""

from __future__ import annotations


class AppError(Exception):
    """Base class for all application errors."""


class NotFoundError(AppError):
    """Raised when an expected resource is missing."""


class InputValidationError(AppError):
    """Raised when caller-supplied input fails domain validation.

    Distinct from `pydantic.ValidationError`, which is library-internal and
    should be wrapped at the boundary (typically as `ExternalServiceError`
    when upstream data is malformed, or `InputValidationError` when the
    caller passed bad input).
    """


class ExternalServiceError(AppError):
    """Raised when an external dependency fails (transport, 5xx, malformed response)."""


class ToolMissingError(ExternalServiceError):
    """Raised when an external program a step runs is not on PATH."""


class EarError(AppError):
    """Raised when the local recognizers cannot run here, or a run of one fails."""


class SpendCapError(AppError):
    """Raised when a paid call could take a run's spending past its cap."""
