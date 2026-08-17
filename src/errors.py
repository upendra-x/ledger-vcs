"""The one exception hierarchy, and the one place HTTP status codes are decided.

Errors must *carry state* rather than just a code: a 409
returns the ref's current target and generation so a caller can rebase without a
second round trip, a 422 marks idempotency-key reuse with a different payload,
a 429 carries ``Retry-After``. That contract only holds if there is exactly one
place it is expressed — otherwise the API layer re-derives it per route and the
derivations drift.

So every error carries three things: an HTTP status, a stable machine-readable
``code`` that clients may branch on, and a ``details`` mapping that becomes the
response body. The API layer's exception handler is then a single function with
no per-route knowledge.

Layering note: nothing here imports a web framework. These exceptions are raised
by pure and service code that must remain testable without HTTP.
"""

from __future__ import annotations

from typing import Any, ClassVar

__all__ = [
    "BackendUnavailable",
    "CodecError",
    "Conflict",
    "CorruptObject",
    "DomainError",
    "Forbidden",
    "IdempotencyMismatch",
    "InvalidRequest",
    "LedgerError",
    "MalformedObject",
    "NotCanonical",
    "NotFound",
    "ObjectNotFound",
    "RateLimited",
    "StoreError",
    "Unauthenticated",
    "UnsupportedFormatVersion",
]


class LedgerError(Exception):
    """Base for every error Ledger raises deliberately.

    An exception that is *not* a ``LedgerError`` reaching the API boundary is a
    bug, and is reported as a 500 with no detail — we never leak an internal
    message to a caller.
    """

    status_code: ClassVar[int] = 500
    code: ClassVar[str] = "internal_error"

    def __init__(self, message: str, **details: Any) -> None:
        super().__init__(message)
        self.message = message
        self.details: dict[str, Any] = details

    def to_payload(self) -> dict[str, Any]:
        """The response body. Stable across versions; clients branch on ``code``."""
        payload: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.details:
            payload["details"] = self.details
        return payload

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.message!r}, {self.details!r})"


# ─────────────────────────────────────────────────────────────────────────────
# Codec — the bytes we were handed are not a well-formed, canonical object
# ─────────────────────────────────────────────────────────────────────────────


class CodecError(LedgerError):
    """Encoding or decoding failed. Always the caller's fault; always a 400."""

    status_code = 400
    code = "codec_error"


class MalformedObject(CodecError):
    """Structurally invalid: truncated, trailing bytes, impossible field values."""

    code = "malformed_object"


class NotCanonical(CodecError):
    """Well-formed and self-consistent, but not the *canonical* encoding.

    This is the subtle one, and it is why ingest re-encodes rather than only
    rehashing. A client can construct a tree whose entries are out of order,
    hash those exact bytes, and offer them under that hash — so the hash check
    passes. One logical tree then has two names, and deduplication silently
    splits, which is precisely the failure content addressing exists to prevent.
    One name per content requires canonical form, not merely self-consistency.
    """

    code = "not_canonical"


class UnsupportedFormatVersion(CodecError):
    """A format version this build has never emitted and cannot interpret.

    Readers accept every version ever emitted; this fires only for
    a version from the future.
    """

    code = "unsupported_format_version"


# ─────────────────────────────────────────────────────────────────────────────
# Store — the object store could not serve or accept bytes
# ─────────────────────────────────────────────────────────────────────────────


class StoreError(LedgerError):
    status_code = 500
    code = "store_error"


class ObjectNotFound(StoreError):
    status_code = 404
    code = "object_not_found"


class CorruptObject(StoreError):
    """Stored bytes did not match the name they were requested by.

    Never downgraded to a miss and never returned as data. This is one of the two
    signals that must never be silent: it means a storage medium,
    a cache entry, or a code path is actively wrong.
    """

    status_code = 500
    code = "corrupt_object"


class BackendUnavailable(StoreError):
    status_code = 503
    code = "backend_unavailable"


# ─────────────────────────────────────────────────────────────────────────────
# Domain — the request was understood and refused
# ─────────────────────────────────────────────────────────────────────────────


class DomainError(LedgerError):
    status_code = 400
    code = "domain_error"


class InvalidRequest(DomainError):
    status_code = 400
    code = "invalid_request"


class Unauthenticated(DomainError):
    status_code = 401
    code = "unauthenticated"


class Forbidden(DomainError):
    """Decided before any resolution work happens.

    Carries no information about whether the target exists — a 403 that leaks
    existence is an enumeration oracle over the corpus.
    """

    status_code = 403
    code = "forbidden"


class NotFound(DomainError):
    status_code = 404
    code = "not_found"


class Conflict(DomainError):
    """A compare-and-swap lost. Carries the current state so the caller can rebase.

    "a 409 is not a bare failure". The whole point of comparing on
    ``generation`` rather than on target is that this response tells the
    caller exactly which state it lost to.
    """

    status_code = 409
    code = "conflict"


class IdempotencyMismatch(DomainError):
    """The same idempotency key was reused with a different payload.

    A client bug, not a race: replaying a key must mean replaying a request.
    """

    status_code = 422
    code = "idempotency_mismatch"


class RateLimited(DomainError):
    """Retryable, never a broken result. Always accompanied by ``Retry-After``."""

    status_code = 429
    code = "rate_limited"

    def __init__(self, message: str, *, retry_after_s: float, **details: Any) -> None:
        super().__init__(message, retry_after_s=retry_after_s, **details)
        self.retry_after_s = retry_after_s
