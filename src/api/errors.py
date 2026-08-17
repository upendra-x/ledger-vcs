"""The HTTP error contract, in one place.

Errors must *carry state* rather than just a code — a 409
returns the ref's current target and generation so a caller can rebase without a
second round trip. That only holds if the mapping lives in exactly one function,
because a per-route translation drifts.

Two rules the handlers enforce:

**A 422 means exactly two things.** FastAPI's default is to return 422 for a
request-body validation failure, which would make it ambiguous with the stated design's
idempotency-key mismatch. Request validation is remapped to 400, so a client
seeing 422 knows it reused a key with a different payload.

**An unexpected exception leaks nothing.** Anything that is not a ``LedgerError``
reaching this layer is a bug, and is reported as a bare 500 — never with an
internal message.

The registry surface has a **different** error contract, because the OCI
distribution specification defines one and a container runtime parses it. That
translation is selected by path here rather than by a per-route wrapper: a route
someone adds later gets it without having to remember, which is the only way a
contract stays true across a codebase.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.api.oci import REGISTRY_PREFIX, oci_error_response
from src.errors import InvalidRequest, LedgerError, RateLimited

__all__ = ["install_error_handlers"]

logger = logging.getLogger("ledger.api")


def _is_registry(request: Request) -> bool:
    return request.url.path.startswith(f"{REGISTRY_PREFIX}/") or (
        request.url.path == REGISTRY_PREFIX
    )


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(LedgerError)
    async def _ledger_error(request: Request, exc: LedgerError) -> JSONResponse:
        if _is_registry(request):
            return oci_error_response(exc)
        headers: dict[str, str] = {}
        if isinstance(exc, RateLimited):
            # Retryable, never a broken result. With an idempotency key a client
            # that backs off cannot duplicate work.
            headers["Retry-After"] = str(max(1, int(exc.retry_after_s)))
        return JSONResponse(status_code=exc.status_code, content=exc.to_payload(), headers=headers)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        """Remapped to 400 so that 422 keeps its single meaning."""
        if _is_registry(request):
            return oci_error_response(
                InvalidRequest("the request is not valid", oci_code="UNSUPPORTED")
            )
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "code": "invalid_request",
                "message": "the request body or parameters are not valid",
                "details": {"errors": exc.errors()},
            },
        )

    @app.exception_handler(ValueError)
    async def _value_error(request: Request, exc: ValueError) -> JSONResponse:
        """Identifier parsing raises ``ValueError``; that is a client error.

        Deliberately narrow: the message is the parser's own, which describes the
        malformed input and nothing about internal state.
        """
        if _is_registry(request):
            return oci_error_response(InvalidRequest(str(exc), oci_code="UNSUPPORTED"))
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"code": "invalid_request", "message": str(exc)},
        )

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled error serving %s %s", request.method, request.url.path)
        del exc
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"code": "internal_error", "message": "internal error"},
        )
