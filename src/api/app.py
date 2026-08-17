"""The application factory — the composition root for the service.

Everything the request path needs is assembled once, here, and handed to routes
through one dependency. Nothing constructs a store, a signer or a policy engine
on its own, which is what keeps ``ObjectStore``'s no-default tombstone rule
meaningful: there is exactly one place that decision is made.

The API plane holds **no state of its own**. It reads and writes
through the two stores and keeps nothing between requests, so reads scale by
adding replicas and writes scale independently of them.
"""

from __future__ import annotations

import logging
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI

from src.api.deps import AppState
from src.api.errors import install_error_handlers
from src.api.jj import router as jj_router
from src.api.oci import router as oci_router
from src.api.routes import router
from src.api.ui.routes import router as ui_router
from src.auth.policy import PolicyEngine
from src.auth.tokens import TicketSigner, TokenSigner, TokenVerifier
from src.instance import Ledger
from src.service.commits import CommitService

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from src.clock import Clock

__all__ = ["build_app"]

logger = logging.getLogger("ledger.api")


def build_app(
    data_dir: Path | str = "./data",
    *,
    clock: Clock | None = None,
    dev_mode: bool = False,
    ledger: Ledger | None = None,
) -> FastAPI:
    """Assemble the service.

    ``dev_mode`` makes unauthenticated requests act as a fully-privileged local
    operator. It exists so a single-machine demo does not need a token dance,
    it is off by default, and it announces itself loudly — a convenience, never
    a deployment mode.
    """
    instance = ledger if ledger is not None else Ledger(data_dir, clock=clock)

    signer = TokenSigner.generate(key_id="k1", clock=instance.clock)
    verifier = TokenVerifier({signer.key_id: signer.public_key}, clock=instance.clock)
    tickets = TicketSigner(secrets.token_bytes(32), clock=instance.clock)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        # Only close what this factory opened. A caller that supplied its own
        # Ledger owns its lifetime — closing it here would surprise a test that
        # still wants to inspect the stores afterwards.
        if ledger is None:
            instance.close()

    app = FastAPI(
        lifespan=lifespan,
        title="Ledger",
        summary="A version control system for RL environments.",
        version="0.1.0",
        docs_url="/docs",
    )
    app.state.ledger_state = AppState(
        ledger=instance,
        policy=PolicyEngine(instance),
        commits=CommitService(instance),
        verifier=verifier,
        signer=signer,
        tickets=tickets,
        dev_mode=dev_mode,
    )

    install_error_handlers(app)
    app.include_router(router)
    # The OCI distribution endpoint. A separate surface with its
    # own error contract and its own credential shape, over exactly the same
    # objects and exactly the same authorization.
    app.include_router(jj_router)
    app.include_router(oci_router)
    # The read-only browser. Last, so its catch-all paths cannot shadow an
    # API route, and strictly a view over the same reads and the same
    # authorization — there is no write anywhere in it.
    app.include_router(ui_router)

    if dev_mode:
        logger.warning(
            "DEV MODE: unauthenticated requests act as a full administrator. "
            "Never enable this outside a local machine."
        )

    return app
