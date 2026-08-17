"""The demo portal: the product's service, with a console mounted beside it.

    uv run python -m demo.portal        # then open http://127.0.0.1:8080/console

**Nothing under ``src/`` is modified to make this work.** ``build_app``
already accepts a ready-made ``Ledger`` and attaches its routers with
``include_router``, so the portal composes the real service — the same ``/v1``,
the same ``/v2`` registry, the same read-only browser, the same authorization —
and adds three routers of its own:

* ``/console`` — the page, the name distribution, the corpus stats;
* ``/demo``    — the handful of operations ``/v1`` deliberately does not expose;

The browser calls ``/v1`` for everything the product API already supports, which
is the point: a console that had to reach around the API would be evidence the
API was not finished. The request log on the page shows which prefix each call
went to, so the line between product and scaffolding is visible rather than
asserted.

Two things here are deliberately *not* production shapes, and both are visible
on the page rather than hidden:

**Dev mode is on.** Unauthenticated requests act as a local administrator, so the
demonstration does not open with a token dance. The scoped-authorization card
then mints a real narrow token and shows it being refused four ways, which is
the part that matters.

**The clock is hand-cranked.** Leases, ephemeral branches and the collector's
grace period are all *timed* guarantees, and none of them can be shown to an
audience in real time.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Final, final

# Run as ``python demo/portal/app.py`` and the project root is not on the path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from fastapi import FastAPI

from demo.e2e import DEMO_CHUNK_PARAMS, DEMO_EPOCH_US, DEMO_SHAPE_PARAMS
from demo.portal import console, operations
from demo.portal.context import DEMO_DATA, SAMPLE_ENVIRONMENT, BuildRuntime, DemoState
from src.api.app import build_app
from src.clock import ManualClock
from src.instance import Ledger

__all__ = ["DEFAULT_PORT", "Portal", "main"]

#: The port the recorded session uses, matching ``demo/stage.py`` — it is typed
#: into a browser and into ``docker pull`` on camera, and a memorable one is
#: worth more than an automatically-chosen free one.
DEFAULT_PORT: Final = 8080


@final
class Portal:
    """Owns the demonstration's Ledger and the app that serves it."""

    __slots__ = ("_app", "_data_dir", "_port", "_workspace")

    def __init__(
        self,
        *,
        data_dir: Path = DEMO_DATA,
        workspace: Path = SAMPLE_ENVIRONMENT,
        port: int = DEFAULT_PORT,
    ) -> None:
        self._data_dir = Path(data_dir) / "portal"
        self._workspace = Path(workspace)
        self._port = port

        self._app = self._assemble()
        self._app.state.portal = self
        self._app.include_router(console.router)
        self._app.include_router(operations.router)

    @property
    def app(self) -> FastAPI:
        return self._app

    @property
    def state(self) -> DemoState:
        return self._app.state.demo_state  # type: ignore[no-any-return]

    def reset(self) -> DemoState:
        """Throw the corpus away and start the walkthrough from nothing.

        A retake needs a clean slate, and restarting the server to get one costs
        the recording a minute of dead air. So the Ledger is closed, its
        directory removed, and a fresh one assembled in place — the routers and
        the socket are untouched, because nothing in the request path holds a
        reference to the old instance beyond ``app.state``.
        """
        self.state.ledger.close()
        shutil.rmtree(self._data_dir, ignore_errors=True)

        # Assembled by ``build_app`` rather than by hand. The throwaway app is
        # discarded and only its state is kept, which is a little wasteful and
        # buys the one property worth having: the portal cannot end up with a
        # differently-wired service than the product would have built.
        fresh = self._assemble()
        self._app.state.ledger_state = fresh.state.ledger_state
        self._app.state.demo_state = fresh.state.demo_state
        return self.state

    def close(self) -> None:
        self.state.ledger.close()

    # ── internals ────────────────────────────────────────────────────────────

    def _assemble(self) -> FastAPI:
        clock = ManualClock(start_us=DEMO_EPOCH_US)
        ledger = Ledger(
            self._data_dir,
            clock=clock,
            # The demonstration's parameters, imported from ``demo/e2e.py`` so a
            # number measured in the browser is the number `make demo-numbers`
            # prints.
            chunk_params=DEMO_CHUNK_PARAMS,
            shape_params=DEMO_SHAPE_PARAMS,
        )
        app = build_app(ledger=ledger, dev_mode=True)
        app.state.demo_state = DemoState(
            ledger=ledger,
            clock=clock,
            workspace=self._workspace,
            data_dir=self._data_dir,
            port=self._port,
            builds=BuildRuntime(ledger),
        )
        return app


def build_portal(
    *,
    data_dir: Path = DEMO_DATA,
    workspace: Path = SAMPLE_ENVIRONMENT,
    port: int = DEFAULT_PORT,
) -> FastAPI:
    """The app, for a test client or an ASGI server."""
    return Portal(data_dir=data_dir, workspace=workspace, port=port).app


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve the Ledger demo portal.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Default {DEFAULT_PORT}.")
    parser.add_argument("--data-dir", type=Path, default=DEMO_DATA)
    parser.add_argument("--workspace", type=Path, default=SAMPLE_ENVIRONMENT)
    arguments = parser.parse_args()

    if not arguments.workspace.is_dir():
        print(f"no sample environment at {arguments.workspace}")
        print("  → uv run python demo/stage.py")
        return 1

    import uvicorn

    portal = Portal(data_dir=arguments.data_dir, workspace=arguments.workspace, port=arguments.port)
    url = f"http://127.0.0.1:{arguments.port}/console"
    print(f"\n  Ledger demo portal → {url}\n")
    uvicorn.run(portal.app, host="127.0.0.1", port=arguments.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
