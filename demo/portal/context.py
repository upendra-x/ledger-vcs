"""What the portal holds, and how a route reaches it.

The product's request handlers get everything they need from
``request.app.state.ledger_state``. The portal adds exactly one more thing —
``request.app.state.demo_state`` — and it holds only what a *demonstration*
needs and a service would not: where the sample environment lives on disk, and
the hand-cranked clock that makes the reclamation story watchable.

Nothing in ``src/`` knows this module exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Final, final

from fastapi import Depends, Request

# Imported at runtime rather than under TYPE_CHECKING, for the reason
# ``src.api.deps`` gives: FastAPI resolves dependency annotations when it
# builds its graph, and an unresolvable forward reference is silently
# downgraded to a query parameter instead of raising.
from demo.stage import DEMO_DATA, ROOT, SAMPLE_ENVIRONMENT
from src.build.pipeline import BuildWorker, Dispatcher
from src.build.queue import BuildQueue
from src.build.runner import RecordingRunner
from src.build.sync import RecordingPlatform
from src.clock import ManualClock
from src.instance import Ledger

__all__ = [
    "DEMO_DATA",
    "DEMO_ENV",
    "DEMO_ORG",
    "MAIN",
    "ROOT",
    "SAMPLE_ENVIRONMENT",
    "BuildRuntime",
    "DemoDep",
    "DemoState",
    "demo_state",
]

#: The environment the walkthrough builds. The same name ``demo/e2e.py`` uses,
#: so the two demonstrations are recognisably of the same system.
DEMO_ORG: Final = "proximal"
DEMO_ENV: Final = f"{DEMO_ORG}/demo"
MAIN: Final = "refs/heads/main"

# ``DEMO_DATA`` and ``SAMPLE_ENVIRONMENT`` are re-exported from ``demo/stage.py``
# rather than defined again, because staging is what creates both. Deriving them
# here independently is how the portal ended up committing the directory *above*
# the sample environment, leaving every later card reading a path that the tree
# did not have.


@final
class BuildRuntime:
    """A build worker, kept alive across requests.

    In production the dispatcher and the workers are separate processes reading
    the change stream; there is no reason for the API plane to host one. The
    portal hosts one anyway, because the requirement worth demonstrating is that
    *a build is a pure function of a commit* — and the evidence for it is a
    counter that does **not** move when a fork asks for a commit that has
    already been built. A worker rebuilt per request would reset that counter
    and quietly turn the proof into a tautology.
    """

    __slots__ = ("dispatcher", "platform", "queue", "runner", "worker")

    def __init__(self, ledger: Ledger) -> None:
        self.queue = BuildQueue(ledger.meta, clock=ledger.clock)
        self.runner = RecordingRunner()
        self.platform = RecordingPlatform()
        self.dispatcher = Dispatcher(ledger, self.queue)
        self.worker = BuildWorker(
            ledger, runner=self.runner, platform=self.platform, queue=self.queue
        )


@final
@dataclass(frozen=True, slots=True)
class DemoState:
    """The demonstration's own context, assembled once per reset."""

    ledger: Ledger
    #: Hand-cranked on purpose. Three of the guarantees are *timed* —
    #: a lease expires, an ephemeral branch expires, and collection waits out a
    #: grace period the operation log extends — and none of them can be shown to
    #: an audience in real time. A button that advances nine days turns the best
    #: argument into something you can watch.
    clock: ManualClock
    #: The directory the first commit is made from. Server-side, because the
    #: browser has no filesystem to offer.
    workspace: Path
    data_dir: Path
    port: int
    builds: BuildRuntime


def demo_state(request: Request) -> DemoState:
    return request.app.state.demo_state  # type: ignore[no-any-return]


DemoDep = Annotated[DemoState, Depends(demo_state)]
