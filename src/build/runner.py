"""Running a build. The one part of the pipeline that is not Ledger's business.

Ledger's job ends at "here is the environment, materialized byte for byte, and
here is what its manifest asks for". What actually runs is a policy of the
platform, which is why this is an interface with two small implementations rather
than a build system.

The interface is narrow on purpose. A runner gets a directory and a manifest and
returns an exit code, some output and the images it published — nothing about
refs, commits, storage or authorization. That is what keeps a container build, a
remote executor and a dry run interchangeable.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Protocol, final

if TYPE_CHECKING:
    from pathlib import Path

    from src.build.manifest import Manifest

__all__ = ["RecordingRunner", "RunOutcome", "RunRequest", "Runner", "SubprocessRunner"]

#: How much build output is kept. A build result is read far more often than it
#: is written — every fork, every retrigger — so an unbounded log would make the
#: cheapest read in the pipeline the most expensive one.
LOG_EXCERPT_BYTES: Final = 4096


@final
@dataclass(frozen=True, slots=True)
class RunRequest:
    """Everything a runner is told. Deliberately no Ledger vocabulary."""

    commit: str
    env_name: str
    workspace: Path
    manifest: Manifest
    #: Images the commit pinned, as ``name@sha256:…``. Passed in rather than
    #: discovered, because what a version contains is decided by the commit and
    #: a runner that went looking could find something else.
    images: tuple[str, ...] = ()


@final
@dataclass(frozen=True, slots=True)
class RunOutcome:
    exit_code: int
    output: str = ""
    #: Images the build *published*, which may differ from what it was given.
    images: tuple[str, ...] = ()

    @property
    def succeeded(self) -> bool:
        return self.exit_code == 0


class Runner(Protocol):
    def run(self, request: RunRequest) -> RunOutcome: ...


@final
class SubprocessRunner:
    """Runs the manifest's command in the materialized environment.

    The workspace is a real directory containing exactly what the commit
    contains, so a build here is reproducible in the strongest sense available:
    materializing a two-year-old commit gives byte-identical inputs, including
    the container layers it was committed with.

    A timeout is always applied. The manifest may ask for less than the ceiling
    but never more, because one environment must not be able to hold a worker
    indefinitely.
    """

    __slots__ = ()

    def run(self, request: RunRequest) -> RunOutcome:
        step = request.manifest.build
        if step is None:
            return RunOutcome(exit_code=0, output="no build step in the manifest")

        workdir = (request.workspace / step.workdir).resolve()
        if not workdir.is_relative_to(request.workspace.resolve()):
            # A manifest is content someone uploaded. `workdir:../../etc` must
            # not run the build outside the materialized environment.
            return RunOutcome(exit_code=126, output="build workdir escapes the environment")

        environment = {
            "LEDGER_COMMIT": request.commit,
            "LEDGER_ENV": request.env_name,
            "LEDGER_IMAGES": " ".join(request.images),
            **step.env_map,
        }
        try:
            completed = subprocess.run(
                list(step.command),
                cwd=workdir,
                env=environment,
                capture_output=True,
                text=True,
                timeout=step.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return RunOutcome(
                exit_code=124, output=f"build exceeded {step.timeout_seconds}s and was killed"
            )
        except OSError as exc:
            return RunOutcome(exit_code=127, output=f"could not start the build: {exc}")

        output = (completed.stdout + completed.stderr)[-LOG_EXCERPT_BYTES:]
        return RunOutcome(exit_code=completed.returncode, output=output, images=request.images)


@final
class RecordingRunner:
    """Records what it was asked to build and runs nothing.

    Two real uses. It is how an operator dry-runs the pipeline against a corpus
    without executing anybody's build commands. And its invocation count is what
    makes "a build is a pure function of a commit" checkable rather than merely
    stated: fork an environment, trigger a build, and assert the count **did not
    move** — which is only true if the fork inherited its parent's result
    instead of rebuilding.
    """

    __slots__ = ("invocations",)

    def __init__(self) -> None:
        self.invocations: list[RunRequest] = []

    def run(self, request: RunRequest) -> RunOutcome:
        self.invocations.append(request)
        return RunOutcome(
            exit_code=0,
            output=f"recorded build of {request.commit}",
            images=request.images,
        )

    @property
    def count(self) -> int:
        return len(self.invocations)
