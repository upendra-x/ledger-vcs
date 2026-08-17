"""What the pipeline records, and where each record belongs.

The outcome is written back two ways, and the distinction is the whole design
rather than a filing convention::

    build#c9f2…        { status: ok, images: […], duration_us: … }   keyed by COMMIT
    note#c9f2…#sync    { platform_id: env-8821, synced_at: … }       keyed by ENVIRONMENT

The **build result is global and keyed by commit**, because a build is a pure
function of a commit. That is what lets two refs at the same commit build once,
and what lets a fork that changed nothing inherit its parent's result rather than
rebuilding a forty-gigabyte environment to discover it is identical.

The **sync record is a note in the environment's partition**, because which
platform entry a commit corresponds to is a fact about the environment, not about
the commit — a fork syncs to a different platform entry from the same bytes.

Neither can live *inside* the commit: a commit's name is the hash of its content,
so appending an outcome would change its identity and break every reference to it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, final

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "BuildOutcome",
    "BuildResult",
    "BuildStatus",
    "QueuedBuild",
    "SyncRecord",
]


class BuildStatus(StrEnum):
    """Where a build got to.

    ``RUNNING`` is durable rather than in-memory: a worker records it before
    doing any work, so a second worker that picks up the same commit can see it
    is already being built instead of duplicating five minutes of effort.
    """

    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"

    @property
    def terminal(self) -> bool:
        return self in {BuildStatus.SUCCEEDED, BuildStatus.FAILED}


@final
@dataclass(frozen=True, slots=True)
class BuildResult:
    """The outcome of building one commit. Global, keyed by the commit hash."""

    commit: str
    status: BuildStatus
    started_at_us: int
    finished_at_us: int = 0
    exit_code: int | None = None
    #: Image names published by this build, as ``name@sha256:…``. Digests rather
    #: than tags, because a tag is exactly the thing a version must not depend on.
    images: tuple[str, ...] = ()
    #: The tail of the build's output. Bounded on purpose — a build result is
    #: read far more often than it is written, and an unbounded log would make
    #: the cheapest read in the pipeline the most expensive.
    log_excerpt: str = ""
    attempts: int = 1
    manifest: Mapping[str, Any] = field(default_factory=dict)
    #: The worker that produced it, so a stuck build can be traced to a host.
    worker: str = ""

    @property
    def duration_us(self) -> int:
        return max(0, self.finished_at_us - self.started_at_us)

    @property
    def succeeded(self) -> bool:
        return self.status is BuildStatus.SUCCEEDED

    def to_body(self) -> dict[str, Any]:
        return {
            "commit": self.commit,
            "status": str(self.status),
            "started_at_us": self.started_at_us,
            "finished_at_us": self.finished_at_us,
            "exit_code": self.exit_code,
            "images": list(self.images),
            "log_excerpt": self.log_excerpt,
            "attempts": self.attempts,
            "manifest": dict(self.manifest),
            "worker": self.worker,
        }

    @classmethod
    def from_body(cls, body: Mapping[str, Any]) -> BuildResult:
        return cls(
            commit=str(body["commit"]),
            status=BuildStatus(body["status"]),
            started_at_us=int(body.get("started_at_us", 0)),
            finished_at_us=int(body.get("finished_at_us", 0)),
            exit_code=body.get("exit_code"),
            images=tuple(str(i) for i in body.get("images", ())),
            log_excerpt=str(body.get("log_excerpt", "")),
            attempts=int(body.get("attempts", 1)),
            manifest=dict(body.get("manifest", {})),
            worker=str(body.get("worker", "")),
        )


@final
@dataclass(frozen=True, slots=True)
class QueuedBuild:
    """One unit of work: build this commit, on behalf of this environment.

    Ordered within an environment by ``op_sequence`` — the operation log's own
    dense counter — so the queue inherits the publish order for free rather than
    inventing a second notion of "which came first". That is what stops version
    49 from syncing after version 50.
    """

    env_id: str
    commit: str
    ref: str
    op_sequence: int
    enqueued_at_us: int
    attempts: int = 0
    #: Set when a build is asked for directly rather than triggered by a ref
    #: update — reruns and backfills.
    manual: bool = False

    def to_body(self) -> dict[str, Any]:
        return {
            "env_id": self.env_id,
            "commit": self.commit,
            "ref": self.ref,
            "op_sequence": self.op_sequence,
            "enqueued_at_us": self.enqueued_at_us,
            "attempts": self.attempts,
            "manual": self.manual,
        }

    @classmethod
    def from_body(cls, body: Mapping[str, Any]) -> QueuedBuild:
        return cls(
            env_id=str(body["env_id"]),
            commit=str(body["commit"]),
            ref=str(body["ref"]),
            op_sequence=int(body["op_sequence"]),
            enqueued_at_us=int(body.get("enqueued_at_us", 0)),
            attempts=int(body.get("attempts", 0)),
            manual=bool(body.get("manual", False)),
        )


@final
@dataclass(frozen=True, slots=True)
class SyncRecord:
    """Which platform entry a commit was synced to. A note, in the environment."""

    platform_id: str
    commit: str
    synced_at_us: int
    images: tuple[str, ...] = ()

    def to_body(self) -> dict[str, Any]:
        return {
            "platform_id": self.platform_id,
            "commit": self.commit,
            "synced_at_us": self.synced_at_us,
            "images": list(self.images),
        }


@final
@dataclass(frozen=True, slots=True)
class BuildOutcome:
    """What one turn of the worker did — the thing a test or an operator reads."""

    work: QueuedBuild
    result: BuildResult
    #: True when the result came from the commit-keyed store rather than from
    #: running anything. This is a build's purity made observable: a fork that
    #: changed nothing must show ``cache_hit`` and leave the runner untouched.
    cache_hit: bool
    synced: SyncRecord | None = None

    @property
    def commit(self) -> str:
        return self.work.commit
