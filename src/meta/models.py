"""Domain records held in the metadata store.

These are the mutable side of the split: named by us rather than by
their own bytes, small, and changed only by compare-and-swap. Every one of them
serialises to a JSON body and back, and the round trip is property-tested —
because a field silently dropped on the way out is a ref that forgets its
lifecycle, or a session that forgets what it uploaded.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Self, final

from src.ids import ChangeId, EnvId, EnvName, ObjectName, RefName, SessionId

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

__all__ = [
    "ChangeRecord",
    "EnvState",
    "Environment",
    "IdempotencyRecord",
    "Note",
    "OpKind",
    "OpLogEntry",
    "Ref",
    "RefLifecycle",
    "WriteSession",
]


class EnvState(StrEnum):
    """99% of the corpus is finished and read-only, and saying so
    explicitly is what lets storage be tiered and a write to a finished
    environment be refused without reading anything else.
    """

    ACTIVE = "active"
    READY = "ready"
    ARCHIVED = "archived"


class RefLifecycle(StrEnum):
    """Forks live indefinitely; branches are discarded and their storage
    reclaimed. The distinction is carried here rather than
    inferred from a naming convention, because a sweeper has to act on it.
    """

    PERMANENT = "permanent"
    EPHEMERAL = "ephemeral"


class OpKind(StrEnum):
    CREATE_ENV = "CreateEnv"
    RENAME_ENV = "RenameEnv"
    ARCHIVE_ENV = "ArchiveEnv"
    FORK_ENV = "ForkEnv"
    CREATE_REF = "CreateRef"
    UPDATE_REF = "UpdateRef"
    DELETE_REF = "DeleteRef"
    PUT_NOTE = "PutNote"
    UNDO = "Undo"


@final
@dataclass(frozen=True, slots=True)
class Environment:
    env_id: EnvId
    name: EnvName
    default_ref: RefName
    state: EnvState = EnvState.ACTIVE
    owner: str = ""
    #: Key/value pairs the authorization layer scopes grants against.
    labels: Mapping[str, str] = ()  # type: ignore[assignment]
    #: Provenance only — deliberately *not* a storage dependency. Archiving or
    #: deleting a parent does not affect a fork, because retention is decided
    #: per-ref over globally shared objects.
    forked_from_env: EnvId | None = None
    forked_from_commit: ObjectName | None = None
    created_at_us: int = 0

    def to_body(self) -> dict[str, Any]:
        return {
            "env_id": str(self.env_id),
            "name": str(self.name),
            "default_ref": str(self.default_ref),
            "state": str(self.state),
            "owner": self.owner,
            "labels": dict(self.labels),
            "forked_from_env": str(self.forked_from_env) if self.forked_from_env else None,
            "forked_from_commit": (
                str(self.forked_from_commit) if self.forked_from_commit else None
            ),
            "created_at_us": self.created_at_us,
        }

    @classmethod
    def from_body(cls, body: Mapping[str, Any]) -> Self:
        return cls(
            env_id=EnvId(body["env_id"]),
            name=EnvName(body["name"]),
            default_ref=RefName(body["default_ref"]),
            state=EnvState(body["state"]),
            owner=body.get("owner", ""),
            labels=dict(body.get("labels") or {}),
            forked_from_env=EnvId(body["forked_from_env"]) if body.get("forked_from_env") else None,
            forked_from_commit=(
                ObjectName.parse(body["forked_from_commit"])
                if body.get("forked_from_commit")
                else None
            ),
            created_at_us=body.get("created_at_us", 0),
        )


@final
@dataclass(frozen=True, slots=True)
class Ref:
    """A mutable name pointing at a commit — the entire mutable surface of a
    versioned environment.
    """

    name: RefName
    target: ObjectName
    #: Incremented on every successful update. Updates compare on *this*, not on
    #: the target; see ``GenerationIs``.
    generation: int
    lifecycle: RefLifecycle = RefLifecycle.PERMANENT
    expires_at_us: int | None = None
    updated_by: str = ""
    updated_at_us: int = 0

    def to_body(self) -> dict[str, Any]:
        return {
            "name": str(self.name),
            "target": str(self.target),
            "generation": self.generation,
            "lifecycle": str(self.lifecycle),
            "expires_at_us": self.expires_at_us,
            "updated_by": self.updated_by,
            "updated_at_us": self.updated_at_us,
        }

    @classmethod
    def from_body(cls, body: Mapping[str, Any]) -> Self:
        return cls(
            name=RefName(body["name"]),
            target=ObjectName.parse(body["target"]),
            generation=body["generation"],
            lifecycle=RefLifecycle(body.get("lifecycle", RefLifecycle.PERMANENT)),
            expires_at_us=body.get("expires_at_us"),
            updated_by=body.get("updated_by", ""),
            updated_at_us=body.get("updated_at_us", 0),
        )


@final
@dataclass(frozen=True, slots=True)
class ChangeRecord:
    """``change_id`` → its commits, newest first.

    Neither resolution nor obsolescence is derivable from content: commits point
    at parents rather than at what they supersede. So the list is maintained
    explicitly, appended in the same transaction as the ref move.
    """

    change_id: ChangeId
    commits: Sequence[ObjectName]

    def to_body(self) -> dict[str, Any]:
        return {"change_id": str(self.change_id), "commits": [str(c) for c in self.commits]}

    @classmethod
    def from_body(cls, body: Mapping[str, Any]) -> Self:
        return cls(
            change_id=ChangeId(body["change_id"]),
            commits=[ObjectName.parse(c) for c in body["commits"]],
        )

    @property
    def current(self) -> ObjectName:
        return self.commits[0]


@final
@dataclass(frozen=True, slots=True)
class Note:
    """A fact about a commit, attached from outside it.

    It cannot go inside the commit: a commit's name is the hash of its content,
    so appending a QA verdict would change its identity, break every reference to
    it, and stop two identical environments from sharing it at all.

    Notes are **not** GC roots. A verdict is a fact about a version, not a reason
    to retain one, so annotating a commit can never change what storage costs.
    """

    commit: ObjectName
    namespace: str
    body: Mapping[str, Any]
    updated_by: str = ""
    updated_at_us: int = 0

    def to_body(self) -> dict[str, Any]:
        return {
            "commit": str(self.commit),
            "namespace": self.namespace,
            "body": dict(self.body),
            "updated_by": self.updated_by,
            "updated_at_us": self.updated_at_us,
        }

    @classmethod
    def from_body(cls, body: Mapping[str, Any]) -> Self:
        return cls(
            commit=ObjectName.parse(body["commit"]),
            namespace=body["namespace"],
            body=dict(body["body"]),
            updated_by=body.get("updated_by", ""),
            updated_at_us=body.get("updated_at_us", 0),
        )


@final
@dataclass(frozen=True, slots=True)
class OpLogEntry:
    """One mutation, appended in the same transaction as the mutation itself —
    so the log can never disagree with the refs it describes.

    It serves undo, audit and debugging. It is also why commits named in the
    retained log are GC roots: undo restores a ref to an old commit, and it
    cannot resurrect objects that collection has already reclaimed.
    """

    sequence: int
    kind: OpKind
    ref: RefName | None
    before: ObjectName | None
    after: ObjectName | None
    principal: str
    at_us: int
    detail: Mapping[str, Any] = ()  # type: ignore[assignment]

    def to_body(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "kind": str(self.kind),
            "ref": str(self.ref) if self.ref else None,
            "before": str(self.before) if self.before else None,
            "after": str(self.after) if self.after else None,
            "principal": self.principal,
            "at_us": self.at_us,
            "detail": dict(self.detail),
        }

    @classmethod
    def from_body(cls, body: Mapping[str, Any]) -> Self:
        return cls(
            sequence=body["sequence"],
            kind=OpKind(body["kind"]),
            ref=RefName(body["ref"]) if body.get("ref") else None,
            before=ObjectName.parse(body["before"]) if body.get("before") else None,
            after=ObjectName.parse(body["after"]) if body.get("after") else None,
            principal=body.get("principal", ""),
            at_us=body.get("at_us", 0),
            detail=dict(body.get("detail") or {}),
        )


@final
@dataclass(frozen=True, slots=True)
class WriteSession:
    """An open write, and the lease that keeps its objects alive.

    Objects uploaded under a session are GC roots until it ends. A session ends
    one of two ways: ``UpdateRef`` succeeds and its objects graduate into the
    environment's keep-set, or the lease expires and they become ordinary
    garbage.

    The consequence is that **an abandoned write requires no cleanup by anyone**
    — no rollback, no compensating transaction, no half-written state to repair.
    """

    session_id: SessionId
    principal: str
    expires_at_us: int
    uploaded: Sequence[ObjectName] = ()
    created_at_us: int = 0

    def to_body(self) -> dict[str, Any]:
        return {
            "session_id": str(self.session_id),
            "principal": self.principal,
            "expires_at_us": self.expires_at_us,
            "uploaded": [str(n) for n in self.uploaded],
            "created_at_us": self.created_at_us,
        }

    @classmethod
    def from_body(cls, body: Mapping[str, Any]) -> Self:
        return cls(
            session_id=SessionId(body["session_id"]),
            principal=body.get("principal", ""),
            expires_at_us=body["expires_at_us"],
            uploaded=[ObjectName.parse(n) for n in body.get("uploaded", ())],
            created_at_us=body.get("created_at_us", 0),
        )


@final
@dataclass(frozen=True, slots=True)
class IdempotencyRecord:
    """A completed mutation, keyed by the caller's idempotency key.

    ``fingerprint`` is what distinguishes a genuine retry from a client bug:
    the same key with a *different* request is a 422, because replaying a key
    must mean replaying a request.
    """

    key: str
    fingerprint: str
    outcome: Mapping[str, Any]
    expires_at_us: int
    created_at_us: int = 0

    def to_body(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "fingerprint": self.fingerprint,
            "outcome": dict(self.outcome),
            "expires_at_us": self.expires_at_us,
            "created_at_us": self.created_at_us,
        }

    @classmethod
    def from_body(cls, body: Mapping[str, Any]) -> Self:
        return cls(
            key=body["key"],
            fingerprint=body["fingerprint"],
            outcome=dict(body["outcome"]),
            expires_at_us=body["expires_at_us"],
            created_at_us=body.get("created_at_us", 0),
        )
