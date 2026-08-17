"""Request and response models.

Shaped for automated callers rather than for a person at a terminal: every read names a commit
and a path, requests batch, and listings
paginate by cursor. There is no clone step — reading one file out of one version
costs one request regardless of how large the environment is.

Object names cross the wire as strings and are parsed at the boundary, so an
invalid name is a 400 at the edge rather than a failure several layers in.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "ChangeModel",
    "CreateEnvRequest",
    "DiffResponse",
    "DirectoryEntryModel",
    "EnvResponse",
    "HasObjectsRequest",
    "HasObjectsResponse",
    "ListDirResponse",
    "ListRefsResponse",
    "LogResponse",
    "MintTokenRequest",
    "MintTokenResponse",
    "NoteRequest",
    "OpEntryModel",
    "PutNoteResponse",
    "ReadFileResponse",
    "RefResponse",
    "ResolveResponse",
    "SessionResponse",
    "UpdateRefRequest",
]

ObjectNameStr = Annotated[str, Field(pattern=r"^b3:[0-9a-f]{64}$", examples=["b3:9f2c…"])]


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# ── environments ─────────────────────────────────────────────────────────────


class CreateEnvRequest(Model):
    name: str = Field(description="org/environment")
    owner: str = ""
    labels: dict[str, str] = Field(default_factory=dict)


class EnvResponse(Model):
    env_id: str
    name: str
    state: str
    default_ref: str
    labels: dict[str, str] = Field(default_factory=dict)
    forked_from_env: str | None = None
    forked_from_commit: str | None = None


class UpdateEnvRequest(Model):
    """Every field optional: a PATCH says what changes, not what everything is."""

    name: str | None = None
    state: Literal["active", "ready", "archived"] | None = None
    labels: dict[str, str] | None = None


class ForkEnvRequest(Model):
    name: str = Field(description="Name for the new environment.")
    from_ref: str = "refs/heads/main"
    owner: str = ""


# ── refs ─────────────────────────────────────────────────────────────────────


class ResolveResponse(Model):
    """The one read that touches mutable state.

    Everything after it is addressed by content, so a scheduler resolves once
    and hands the hash to a thousand workers.
    """

    env_id: str
    ref: str
    commit: ObjectNameStr
    generation: int


class RefResponse(Model):
    name: str
    target: ObjectNameStr
    generation: int
    lifecycle: str
    expires_at_us: int | None = None


class ListRefsResponse(Model):
    refs: list[RefResponse]


class UpdateRefRequest(Model):
    target: ObjectNameStr
    #: Compared against the ref's current generation, never against its target —
    #: a ref that moved away and back is indistinguishable by target alone.
    expected_generation: int | None = Field(
        default=None, description="Omit to create a ref that does not exist yet."
    )
    change_id: str | None = None


class CreateRefRequest(Model):
    target: ObjectNameStr | None = Field(
        default=None, description="Defaults to the environment's default ref."
    )
    ephemeral: bool = False
    ttl_days: int = 14


# ── writing ──────────────────────────────────────────────────────────────────


class SessionResponse(Model):
    session_id: str
    expires_at_us: int


class HasObjectsRequest(Model):
    """Phase 2 of the write protocol — where deduplication is realised as *transfer*.

    The client offers hashes and uploads only what Ledger has never seen,
    whether that content came from this environment, a fork, or an unrelated one.
    """

    names: list[ObjectNameStr] = Field(max_length=1000)


class HasObjectsResponse(Model):
    missing: list[ObjectNameStr]


# ── reading ──────────────────────────────────────────────────────────────────


class DirectoryEntryModel(Model):
    name: str
    kind: str
    target: ObjectNameStr
    size: int
    executable: bool = False


class ListDirResponse(Model):
    entries: list[DirectoryEntryModel]
    #: The last name returned, or null when exhausted. Opaque by contract, and
    #: deliberately readable so a paginated API is debuggable.
    cursor: str | None = None


class ReadFileResponse(Model):
    """Returned when a caller asks for metadata rather than bytes.

    ``content_url`` is a short-lived ticket-bearing URL: the service resolves,
    the edge delivers, and the API plane stays off the bandwidth path.
    """

    path: str
    size: int
    object_name: ObjectNameStr
    content_url: str


class CommitModel(Model):
    name: ObjectNameStr
    tree: ObjectNameStr
    parents: list[ObjectNameStr]
    change_id: str
    author: str
    committer: str
    timestamp_us: int
    message: str
    metadata: dict[str, str] = Field(default_factory=dict)


class LogResponse(Model):
    commits: list[CommitModel]


class ChangeModel(Model):
    path: str
    kind: str
    size_delta: int


class DiffResponse(Model):
    changes: list[ChangeModel]
    #: A two-million-entry change must not have to fit in one response.
    truncated: bool = False


# ── notes and operations ─────────────────────────────────────────────────────


class NoteRequest(Model):
    body: dict[str, Any]


class PutNoteResponse(Model):
    commit: ObjectNameStr
    namespace: str


class NoteResponse(Model):
    commit: ObjectNameStr
    namespace: str
    body: dict[str, Any]
    updated_by: str
    updated_at_us: int


class OpEntryModel(Model):
    sequence: int
    kind: str
    ref: str | None
    before: ObjectNameStr | None
    after: ObjectNameStr | None
    principal: str
    at_us: int


class ListOpsResponse(Model):
    operations: list[OpEntryModel]


# ── build and sync ───────────────────────────────────────────────────────────


class TriggerBuildRequest(Model):
    """Ask for a build of what a ref currently points at.

    Ordinary builds need no call at all — they are triggered by the ref update
    itself, through the change stream. This exists for reruns and backfills.
    """

    ref: str = "refs/heads/main"
    #: Forget the cached result first. A build is a pure function of a commit,
    #: so asking to rebuild is a statement that the previous *answer* is no
    #: longer trusted — and that has to be said where every consumer sees it.
    rebuild: bool = False


class TriggerBuildResponse(Model):
    commit: ObjectNameStr
    ref: str
    #: False when this commit was already queued. Not an error: the work the
    #: caller wanted done is going to be done.
    queued: bool


class BuildResponse(Model):
    """A build result. Keyed by commit, so a fork sees its parent's."""

    commit: ObjectNameStr
    status: str
    started_at_us: int
    finished_at_us: int
    duration_us: int
    exit_code: int | None = None
    images: list[str] = Field(default_factory=list)
    attempts: int = 1
    worker: str = ""
    log_excerpt: str = ""


class BuildFailuresResponse(Model):
    """Failures across the whole corpus, in one query.

    Because results are keyed by commit rather than buried per environment, a
    systemic build regression is one signal rather than ten million silent ones.
    """

    failures: list[BuildResponse]


# ── authorization ────────────────────────────────────────────────────────────


class MintTokenRequest(Model):
    principal: str
    ttl_seconds: int = Field(default=3600, ge=1, le=86_400)
    #: Narrow the minted token below the principal's grants. Only ever narrows —
    #: widening is not expressible.
    operations: list[str] | None = None
    env_id: str | None = None
    commit: ObjectNameStr | None = None


class MintTokenResponse(Model):
    token: str
    expires_at_us: int
    operations: list[str]
