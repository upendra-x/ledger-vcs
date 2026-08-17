"""The HTTP surface — the operation list.

Routes are declared ``def`` rather than ``async def`` on purpose. Every call
below reaches synchronous SQLite or filesystem I/O, and BLAKE3 hashing is
CPU-bound; Starlette runs a sync handler in its threadpool, so the event loop is
never blocked. Declaring them ``async`` and forgetting one ``run_in_threadpool``
is the failure this avoids by construction.

Every route names the operation it requires through ``require(...)``, which is
what lets ``test_every_route_declares_authorization`` assert that none was
forgotten — a missing check is otherwise invisible until someone exploits it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

from fastapi import APIRouter, Body, Header, Query, Response, status
from fastapi.responses import PlainTextResponse

from src.api import schemas as s
from src.api.deps import Authorized, CurrentCapability, LedgerDep, require, require_global
from src.auth.model import Operation, Principal
from src.auth.tokens import Capability
from src.build.models import BuildResult
from src.errors import Forbidden, InvalidRequest, NotFound
from src.format.constants import EntryKind, ObjectKind
from src.format.model import Commit
from src.fs.blob import BlobReader
from src.fs.tree import list_dir, resolve_path
from src.ids import ChangeId, EnvName, ObjectName, RefName, SessionId
from src.meta.models import EnvState
from src.service.refs import RefService

if TYPE_CHECKING:
    from src.format.model import TreeEntry

__all__ = ["router"]

router = APIRouter(prefix="/v1")

#: "every mutating call must carry a client-generated key". A repeat
#: returns the recorded outcome; the same key with a different payload is a 422.
#: Optional on the wire so a caller may decline the protection, never so the
#: server may.
IdempotencyKey = Annotated[str | None, Header(alias="Idempotency-Key")]

ReadAuth = Annotated[Authorized, require(Operation.READ)]
WriteAuth = Annotated[Authorized, require(Operation.WRITE)]
AnnotateAuth = Annotated[Authorized, require(Operation.ANNOTATE)]
ForkAuth = Annotated[Authorized, require(Operation.FORK)]
AdminAuth = Annotated[Authorized, require(Operation.ADMIN)]
BuildAuth = Annotated[Authorized, require(Operation.BUILD)]

#: Operations with no environment to select against yet — creating one.
#: Checked against the token's operation set alone.
CreateAuth = Annotated[Capability, require_global(Operation.CREATE)]

#: Corpus-wide reads, which no environment selector can scope. Administrative
#: by construction: the question "which builds failed today" is about every
#: environment at once.
AdminOnly = Annotated[Capability, require_global(Operation.ADMIN)]


def _commit_model(name: ObjectName, commit: Commit) -> s.CommitModel:
    return s.CommitModel(
        name=str(name),
        tree=str(commit.tree),
        parents=[str(p) for p in commit.parents],
        change_id=str(commit.change_id),
        author=commit.author,
        committer=commit.committer,
        timestamp_us=commit.timestamp_us,
        message=commit.message,
        metadata=dict(commit.metadata),
    )


def _entry_model(entry: TreeEntry) -> s.DirectoryEntryModel:
    return s.DirectoryEntryModel(
        name=entry.name.decode(errors="replace"),
        kind=entry.kind.name.lower(),
        target=str(entry.target),
        size=entry.size,
        executable=entry.mode == 0o755,
    )


def _root_tree(auth: Authorized, commit: ObjectName) -> ObjectName:
    """Resolve a commit to its tree, having first bound it to the environment.

    The binding is the point: without it a commit hash observed from another
    environment would read that environment's content through this token.
    """
    auth.state.policy.authorize_commit(auth.capability, auth.env_id, commit)
    return auth.ledger.store.get_as(commit, Commit).tree


# ─────────────────────────────────────────────────────────────────────────────
# Environments
# ─────────────────────────────────────────────────────────────────────────────


@router.post("/envs", status_code=status.HTTP_201_CREATED, tags=["environments"])
def create_env(
    body: s.CreateEnvRequest,
    state: LedgerDep,
    capability: CreateAuth,
    idempotency_key: IdempotencyKey = None,
) -> s.EnvResponse:
    """Create an environment in a namespace this token holds.

    ``env:create`` is a *global* operation — there is no environment to
    authorize against yet — but "global" must not mean "anywhere". Checking only
    the operation bit would let a token scoped to ``proximal/*`` create
    ``someone-else/anything``, and because creation claims the name globally,
    that is not merely an overstep: it permanently denies the real owner their
    own namespace.

    ``could_name`` is exactly the right question here, and for the same reason it
    is the right question for a 404 — it asks about authority over a *name*
    rather than over something that exists.
    """
    name = EnvName(body.name)
    if not capability.scope.could_name(name):
        raise Forbidden(
            "this token does not permit creating an environment with that name",
            operation=Operation.CREATE.label,
        )
    record = state.ledger.repo.create_env(
        name, owner=body.owner, labels=body.labels, idempotency_key=idempotency_key
    )
    return s.EnvResponse(
        env_id=str(record.env_id),
        name=str(record.name),
        state=str(record.state),
        default_ref=str(record.default_ref),
        labels=dict(record.labels),
    )


@router.get("/envs/{org}/{env}", tags=["environments"])
def get_env(org: str, env: str, auth: ReadAuth) -> s.EnvResponse:
    del org, env
    record = auth.ledger.repo.get_env(auth.env_id)
    return s.EnvResponse(
        env_id=str(record.env_id),
        name=str(record.name),
        state=str(record.state),
        default_ref=str(record.default_ref),
        labels=dict(record.labels),
        forked_from_env=str(record.forked_from_env) if record.forked_from_env else None,
        forked_from_commit=(str(record.forked_from_commit) if record.forked_from_commit else None),
    )


@router.patch("/envs/{org}/{env}", tags=["environments"])
def update_env(
    org: str,
    env: str,
    body: s.UpdateEnvRequest,
    auth: AdminAuth,
) -> s.EnvResponse:
    """Rename, relabel, or change an environment's state.

    ``env:admin`` carries "rename, archive, change policy", and
    ``state`` its three values. Both existed in the metadata repository with no
    way to reach them, which made ``env:admin`` a scope that gated exactly one
    corpus-wide read and nothing an administrator would actually want to do.

    A rename claims the new name before releasing the old, so a crash leaves the
    environment reachable under one of them rather than neither. Swapping two
    names is deliberately *not* atomic — see ``rename_env``.
    """
    del org, env
    repo = auth.ledger.repo
    record = repo.get_env(auth.env_id)
    if body.name is not None and body.name != str(record.name):
        new_name = EnvName(body.name)
        if not auth.capability.scope.could_name(new_name):
            raise Forbidden(
                "this token does not permit renaming into that namespace",
                operation=Operation.ADMIN.label,
            )
        record = repo.rename_env(auth.env_id, new_name)
    if body.labels is not None:
        record = repo.set_env_labels(auth.env_id, body.labels)
    if body.state is not None:
        record = repo.set_env_state(auth.env_id, EnvState(body.state))
    return s.EnvResponse(
        env_id=str(record.env_id),
        name=str(record.name),
        state=str(record.state),
        default_ref=str(record.default_ref),
        labels=dict(record.labels),
        forked_from_env=str(record.forked_from_env) if record.forked_from_env else None,
        forked_from_commit=(str(record.forked_from_commit) if record.forked_from_commit else None),
    )


@router.post("/envs/{org}/{env}/fork", status_code=status.HTTP_201_CREATED, tags=["environments"])
def fork_env(
    org: str,
    env: str,
    body: s.ForkEnvRequest,
    auth: ForkAuth,
    idempotency_key: IdempotencyKey = None,
) -> s.EnvResponse:
    """Fork an environment. **Copies no bytes.**

    Objects carry no environment identity, so a fork is one
    environment record and one ref pointing at a commit that already exists.
    """
    del org, env
    from src.service.environments import EnvironmentService

    result = EnvironmentService(auth.ledger).fork(
        auth.env_id,
        EnvName(body.name),
        from_ref=RefName(body.from_ref),
        owner=body.owner,
        principal=str(auth.principal),
        idempotency_key=idempotency_key,
    )
    record = result.environment
    return s.EnvResponse(
        env_id=str(record.env_id),
        name=str(record.name),
        state=str(record.state),
        default_ref=str(record.default_ref),
        forked_from_env=str(auth.env_id),
        forked_from_commit=str(result.source_commit),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Refs
# ─────────────────────────────────────────────────────────────────────────────


@router.get("/envs/{org}/{env}/refs", tags=["refs"])
def list_refs(org: str, env: str, auth: ReadAuth) -> s.ListRefsResponse:
    del org, env
    return s.ListRefsResponse(
        refs=[
            s.RefResponse(
                name=str(r.name),
                target=str(r.target),
                generation=r.generation,
                lifecycle=str(r.lifecycle),
                expires_at_us=r.expires_at_us,
            )
            for r in auth.ledger.repo.list_refs(auth.env_id)
        ]
    )


@router.get("/envs/{org}/{env}/refs/{ref:path}/resolve", tags=["refs"])
def resolve(org: str, env: str, ref: str, auth: ReadAuth) -> s.ResolveResponse:
    """Resolve a ref to a commit — the only read that touches mutable state."""
    del org, env
    record = auth.ledger.repo.get_ref(auth.env_id, RefName(ref))
    return s.ResolveResponse(
        env_id=str(auth.env_id),
        ref=str(record.name),
        commit=str(record.target),
        generation=record.generation,
    )


@router.put("/envs/{org}/{env}/refs/{ref:path}", tags=["refs"])
def update_ref(
    org: str,
    env: str,
    ref: str,
    body: s.UpdateRefRequest,
    auth: WriteAuth,
    idempotency_key: IdempotencyKey = None,
) -> s.RefResponse:
    """Move a ref. The single atomic point in the system.

    A 409 carries the ref's current target and generation, which is everything
    the caller needs to rebase without a second round trip.
    """
    del org, env
    name = RefName(ref)
    target = ObjectName.parse(body.target)
    change = ChangeId(body.change_id) if body.change_id else None

    if body.expected_generation is None:
        update = auth.ledger.repo.create_ref(
            auth.env_id,
            name,
            target,
            principal=str(auth.principal),
            change_id=change,
            idempotency_key=idempotency_key,
        )
    else:
        update = auth.ledger.repo.update_ref(
            auth.env_id,
            name,
            expected_generation=body.expected_generation,
            target=target,
            principal=str(auth.principal),
            change_id=change,
            idempotency_key=idempotency_key,
        )
    return s.RefResponse(
        name=str(update.ref.name),
        target=str(update.ref.target),
        generation=update.ref.generation,
        lifecycle=str(update.ref.lifecycle),
        expires_at_us=update.ref.expires_at_us,
    )


@router.post(
    "/envs/{org}/{env}/refs/{ref:path}", status_code=status.HTTP_201_CREATED, tags=["refs"]
)
def create_ref(
    org: str,
    env: str,
    ref: str,
    body: s.CreateRefRequest,
    auth: WriteAuth,
    idempotency_key: IdempotencyKey = None,
) -> s.RefResponse:
    """Create a branch. Costs no bytes — one metadata row over existing content."""
    del org, env
    update = RefService(auth.ledger).create(
        auth.env_id,
        RefName(ref),
        target=ObjectName.parse(body.target) if body.target else None,
        principal=str(auth.principal),
        ephemeral=body.ephemeral,
        ttl_days=body.ttl_days,
        idempotency_key=idempotency_key,
    )
    return s.RefResponse(
        name=str(update.ref.name),
        target=str(update.ref.target),
        generation=update.ref.generation,
        lifecycle=str(update.ref.lifecycle),
        expires_at_us=update.ref.expires_at_us,
    )


@router.delete(
    "/envs/{org}/{env}/refs/{ref:path}", status_code=status.HTTP_204_NO_CONTENT, tags=["refs"]
)
def delete_ref(
    org: str,
    env: str,
    ref: str,
    auth: WriteAuth,
    expected_generation: Annotated[int, Query()],
    idempotency_key: IdempotencyKey = None,
) -> Response:
    """Discard a branch. Instant; reclaiming storage happens later."""
    del org, env
    RefService(auth.ledger).delete(
        auth.env_id,
        RefName(ref),
        expected_generation=expected_generation,
        principal=str(auth.principal),
        idempotency_key=idempotency_key,
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# ─────────────────────────────────────────────────────────────────────────────
# Writing objects
# ─────────────────────────────────────────────────────────────────────────────


@router.post("/envs/{org}/{env}/sessions", status_code=status.HTTP_201_CREATED, tags=["write"])
def begin_write(org: str, env: str, auth: WriteAuth) -> s.SessionResponse:
    """Open a write session. Everything uploaded under it is a GC root."""
    del org, env
    session = auth.ledger.repo.begin_write(auth.env_id, principal=str(auth.principal))
    return s.SessionResponse(
        session_id=str(session.session_id), expires_at_us=session.expires_at_us
    )


@router.post("/envs/{org}/{env}/sessions/{session}/renew", tags=["write"])
def renew_write(org: str, env: str, session: str, auth: WriteAuth) -> s.SessionResponse:
    """Extend an open session's lease.

    The lease is meant to be "renewable" and it was not: a write that took
    longer than the TTL — a first commit of a very large environment over a slow
    link — lost its protection mid-upload, and the collector became entitled to
    the objects it had already sent.

    Renewing an expired session is a 404 rather than a fresh lease. The content
    it protected may already be gone, so silently issuing a new one would report
    safety that no longer exists.
    """
    del org, env
    renewed = auth.ledger.repo.renew_session(auth.env_id, SessionId(session))
    return s.SessionResponse(
        session_id=str(renewed.session_id), expires_at_us=renewed.expires_at_us
    )


@router.delete(
    "/envs/{org}/{env}/sessions/{session}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["write"],
)
def end_write(org: str, env: str, session: str, auth: WriteAuth) -> Response:
    """Close a session, releasing its lease.

    Optional — an abandoned session needs no cleanup by anyone, because the lease
    is a timer rather than a lock. Calling it is how a well-behaved writer stops
    pinning content it decided not to publish.
    """
    del org, env
    auth.ledger.repo.end_session(auth.env_id, SessionId(session))
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/envs/{org}/{env}/objects/missing", tags=["write"])
def has_objects(
    org: str, env: str, body: s.HasObjectsRequest, auth: WriteAuth
) -> s.HasObjectsResponse:
    """Which of these objects must still be uploaded.

    Deduplication as *transfer*: the client offers hashes and sends only what
    Ledger has never seen. The answer includes swept hashes, so a tombstoned
    object is re-uploaded rather than resurrected.
    """
    del org, env
    names = [ObjectName.parse(n) for n in body.names]
    return s.HasObjectsResponse(missing=[str(n) for n in sorted(auth.ledger.store.missing(names))])


@router.put("/envs/{org}/{env}/objects/{name}", status_code=status.HTTP_201_CREATED, tags=["write"])
def put_object(
    org: str,
    env: str,
    name: str,
    # Declared explicitly as an octet-stream body: without it FastAPI would try
    # to parse the bytes as JSON and reject every upload as malformed.
    body: Annotated[bytes, Body(media_type="application/octet-stream")],
    auth: WriteAuth,
    session_id: Annotated[str | None, Header(alias="X-Ledger-Session")] = None,
) -> Response:
    """Store one object under the name the caller claims for it.

    The server rehashes and rejects any mismatch, and re-encodes to reject a
    non-canonical spelling. Under global deduplication that check is a tenancy
    boundary, not a formality: a writer able to store arbitrary bytes under a
    chosen name would poison an environment it cannot reach.
    """
    del org, env
    object_name = ObjectName.parse(name)
    outcome = auth.ledger.store.put(object_name, body)

    if session_id is not None:
        from src.ids import SessionId

        auth.ledger.repo.record_uploaded(auth.env_id, SessionId(session_id), [object_name])

    return Response(
        status_code=status.HTTP_201_CREATED if outcome.created else status.HTTP_200_OK,
        headers={"X-Ledger-Created": "1" if outcome.created else "0"},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Reading
# ─────────────────────────────────────────────────────────────────────────────


@router.get("/envs/{org}/{env}/commits/{commit}", tags=["read"])
def get_commit(org: str, env: str, commit: str, auth: ReadAuth) -> s.CommitModel:
    del org, env
    name = ObjectName.parse(commit)
    auth.state.policy.authorize_commit(auth.capability, auth.env_id, name)
    return _commit_model(name, auth.ledger.store.get_as(name, Commit))


@router.get("/envs/{org}/{env}/commits/{commit}/log", tags=["read"])
def log(
    org: str, env: str, commit: str, auth: ReadAuth, limit: Annotated[int, Query(ge=1, le=500)] = 50
) -> s.LogResponse:
    """Walk history. A plain read: snapshots over shared content, not a delta replay."""
    del org, env
    name = ObjectName.parse(commit)
    auth.state.policy.authorize_commit(auth.capability, auth.env_id, name)
    return s.LogResponse(
        commits=[
            _commit_model(entry.name, entry.commit)
            for entry in auth.state.commits.walk(name, limit=limit)
        ]
    )


@router.get("/envs/{org}/{env}/commits/{commit}/tree/{path:path}", tags=["read"])
def list_directory(
    org: str,
    env: str,
    commit: str,
    path: str,
    auth: ReadAuth,
    after: Annotated[str | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
) -> s.ListDirResponse:
    """List a directory, with a cursor that is simply the last name you saw.

    A 134-million-entry directory pages exactly like a six-entry one, and under
    immutability the listing is *stable*: entries cannot shift, appear twice or
    vanish between pages.
    """
    del org, env
    tree = _root_tree(auth, ObjectName.parse(commit))
    if path:
        resolved = resolve_path(auth.ledger.store, tree, path)
        if resolved.kind is not EntryKind.TREE:
            raise InvalidRequest("that path is a file, not a directory", path=path)
        tree = resolved.target

    page = list_dir(auth.ledger.store, tree, after=after.encode() if after else None, limit=limit)
    return s.ListDirResponse(
        entries=[_entry_model(e) for e in page.entries],
        cursor=page.cursor.decode(errors="replace") if page.cursor else None,
    )


@router.get(
    "/envs/{org}/{env}/commits/{commit}/file/{path:path}",
    tags=["read"],
    # The response is either raw bytes or a description of where to get
    # them, so there is no single model to generate.
    response_model=None,
)
def read_file(
    org: str,
    env: str,
    commit: str,
    path: str,
    auth: ReadAuth,
    offset: Annotated[int, Query(ge=0)] = 0,
    length: Annotated[int | None, Query(ge=0)] = None,
    inline: Annotated[bool, Query(description="Return bytes rather than a URL.")] = True,
) -> Response | s.ReadFileResponse:
    """Read one file, or a byte range of it. No clone, no size limit.

    With ``inline=false`` the response is a short-lived ticket-bearing URL and
    the bytes are served by a separate route: the service resolves, the edge
    delivers, and the API plane stays off the bandwidth path.
    """
    del org, env
    tree = _root_tree(auth, ObjectName.parse(commit))
    resolved = resolve_path(auth.ledger.store, tree, path)
    if resolved.kind is EntryKind.TREE:
        raise InvalidRequest("that path is a directory, not a file", path=path)

    if not inline:
        ticket = auth.state.tickets.issue(resolved.target, auth.principal)
        return s.ReadFileResponse(
            path=path,
            size=resolved.entry.size,
            object_name=str(resolved.target),
            content_url=(
                f"/v1/content/{resolved.target}?ticket={ticket}&principal={auth.principal}"
            ),
        )

    payload = BlobReader(auth.ledger.store, resolved.target).read(offset, length)
    return Response(content=payload, media_type="application/octet-stream")


@router.get("/content/{name}", tags=["read"])
def content(
    name: str,
    state: LedgerDep,
    ticket: Annotated[str, Query()],
    principal: Annotated[str, Query()],
) -> Response:
    """Serve bytes against a ticket.

    Deliberately **not** authorized by token scope: the ticket already encodes an
    authorization decision made against a *path* in a commit. Possessing the hash
    alone is not enough, which is the point — hashes leak through logs and diffs,
    and a bare-hash read would make a hash a credential.
    """
    object_name = ObjectName.parse(name)
    state.tickets.verify(object_name, Principal(principal), ticket)

    framed = state.ledger.store.get(object_name)
    from src.format.codec import HEADER_BYTES, peek_kind

    if peek_kind(framed) is ObjectKind.CHUNK:
        return Response(content=framed[HEADER_BYTES:], media_type="application/octet-stream")

    payload = BlobReader(state.ledger.store, object_name).read()
    return Response(content=payload, media_type="application/octet-stream")


@router.get("/envs/{org}/{env}/diff", tags=["read"])
def diff(
    org: str,
    env: str,
    auth: ReadAuth,
    before: Annotated[str, Query(description="Commit to compare from.")],
    after: Annotated[str, Query(description="Commit to compare to.")],
    limit: Annotated[int, Query(ge=1, le=5000)] = 1000,
) -> s.DiffResponse:
    """Compare two versions.

    Cost is proportional to what changed: an unchanged subtree has an unchanged
    hash, so an entire branch is dismissed by comparing two names.
    """
    del org, env
    from src.fs.diff import diff_trees

    left = _root_tree(auth, ObjectName.parse(before))
    right = _root_tree(auth, ObjectName.parse(after))

    changes: list[s.ChangeModel] = []
    for change in diff_trees(auth.ledger.store, left, right):
        if len(changes) == limit:
            return s.DiffResponse(changes=changes, truncated=True)
        changes.append(
            s.ChangeModel(
                path=change.display_path,
                kind=str(change.kind),
                size_delta=change.size_delta,
            )
        )
    return s.DiffResponse(changes=changes, truncated=False)


# ─────────────────────────────────────────────────────────────────────────────
# Notes, operations
# ─────────────────────────────────────────────────────────────────────────────


@router.put("/envs/{org}/{env}/commits/{commit}/notes/{namespace}", tags=["notes"])
def put_note(
    org: str, env: str, commit: str, namespace: str, body: s.NoteRequest, auth: AnnotateAuth
) -> s.PutNoteResponse:
    """Attach a fact to a commit from outside it.

    It cannot go inside: a commit's name is the hash of its content, so
    appending a verdict would change its identity and break every reference to
    it. Notes are **not** GC roots — annotating a commit never changes what
    storage costs.
    """
    del org, env
    name = ObjectName.parse(commit)
    auth.ledger.repo.put_note(
        auth.env_id, name, namespace, body.body, principal=str(auth.principal)
    )
    return s.PutNoteResponse(commit=str(name), namespace=namespace)


@router.get("/envs/{org}/{env}/commits/{commit}/notes/{namespace}", tags=["notes"])
def get_note(org: str, env: str, commit: str, namespace: str, auth: ReadAuth) -> s.NoteResponse:
    del org, env
    note = auth.ledger.repo.get_note(auth.env_id, ObjectName.parse(commit), namespace)
    return s.NoteResponse(
        commit=str(note.commit),
        namespace=note.namespace,
        body=dict(note.body),
        updated_by=note.updated_by,
        updated_at_us=note.updated_at_us,
    )


@router.get("/envs/{org}/{env}/operations", tags=["history"])
def list_operations(
    org: str, env: str, auth: ReadAuth, limit: Annotated[int, Query(ge=1, le=500)] = 50
) -> s.ListOpsResponse:
    """Every mutation, in order, with the principal that made it."""
    del org, env
    return s.ListOpsResponse(
        operations=[
            s.OpEntryModel(
                sequence=entry.sequence,
                kind=str(entry.kind),
                ref=str(entry.ref) if entry.ref else None,
                before=str(entry.before) if entry.before else None,
                after=str(entry.after) if entry.after else None,
                principal=entry.principal,
                at_us=entry.at_us,
            )
            for entry in auth.ledger.repo.list_ops(auth.env_id, limit=limit)
        ]
    )


@router.post("/envs/{org}/{env}/operations/{sequence}/undo", tags=["history"])
def undo(
    org: str,
    env: str,
    sequence: int,
    auth: WriteAuth,
    idempotency_key: IdempotencyKey = None,
) -> s.RefResponse:
    """Undo an operation — an ordinary ref update, not a rewind.

    It expects the generation the ref has *now*, so a writer who moved it in the
    meantime produces the usual conflict rather than being silently overwritten.
    """
    del org, env
    update = auth.ledger.repo.undo(
        auth.env_id,
        sequence,
        principal=str(auth.principal),
        idempotency_key=idempotency_key,
    )
    return s.RefResponse(
        name=str(update.ref.name),
        target=str(update.ref.target),
        generation=update.ref.generation,
        lifecycle=str(update.ref.lifecycle),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Build and sync
# ─────────────────────────────────────────────────────────────────────────────


@router.post("/envs/{org}/{env}/builds", status_code=status.HTTP_202_ACCEPTED, tags=["build"])
def trigger_build(
    org: str, env: str, body: s.TriggerBuildRequest, auth: BuildAuth
) -> s.TriggerBuildResponse:
    """Ask for a build — reruns and backfills only.

    An ordinary commit needs no call: the ref update itself produces the event
    that queues the build, in the same transaction that published it. There is
    nothing here to forget to call.

    202, not 200: the build has been *accepted*, and its result appears under
    the commit when a worker gets to it.
    """
    del org, env
    from src.build.pipeline import trigger

    ref = RefName(body.ref)
    commit = auth.ledger.repo.get_ref(auth.env_id, ref).target
    queued = trigger(auth.ledger, auth.env_id, ref, rebuild=body.rebuild)
    return s.TriggerBuildResponse(commit=str(commit), ref=str(ref), queued=queued)


@router.get("/envs/{org}/{env}/commits/{commit}/build", tags=["build"])
def get_build(org: str, env: str, commit: str, auth: ReadAuth) -> s.BuildResponse:
    """The build result for a commit.

    Global and keyed by commit — so a fork that changed nothing reads its
    parent's result here rather than waiting for a rebuild — but reached through
    an environment, because a commit hash alone must never be enough to read
    anything.
    """
    del org, env
    from src.build.results import BuildResults

    name = ObjectName.parse(commit)
    auth.state.policy.authorize_commit(auth.capability, auth.env_id, name)
    result = BuildResults(auth.ledger.meta, clock=auth.ledger.clock).get(name)
    if result is None:
        raise NotFound("this commit has no build result", commit=commit)
    return _build_model(result)


@router.get("/builds/failures", tags=["build"])
def build_failures(
    state: LedgerDep,
    caller: AdminOnly,
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
) -> s.BuildFailuresResponse:
    """Every failed build, corpus-wide.

    One scan rather than a walk over ten million partitions — which is what makes
    a systemic build regression one signal instead of ten million silent ones. Administrative,
    because it deliberately crosses every
    environment.
    """
    del caller
    from src.build.results import BuildResults

    results = BuildResults(state.ledger.meta, clock=state.ledger.clock)
    return s.BuildFailuresResponse(
        failures=[_build_model(r) for r in results.failures(limit=limit)]
    )


def _build_model(result: BuildResult) -> s.BuildResponse:
    return s.BuildResponse(
        commit=result.commit,
        status=str(result.status),
        started_at_us=result.started_at_us,
        finished_at_us=result.finished_at_us,
        duration_us=result.duration_us,
        exit_code=result.exit_code,
        images=list(result.images),
        attempts=result.attempts,
        worker=result.worker,
        log_excerpt=result.log_excerpt,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tokens
# ─────────────────────────────────────────────────────────────────────────────


@router.post("/tokens", tags=["auth"])
def mint_token(
    body: s.MintTokenRequest, state: LedgerDep, caller: CurrentCapability
) -> s.MintTokenResponse:
    """Mint a token for a principal, narrowing the caller's own authority.

    Only ever narrows. A per-rollout token pins one environment and drops every
    operation but read, so the agent code inside — which is untrusted with
    respect to the corpus — cannot reach anything else.
    """
    from src.auth.model import parse_operations

    requested = parse_operations(body.operations) if body.operations else caller.scope.operations
    if requested & ~caller.scope.operations:
        raise Forbidden("a minted token cannot widen the authority of the token minting it")

    scope = caller.scope.narrow(operations=requested, env_id=body.env_id)
    token = state.signer.mint(
        Principal(body.principal),
        scope,
        ttl_us=body.ttl_seconds * 1_000_000,
        commit=ObjectName.parse(body.commit) if body.commit else None,
    )
    from src.auth.model import render_operations

    return s.MintTokenResponse(
        token=token,
        expires_at_us=state.ledger.clock.now_us() + body.ttl_seconds * 1_000_000,
        operations=render_operations(scope.operations),
    )


@router.get("/metrics", tags=["ops"], response_class=PlainTextResponse)
def metrics(state: LedgerDep) -> PlainTextResponse:
    """The assumptions this design rests on, as numbers.

    Not service metrics. Latency and error rates say whether the system is
    working; these say whether the *design* is still the right one — and several
    of them are properties of how environments are built rather than of Ledger,
    which is exactly why they are watched rather than assumed.

    Unauthenticated, like ``/healthz``: it reports corpus-wide aggregates and
    names no environment, no principal and no object. A deployment that wants it
    private puts it behind the same network boundary it puts any other
    operational endpoint behind.
    """
    from src.metrics import render, snapshot

    return PlainTextResponse(content=render(snapshot(state.ledger)))


@router.get("/healthz", tags=["ops"])
def health(state: LedgerDep) -> dict[str, object]:
    """Liveness, plus the format fingerprint.

    Two deployments share a corpus if and only if they share that value, so
    surfacing it here turns "why is nothing deduplicating" into one comparison.
    """
    from src.format.constants import FORMAT_FINGERPRINT

    return {
        "status": "ok",
        "format_fingerprint": FORMAT_FINGERPRINT,
        "shards": state.ledger.meta.shard_count,
    }
