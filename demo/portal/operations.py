"""The operations ``/v1`` does not expose, and why each one is missing.

Everything the browser can do against the product API, it does — create an
environment, move a ref, fork, branch, undo, mint a token, trigger a build,
read a file. What is left over lands here, and it is worth being precise about
why, because "the demo needed an endpoint" is exactly the kind of reasoning that
grows a second API.

**Content that originates on the server.** ``/v1`` has a client upload objects
under names it computed itself, and the server rehashes and rejects a mismatch.
A browser cannot chunk a 3.81 MiB dataset with FastCDC and hash it with BLAKE3 —
and *teaching* it to would be the worst possible outcome, because a second
encoder that disagreed by one byte would name every object differently. The jj
backend does not compute a name for the same reason. So the four content routes
below take a description of the edit and let the server's own ingester do the
naming: ``commit`` (a directory on this machine), ``files``, ``patch``, and
``images``.

**Work that talks to this machine.** ``import`` reads a git repository from the
local filesystem; ``images`` shells out to ``docker save``. Both are properties
of the host rather than of Ledger, which is why the CLI has them and the service
does not.

**Maintenance.** Collection is a scheduled job with a circuit breaker, not
something a caller asks for over HTTP, so ``/v1`` has no route for it and should
not. The demonstration needs to run one on camera, so it lives here — and
``clock`` beside it, because the interesting thing about collection is *when* it
is allowed to take something.

One rule holds across all of them: **no value from a request ever reaches a
subprocess or the filesystem as a path.** A request selects a git repository or
a base image by key from a fixed table; there is no free-form path anywhere
below.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Annotated, Final

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from demo.portal.context import ROOT, DemoDep
from src.api.deps import Authorized, require, require_global
from src.auth.model import Operation
from src.auth.tokens import Capability
from src.errors import InvalidRequest
from src.format.constants import MODE_REGULAR, EntryKind
from src.format.model import Blob, Commit, TreeEntry
from src.fs.blob import BlobReader
from src.fs.edit import empty_tree, set_path
from src.fs.path import parse_path
from src.fs.tree import resolve_path
from src.ids import ChangeId, ObjectName, RefName
from src.meta.models import Ref
from src.migrate.from_git import GitImporter
from src.oci.model import Platform
from src.runtime.ingest import IngestStats
from src.service.commits import CommitService
from src.service.images import ImageService, host_platform

__all__ = ["BASE_IMAGES", "REPOSITORIES", "router"]

router = APIRouter(prefix="/demo", tags=["demo"], include_in_schema=False)

WriteAuth = Annotated[Authorized, require(Operation.WRITE)]
AdminOnly = Annotated[Capability, require_global(Operation.ADMIN)]

#: Git repositories the import card may convert, by key. A *table* rather than a
#: path in the request body: a demo endpoint that accepted a path would be a
#: read primitive over the whole filesystem, dressed as a convenience.
REPOSITORIES: Final[dict[str, Path]] = {"ledger": ROOT}

#: Base images the container card may store, by key. Selected the same way, and
#: for the same reason. ``demo/stage.py`` checks both are pulled before a take,
#: because ``docker save`` fails on an image this machine has never seen.
BASE_IMAGES: Final[dict[str, str]] = {
    "alpine": "alpine:latest",
    "busybox": "busybox:latest",
}


# ─────────────────────────────────────────────────────────────────────────────
# What every content route reports
# ─────────────────────────────────────────────────────────────────────────────


class WriteResult(BaseModel):
    """A published version, and what publishing it cost.

    The cost fields are the whole point of the demonstration, so they are the
    same four the CLI prints and ``demo/e2e.py`` measures — offered against
    created, offered against stored. A route that reported only "ok" would make
    every card on the page an assertion instead of a measurement.
    """

    commit: str
    ref: str
    generation: int
    objects_offered: int
    objects_created: int
    bytes_offered: int
    bytes_stored: int
    replayed: bool = False


def _written(
    commit: str, ref: str, generation: int, stats: IngestStats, *, replayed: bool = False
) -> WriteResult:
    return WriteResult(
        commit=commit,
        ref=ref,
        generation=generation,
        objects_offered=stats.objects_offered,
        objects_created=stats.objects_created,
        bytes_offered=stats.bytes_offered,
        bytes_stored=stats.bytes_stored,
        replayed=replayed,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Content
# ─────────────────────────────────────────────────────────────────────────────


class CommitWorkspaceRequest(BaseModel):
    ref: str = "refs/heads/main"
    message: str = "initial version"
    author: str = "agent-17"


@router.post("/envs/{org}/{env}/commit")
def commit_workspace(
    org: str, env: str, body: CommitWorkspaceRequest, auth: WriteAuth, demo: DemoDep
) -> WriteResult:
    """Commit the sample environment sitting on this machine.

    The directory is fixed — ``demo/stage.py`` builds it, seeded, so every take
    commits byte-identical content and the numbers on camera are the same
    numbers twice running.
    """
    del org, env
    if not demo.workspace.is_dir():
        raise InvalidRequest(
            "the sample environment has not been built", fix="uv run python demo/stage.py"
        )
    result = CommitService(auth.ledger).commit(
        auth.env_id,
        RefName(body.ref),
        demo.workspace,
        author=body.author,
        message=body.message,
    )
    return _written(
        str(result.commit),
        str(result.ref),
        result.generation,
        result.stats,
        replayed=result.replayed,
    )


class WriteFileRequest(BaseModel):
    path: str
    text: str
    ref: str = "refs/heads/main"
    message: str = "edit a file"
    author: str = "agent-17"
    #: A jj change id, carried so two commits can be shown to be two versions of
    #: one *change* rather than two unrelated ones. Generated when absent.
    change_id: str | None = None


@router.post("/envs/{org}/{env}/files")
def write_file(
    org: str, env: str, body: WriteFileRequest, auth: WriteAuth, demo: DemoDep
) -> WriteResult:
    """Replace one text file and publish the result as the next version."""
    del org, env, demo
    return _edit_and_publish(
        auth,
        ref=RefName(body.ref),
        path=body.path,
        content=body.text.encode(),
        message=body.message,
        author=body.author,
        change_id=ChangeId(body.change_id) if body.change_id else None,
    )


class PatchRequest(BaseModel):
    path: str
    offset: int = Field(ge=0)
    length: int = Field(gt=0, le=1 << 20)
    fill: str = Field(min_length=1, max_length=1)
    ref: str = "refs/heads/main"
    message: str = "patch a dataset"
    author: str = "agent-17"


@router.post("/envs/{org}/{env}/patch")
def patch_file(
    org: str, env: str, body: PatchRequest, auth: WriteAuth, demo: DemoDep
) -> WriteResult:
    """Overwrite a byte range inside an existing file, in place.

    The card this serves is the sharpest claim: changing 64 bytes in
    the middle of a multi-megabyte dataset should cost the *region*, not the
    file. Splicing here rather than in the browser keeps the whole artifact off
    the wire, which is itself part of the claim.
    """
    del org, env, demo
    tree = _tree_of(auth, RefName(body.ref))
    resolved = resolve_path(auth.ledger.store, tree, body.path)
    if resolved.kind is EntryKind.TREE:
        raise InvalidRequest("that path is a directory", path=body.path)

    content = bytearray(BlobReader(auth.ledger.store, resolved.target).read())
    if body.offset + body.length > len(content):
        raise InvalidRequest(
            "that range runs past the end of the file",
            path=body.path,
            size=len(content),
        )
    content[body.offset : body.offset + body.length] = body.fill.encode() * body.length

    return _edit_and_publish(
        auth,
        ref=RefName(body.ref),
        path=body.path,
        content=bytes(content),
        message=body.message,
        author=body.author,
    )


def _edit_and_publish(
    auth: Authorized,
    *,
    ref: RefName,
    path: str,
    content: bytes,
    message: str,
    author: str,
    change_id: ChangeId | None = None,
) -> WriteResult:
    """Ingest ``content``, put it at ``path``, and publish the tree.

    Through ``CommitService.publish``, so this gets the ordinary four-phase
    write: a leased session, the closure recorded before the ref moves, the
    generation compare-and-swap, and keep-set graduation after. There is no
    demo-shaped write path — only a demo-shaped way of producing the tree, which
    is exactly what ``publish`` takes as an argument.
    """
    ledger = auth.ledger
    name = parse_path(path)[-1]

    def build(current: Ref | None) -> tuple[ObjectName, IngestStats]:
        root = (
            ledger.store.get_as(current.target, Commit).tree
            if current is not None
            else empty_tree(ledger.store, shape=ledger.shape_params)
        )
        blob, stats = ledger.ingester.ingest_bytes(content)
        entry = TreeEntry(
            name=name,
            kind=EntryKind.BLOB,
            target=blob,
            mode=MODE_REGULAR,
            size=len(content),
        )
        edit = set_path(ledger.store, root, path, entry, shape=ledger.shape_params)
        return edit.tree, stats + IngestStats.counting(edit.written)

    result = CommitService(ledger).publish(
        auth.env_id, ref, build, author=author, message=message, change_id=change_id
    )
    return _written(str(result.commit), str(result.ref), result.generation, result.stats)


# ─────────────────────────────────────────────────────────────────────────────
# Images
# ─────────────────────────────────────────────────────────────────────────────


class AddImageRequest(BaseModel):
    #: A key of ``BASE_IMAGES``, never a reference this request invented.
    base: str = "alpine"
    name: str = "app"
    ref: str = "refs/heads/main"
    author: str = "agent-17"


class AddImageResponse(WriteResult):
    image: str
    manifest_digest: str
    layers: int
    layers_reused: int
    bytes_compressed: int
    bytes_uncompressed: int
    pull_with: str


@router.post("/envs/{org}/{env}/images")
def add_image(
    org: str, env: str, body: AddImageRequest, auth: WriteAuth, demo: DemoDep
) -> AddImageResponse:
    """Store a container image *inside* the version.

    Not a tag pointing at a registry — the layers become ordinary objects in the
    commit, which is what makes restoring an old version restore the same
    containers. The digest the registry then serves is computed from those
    objects, so a `docker pull` against /v2 and this call are looking at one thing.
    """
    del org, env
    reference = BASE_IMAGES.get(body.base)
    if reference is None:
        raise InvalidRequest(
            "unknown base image", requested=body.base, available=sorted(BASE_IMAGES)
        )

    workspace = Path(tempfile.mkdtemp(prefix="ledger-portal-image-"))
    try:
        archive = workspace / "image.tar"
        # A fixed argument list. ``reference`` came out of the table above, not
        # out of the request, and there is no shell anywhere in this call.
        export = subprocess.run(
            ["docker", "save", reference, "-o", str(archive)],
            capture_output=True,
            text=True,
            check=False,
        )
        if export.returncode != 0:
            raise InvalidRequest(
                "docker save failed",
                image=reference,
                error=export.stderr.strip()[:400],
                fix=f"docker pull {reference}",
            )
        result = ImageService(auth.ledger).add(
            auth.env_id,
            RefName(body.ref),
            archive,
            image=body.name,
            author=body.author,
            platform=_platform(),
        )
    finally:
        shutil.rmtree(workspace, ignore_errors=True)

    written = _written(str(result.commit), str(result.ref), result.generation, result.stats)
    tag = body.ref.removeprefix("refs/heads/")
    return AddImageResponse(
        **written.model_dump(),
        image=result.image,
        manifest_digest=str(result.manifest_digest),
        layers=result.layers,
        layers_reused=result.layers_reused,
        bytes_compressed=result.bytes_compressed,
        bytes_uncompressed=result.bytes_uncompressed,
        pull_with=f"127.0.0.1:{demo.port}/{auth.ledger.repo.get_env(auth.env_id).name}"
        f"/{result.image}:{tag}",
    )


def _platform() -> Platform:
    """This machine's platform, so a stored image is one it could also run."""
    return host_platform()


# ─────────────────────────────────────────────────────────────────────────────
# Migration
# ─────────────────────────────────────────────────────────────────────────────


class ImportRequest(BaseModel):
    #: A key of ``REPOSITORIES``.
    repository: str = "ledger"
    ref: str = "refs/heads/imported"
    limit: int = Field(default=50, ge=1, le=1000)
    author: str = "ledger-import"


class ImportResponse(BaseModel):
    head: str
    ref: str
    commits: int
    trees: int
    blobs: int
    reused: int
    git_bytes: int
    objects_created: int
    bytes_stored: int
    dedup_ratio: float
    unsupported: list[str]


@router.post("/envs/{org}/{env}/import")
def import_repository(
    org: str, env: str, body: ImportRequest, auth: WriteAuth, demo: DemoDep
) -> ImportResponse:
    """Convert a git history into Ledger commits.

    A conversion, not a bridge: each git object becomes the Ledger object that
    means the same thing, blobs are re-chunked rather than copied, and nothing
    afterwards points back at git. Re-running it converts nothing — every git
    SHA is already in the alternate-digest index — which is the number worth
    watching on the second click.
    """
    del org, env, demo
    repository = REPOSITORIES.get(body.repository)
    if repository is None:
        raise InvalidRequest(
            "unknown repository", requested=body.repository, available=sorted(REPOSITORIES)
        )

    report = GitImporter(auth.ledger, skip_unsupported=True).import_repository(
        repository,
        auth.env_id,
        RefName(body.ref),
        limit=body.limit,
        author=body.author,
    )
    return ImportResponse(
        head=str(report.head),
        ref=body.ref,
        commits=report.commits,
        trees=report.trees,
        blobs=report.blobs,
        reused=report.reused,
        git_bytes=report.git_bytes,
        objects_created=report.stats.objects_created,
        bytes_stored=report.stats.bytes_stored,
        dedup_ratio=report.dedup_ratio,
        unsupported=[f"{p.path} ({p.reason})" for p in report.unsupported],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Builds
# ─────────────────────────────────────────────────────────────────────────────


class DrainResponse(BaseModel):
    events: int
    enqueued: int
    ignored: int
    built: int
    cache_hits: int
    #: The number that makes "a build is a pure function of a commit" checkable.
    #: A fork asking for a commit that has already been built must leave it
    #: exactly where it was.
    runner_invocations: int
    platform_deliveries: int


@router.post("/builds/drain")
def drain_builds(caller: AdminOnly, demo: DemoDep) -> DrainResponse:
    """Run the dispatcher and a worker, once, in this process.

    In production these are separate processes tailing the change stream, and
    there is no reason for the API plane to host either. The portal hosts them
    so a click can advance the pipeline on camera — the *code* is the product's,
    only the scheduling is the demo's.
    """
    del caller
    builds = demo.builds
    report = builds.dispatcher.poll()
    outcomes = builds.worker.drain()
    return DrainResponse(
        events=report.events,
        enqueued=report.enqueued,
        ignored=report.ignored,
        built=len(outcomes),
        cache_hits=sum(1 for outcome in outcomes if outcome.cache_hit),
        runner_invocations=builds.runner.count,
        platform_deliveries=len(builds.platform.deliveries),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Damage
# ─────────────────────────────────────────────────────────────────────────────


class DamageRequest(BaseModel):
    path: str
    ref: str = "refs/heads/main"
    #: Flip the byte back. The damage is an XOR, so this is the same operation
    #: applied twice — there is no saved copy to lose and nothing to get wrong.
    restore: bool = False


class DamageResponse(BaseModel):
    object: str
    restored: bool


@router.post("/envs/{org}/{env}/damage")
def damage(
    org: str, env: str, body: DamageRequest, auth: WriteAuth, demo: DemoDep
) -> DamageResponse:
    """Rot one byte of one chunk on the medium, deliberately.

    "Damaged data is detected, not served" is a claim that cannot be
    demonstrated without damaging something, and a demonstration that damaged it
    by *unplugging a disk* is not one you can do twice. So this reaches past the
    store and flips a bit in the file underneath it — the one place in this
    package that touches storage as bytes rather than as objects.

    It is an XOR with a fixed mask at a fixed offset, so calling it again puts
    the byte back exactly. No value from the request reaches the filesystem: the
    path selects an *object*, and the object's own hash is its location.
    """
    del org, env
    tree = _tree_of(auth, RefName(body.ref))
    resolved = resolve_path(auth.ledger.store, tree, body.path)
    if resolved.kind is EntryKind.TREE:
        raise InvalidRequest("that path is a directory", path=body.path)

    blob = auth.ledger.store.get_as(resolved.target, Blob)
    if not blob.entries:
        raise InvalidRequest("that file has no chunks to damage", path=body.path)
    target = blob.entries[0].target

    # ``objects/<aa>/<bb>/<hash>`` — an object's key is its own hash, so no
    # index is consulted to find it.
    key = target.hex
    on_disk = demo.data_dir / "objects" / key[:2] / key[2:4] / key
    if not on_disk.is_file():  # pragma: no cover - the store just read it
        raise InvalidRequest("that object is not on this medium", object=str(target))

    stored = bytearray(on_disk.read_bytes())
    # Byte 0 is the compression codec; byte 1 is the first byte of the payload.
    stored[DAMAGE_OFFSET] ^= DAMAGE_MASK
    on_disk.write_bytes(bytes(stored))
    return DamageResponse(object=str(target), restored=body.restore)


#: Past the one-byte compression header, so the frame is still readable and the
#: failure is the one worth showing: the bytes no longer hash to the name they
#: were asked for.
DAMAGE_OFFSET: Final = 1
DAMAGE_MASK: Final = 0xFF


# ─────────────────────────────────────────────────────────────────────────────
# Maintenance, and time
# ─────────────────────────────────────────────────────────────────────────────


class KeepSetsResponse(BaseModel):
    environments: dict[str, int]


@router.post("/keepsets/rebuild")
def rebuild_keep_sets(caller: AdminOnly, demo: DemoDep) -> KeepSetsResponse:
    """Recompute every environment's keep-set from its live refs.

    Keep-sets are *maintained*, not recomputed — adding is free, and only
    subtraction costs anything. But two things make one go stale on a timer
    rather than on an event: an operation-log entry ageing out stops protecting
    what undo could have reached, and that moment belongs to no request. So a
    scheduled rebuild is part of normal operation, and running it before a collection
    is the maintenance sequence rather than a demo shortcut.
    """
    del caller
    ledger = demo.ledger
    sizes = {
        str(name): ledger.gc.rebuild_keep_set(str(name))
        for name in ledger.repo.list_envs(limit=1000)
    }
    return KeepSetsResponse(environments=sizes)


class GcPlanResponse(BaseModel):
    candidates: int
    bytes_reclaimable: int
    corpus_objects: int
    corpus_bytes: int


@router.post("/gc/plan")
def gc_plan(caller: AdminOnly, demo: DemoDep) -> GcPlanResponse:
    """What a collection *would* take, without taking it."""
    del caller
    plan = demo.ledger.gc.plan()
    objects, stored = demo.ledger.store.catalog.total()
    return GcPlanResponse(
        candidates=len(plan.candidates),
        bytes_reclaimable=plan.bytes_reclaimable,
        corpus_objects=objects,
        corpus_bytes=stored,
    )


class GcRunResponse(BaseModel):
    deleted: int
    bytes_freed: int
    corpus_objects: int
    corpus_bytes: int


@router.post("/gc/run")
def gc_run(caller: AdminOnly, demo: DemoDep) -> GcRunResponse:
    """Sweep, for real. Tombstoned, so dedup cannot resurrect what it took."""
    del caller
    report = demo.ledger.gc.run(enforce=True)
    objects, stored = demo.ledger.store.catalog.total()
    return GcRunResponse(
        deleted=report.deleted,
        bytes_freed=report.bytes_freed,
        corpus_objects=objects,
        corpus_bytes=stored,
    )


class AdvanceRequest(BaseModel):
    days: float = Field(gt=0, le=3650)


class ClockResponse(BaseModel):
    now_us: int


@router.post("/clock/advance")
def advance_clock(body: AdvanceRequest, caller: AdminOnly, demo: DemoDep) -> ClockResponse:
    """Move the demonstration's clock forward.

    Every timed guarantee reads this clock: a write session's
    lease, an ephemeral branch's expiry, and the grace period collection waits
    out. None of them can be shown to an audience in real time, and a demo that
    skipped them would skip the most interesting property in the system — that
    how long the operation log is kept *is* how long reclamation takes.
    """
    del caller
    return ClockResponse(now_us=demo.clock.advance_days(body.days))


class ResetResponse(BaseModel):
    now_us: int
    environments: int
    corpus_objects: int


@router.post("/reset")
def reset(request: Request, caller: AdminOnly) -> ResetResponse:
    """Throw the corpus away and start the walkthrough from nothing.

    Recording has failure modes the software does not, and the worst of them is
    state left over from the previous take. Restarting the server would cost a
    minute of dead air on camera, so the Ledger is closed, its directory
    removed, and a fresh one assembled in place.
    """
    del caller
    state = request.app.state.portal.reset()
    objects, _ = state.ledger.store.catalog.total()
    return ResetResponse(
        now_us=state.clock.now_us(),
        environments=len(state.ledger.repo.list_envs(limit=1000)),
        corpus_objects=objects,
    )


def _tree_of(auth: Authorized, ref: RefName) -> ObjectName:
    commit = auth.ledger.repo.get_ref(auth.env_id, ref).target
    return auth.ledger.store.get_as(commit, Commit).tree
