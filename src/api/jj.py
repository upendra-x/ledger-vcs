"""The jj-shaped surface a ``jj_lib::backend::Backend`` talks to.

The strongest answer to *"native for agents"* is not to
build a familiar-looking CLI but to have agents run **real jj**, with Ledger as
its storage: commits, trees and files are Ledger's objects, and chunking and
sharded trees sit below the level jj addresses.

This is that seam. It is deliberately *not* a generic object API — it speaks jj's
model (files, symlinks, trees, commits with two signatures and a change id) and
maps it onto Ledger's four object types.

**The Rust backend never computes an object name.** It posts a jj-shaped object;
Ledger encodes it canonically and returns the name it got. That is the whole
reason this endpoint exists rather than the backend hashing locally: a second
implementation of the encoding — in another language, maintained separately — is
exactly the silent-divergence failure the frozen format exists to prevent. Two
encoders that agree today and disagree after one careless edit would fork the
corpus, and nothing would report an error.

The cost is a round trip per written object. For a local jj that is nothing, and
it buys one canonical encoder forever.

Two mappings are lossy in one direction and are handled explicitly:

* jj's signatures carry a name, an email and a timezone; a Ledger commit carries
  one author string and one timestamp. The rest rides in the commit's
  ``metadata``, so a jj commit round-trips exactly — and a commit made by
  ``ledger commit`` is still readable by jj, with the missing parts synthesized.
* jj represents a *conflicted* tree as a ``Merge`` with more than one term.
  Ledger has no such object — its merge refuses an overlap rather than storing
  one — so a conflicted commit is refused with the reason. That is the stated design's
  stated conflict gap, surfaced rather than papered over.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Annotated, Any

from fastapi import APIRouter, Body, Query
from fastapi.responses import Response

from src.api.deps import Authorized, require
from src.auth.model import Operation
from src.errors import InvalidRequest, NotFound
from src.format.constants import MODE_EXEC, MODE_REGULAR, EntryKind
from src.format.model import Commit, Tree, TreeEntry
from src.format.shape import build_tree
from src.fs.blob import BlobReader
from src.fs.tree import iter_entries
from src.ids import ChangeId, ObjectName

if TYPE_CHECKING:
    from src.instance import Ledger

__all__ = ["router"]

router = APIRouter(prefix="/v1/envs/{org}/{env}/jj", tags=["jj"])

JjRead = Annotated[Authorized, require(Operation.READ)]
JjWrite = Annotated[Authorized, require(Operation.WRITE)]

#: Where jj's extra commit fields live. Namespaced so they cannot collide with
#: metadata a caller of the ordinary API set.
PREFIX = "jj."


# ─────────────────────────────────────────────────────────────────────────────
# Binding an object name to the environment in the path
# ─────────────────────────────────────────────────────────────────────────────


def _visible(auth: Authorized, name: ObjectName) -> ObjectName:
    """Refuse a name this environment does not reach.

    Every route here reads an object **by bare id**, and under global
    deduplication an id is a corpus-wide address: without this, `env:read` on
    ``a/b`` would serve any object in the corpus to anyone who could guess or
    learn its hash, which is precisely the bare-hash read the policy engine
    exists to prevent. The ordinary API binds a commit to an environment by
    walking ancestry; that has nothing to walk from when jj asks for a file id.

    Membership in the environment's **keep-set** is the test, because the
    keep-set already answers exactly this question for the collector: it is the
    set of objects this environment reaches. Reusing it means authorization and
    retention cannot drift apart — anything readable here is by construction
    something collection will not take.

    A miss is ``NotFound`` rather than ``Forbidden``, and deliberately worded
    like any other missing object: to a caller who cannot see it, an object in
    somebody else's environment must look exactly like an object that is not
    there.
    """
    if auth.ledger.keepsets.holds(str(auth.env_id), [name]):
        return name
    raise NotFound("no such object in this environment", object=name.hex)


def _retain(auth: Authorized, *names: ObjectName) -> None:
    """Add freshly written objects to this environment's keep-set.

    jj writes one object per request and only moves a ref at the end, so between
    the first file and the commit there is content nothing references. Two
    things would go wrong without this: the collector could take it mid-write,
    and ``_visible`` would refuse to hand jj back an object it had just written.

    Recorded on the keep-set rather than against a write lease, which is the
    mechanism for that window, because a lease is a *page-appending*
    structure built for one writer publishing one commit — jj drives it as many
    concurrent single-object writes, where the append races itself. The keep-set
    is an idempotent set-insert, so concurrent writers cannot collide.

    The cost of the choice, stated: an object jj writes but never commits stays
    retained until something rebuilds this environment's keep-set from its refs
    and operation log. That is conservative in the safe direction, and it is what
    ``KeepSetStore.replace`` exists to correct.
    """
    auth.ledger.gc.graduate(str(auth.env_id), list(names))


# ─────────────────────────────────────────────────────────────────────────────
# Files and symlinks — ordinary blobs
# ─────────────────────────────────────────────────────────────────────────────


@router.post("/files")
def write_file(
    org: str,
    env: str,
    auth: JjWrite,
    body: Annotated[bytes, Body(media_type="application/octet-stream")] = b"",
) -> dict[str, str]:
    """Store a file's contents. Chunked and deduplicated like anything else.

    The body defaults to empty because an empty file is ordinary content and
    FastAPI cannot otherwise tell "no body" from "a body of zero bytes" — which
    made committing an empty file through jj a 400.
    """
    del org, env
    name, _ = auth.ledger.ingester.ingest_bytes(body)
    _retain(auth, name)
    return {"id": name.hex}


@router.get("/files/{file_id}")
def read_file(org: str, env: str, file_id: str, auth: JjRead) -> Response:
    del org, env
    reader = BlobReader(auth.ledger.store, _visible(auth, _name(file_id)))
    return Response(content=reader.read_all(), media_type="application/octet-stream")


@router.post("/symlinks")
def write_symlink(org: str, env: str, body: dict[str, str], auth: JjWrite) -> dict[str, str]:
    """A symlink's target is content, stored as an ordinary blob.

    That keeps the model at four object types, and means a link to a very long
    path costs nothing special.
    """
    del org, env
    target = body.get("target")
    if not isinstance(target, str):
        raise InvalidRequest("a symlink needs a 'target' string")
    name, _ = auth.ledger.ingester.ingest_bytes(target.encode())
    _retain(auth, name)
    return {"id": name.hex}


@router.get("/symlinks/{symlink_id}")
def read_symlink(org: str, env: str, symlink_id: str, auth: JjRead) -> dict[str, str]:
    del org, env
    payload = BlobReader(auth.ledger.store, _visible(auth, _name(symlink_id))).read_all()
    return {"target": payload.decode()}


# ─────────────────────────────────────────────────────────────────────────────
# Trees
# ─────────────────────────────────────────────────────────────────────────────


@router.get("/empty-tree")
def empty_tree(org: str, env: str, auth: JjRead) -> dict[str, str]:
    """The id of the empty tree. jj asks for it once and compares against it."""
    del org, env
    from src.fs.edit import empty_tree as build_empty

    name = build_empty(auth.ledger.store, shape=auth.ledger.shape_params)
    _retain(auth, name)
    return {"id": name.hex}


@router.post("/trees")
def write_tree(org: str, env: str, body: dict[str, Any], auth: JjWrite) -> dict[str, str]:
    """Build a Ledger tree from jj's entry list.

    jj's tree entries carry no size, and Ledger's do — a tree entry's size is
    what lets a ranged read descend without fetching children. So each file's
    size is looked up here. That is the one place this mapping costs a fetch per
    entry, and it is the honest price of the two models differing.
    """
    del org, env
    raw = body.get("entries")
    if not isinstance(raw, list):
        raise InvalidRequest("a tree needs an 'entries' array")

    entries = [_tree_entry(auth.ledger, item) for item in raw]
    entries.sort(key=lambda entry: entry.name)
    _reject_duplicates(entries)

    written: list[ObjectName] = []

    def emit(name: ObjectName, framed: bytes) -> None:
        written.append(auth.ledger.store.put_encoded(name, framed).name)

    name = build_tree(entries, emit, auth.ledger.shape_params)
    _retain(auth, *written)
    return {"id": name.hex}


@router.get("/trees/{tree_id}")
def read_tree(org: str, env: str, tree_id: str, auth: JjRead) -> dict[str, Any]:
    del org, env
    store = auth.ledger.store
    name = _visible(auth, _name(tree_id))
    store.get_as(name, Tree)  # so a wrong id is a clear error, not an empty tree

    entries = []
    for entry in iter_entries(store, name):
        match entry.kind:
            case EntryKind.TREE:
                kind, executable = "tree", False
            case EntryKind.SYMLINK:
                kind, executable = "symlink", False
            case _:
                kind, executable = "file", entry.mode == MODE_EXEC
        entries.append(
            {
                "name": entry.name.decode(errors="replace"),
                "kind": kind,
                "id": entry.target.hex,
                "executable": executable,
            }
        )
    return {"entries": entries}


def _tree_entry(ledger: Ledger, item: Any) -> TreeEntry:
    if not isinstance(item, dict):
        raise InvalidRequest("each tree entry must be an object")
    name = str(item.get("name", ""))
    kind = str(item.get("kind", ""))
    target = _name(str(item.get("id", "")))
    if not name or "/" in name:
        raise InvalidRequest("a tree entry name must be one path component", name=name[:80])

    match kind:
        case "tree":
            return TreeEntry(name.encode(), EntryKind.TREE, target, 0, 0)
        case "symlink":
            size = BlobReader(ledger.store, target).size
            return TreeEntry(name.encode(), EntryKind.SYMLINK, target, 0, size)
        case "file":
            mode = MODE_EXEC if item.get("executable") else MODE_REGULAR
            size = BlobReader(ledger.store, target).size
            return TreeEntry(name.encode(), EntryKind.BLOB, target, mode, size)
        case "submodule":
            # jj models a git submodule as a commit id inside a tree. Ledger
            # stores content, and a submodule is a pointer to content somewhere
            # else — the same reason foreign container layers are refused.
            raise InvalidRequest(
                "Ledger versions content, and a git submodule is a pointer to "
                "content held elsewhere; import it as real content instead",
                entry=name,
            )
        case _:
            raise InvalidRequest("unknown tree entry kind", kind=kind)


def _reject_duplicates(entries: list[TreeEntry]) -> None:
    for previous, current in itertools.pairwise(entries):
        if previous.name == current.name:
            raise InvalidRequest(
                "a tree cannot hold two entries with the same name",
                name=current.name.decode(errors="replace"),
            )


# ─────────────────────────────────────────────────────────────────────────────
# Commits
# ─────────────────────────────────────────────────────────────────────────────


@router.post("/commits")
def write_commit(org: str, env: str, body: dict[str, Any], auth: JjWrite) -> dict[str, Any]:
    """Store a jj commit as a Ledger commit.

    jj carries more per commit than Ledger's four-field model does — two full
    signatures with emails and timezones, and a predecessor list. Rather than
    widen the frozen object format for one client, the extra fields ride in the
    commit's ``metadata``, which is exactly what it is for. The commit remains an
    ordinary Ledger commit: ``ledger log`` reads it, diff walks it, collection
    protects it.
    """
    del org, env
    roots = body.get("root_tree")
    if not isinstance(roots, list) or not roots:
        raise InvalidRequest("a commit needs a 'root_tree' with at least one term")
    if len(roots) > 1:
        # The stated conflict gap, surfaced rather than papered over.
        raise InvalidRequest(
            "this commit has a conflicted tree, which Ledger does not model: its "
            "merge refuses an overlap rather than storing one. Resolve the "
            "conflict in the working copy and commit the result",
            terms=len(roots),
        )

    author = _signature(body.get("author"), "author")
    committer = _signature(body.get("committer"), "committer")
    metadata: dict[str, str] = {}
    _record_signature(metadata, "author", author)
    _record_signature(metadata, "committer", committer)
    predecessors = [str(p) for p in body.get("predecessors", [])]
    if predecessors:
        metadata[f"{PREFIX}predecessors"] = ",".join(predecessors)

    change = str(body.get("change_id", ""))
    commit = Commit(
        tree=_name(str(roots[0])),
        parents=tuple(_name(str(p)) for p in body.get("parents", [])),
        change_id=ChangeId(change) if change else ChangeId.new(),
        author=_render(author),
        committer=_render(committer),
        timestamp_us=committer["timestamp_millis"] * 1000,
        message=str(body.get("description", "")),
        metadata=tuple(sorted(metadata.items())),
    )
    outcome = auth.ledger.store.put_object(commit)
    _retain(auth, outcome.name)
    return {"id": outcome.name.hex, "commit": _commit_body(commit)}


@router.get("/commits/{commit_id}")
def read_commit(org: str, env: str, commit_id: str, auth: JjRead) -> dict[str, Any]:
    del org, env
    commit = auth.ledger.store.get_as(_visible(auth, _name(commit_id)), Commit)
    return _commit_body(commit)


def _commit_body(commit: Commit) -> dict[str, Any]:
    """Render a Ledger commit in jj's shape.

    A commit written by jj round-trips exactly, because its extra fields were
    kept. A commit written by ``ledger commit`` has none of them, and the missing
    parts are synthesized from what Ledger does record — so **jj can read history
    it did not write**, which is the property that makes adoption incremental
    rather than a flag day.
    """
    metadata = dict(commit.metadata)
    predecessors = [p for p in metadata.get(f"{PREFIX}predecessors", "").split(",") if p]
    return {
        "parents": [p.hex for p in commit.parents],
        "predecessors": predecessors,
        "root_tree": [commit.tree.hex],
        "change_id": str(commit.change_id),
        "description": commit.message,
        "author": _read_signature(metadata, "author", commit.author, commit.timestamp_us),
        "committer": _read_signature(metadata, "committer", commit.committer, commit.timestamp_us),
    }


def _signature(raw: Any, which: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise InvalidRequest(f"a commit needs a '{which}' signature")
    return {
        "name": str(raw.get("name", "")),
        "email": str(raw.get("email", "")),
        "timestamp_millis": int(raw.get("timestamp_millis", 0)),
        "tz_offset": int(raw.get("tz_offset", 0)),
    }


def _record_signature(metadata: dict[str, str], which: str, signature: dict[str, Any]) -> None:
    for field in ("name", "email", "timestamp_millis", "tz_offset"):
        metadata[f"{PREFIX}{which}.{field}"] = str(signature[field])


def _read_signature(
    metadata: dict[str, str], which: str, fallback: str, timestamp_us: int
) -> dict[str, Any]:
    key = f"{PREFIX}{which}."
    if f"{key}name" not in metadata:
        # Written by something other than jj. Synthesize rather than refuse: a
        # backend that could only read its own commits would make history a
        # boundary, and history is the thing being adopted.
        name, _, email = fallback.partition(" <")
        return {
            "name": name or fallback,
            "email": email.rstrip(">"),
            "timestamp_millis": timestamp_us // 1000,
            "tz_offset": 0,
        }
    return {
        "name": metadata[f"{key}name"],
        "email": metadata.get(f"{key}email", ""),
        "timestamp_millis": int(metadata.get(f"{key}timestamp_millis", 0)),
        "tz_offset": int(metadata.get(f"{key}tz_offset", 0)),
    }


def _render(signature: dict[str, Any]) -> str:
    email = signature["email"]
    return f"{signature['name']} <{email}>" if email else str(signature["name"])


def _name(hex_id: str) -> ObjectName:
    """jj addresses objects by bare hex; Ledger renders them with an algorithm
    prefix. The prefix is added here rather than asked of the backend, so jj's
    ``commit_id_length`` stays 32 and every id it validates is exactly 32 bytes.
    """
    try:
        return ObjectName.parse(hex_id if hex_id.startswith("b3:") else f"b3:{hex_id}")
    except ValueError as exc:
        raise NotFound("not a Ledger object id", id=hex_id[:80]) from exc


@router.get("/info")
def backend_info(
    org: str,
    env: str,
    auth: JjRead,
    probe: Annotated[bool, Query(description="Unused; keeps the route shape stable.")] = False,
) -> dict[str, Any]:
    """What the backend needs to know before it can address anything.

    ``commit_id_length`` and ``change_id_length`` are what jj uses to validate
    every id it is handed, so getting them from the server rather than hardcoding
    them is what lets the two move together.
    """
    del org, env, probe
    from src.format.constants import FORMAT_FINGERPRINT
    from src.fs.edit import empty_tree as build_empty

    return {
        "name": "ledger",
        "commit_id_length": 32,
        "change_id_length": 16,
        "empty_tree_id": build_empty(auth.ledger.store, shape=auth.ledger.shape_params).hex,
        "format_fingerprint": FORMAT_FINGERPRINT,
    }
