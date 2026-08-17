"""The pages.

Every one of them is a read through the same services and the same authorization
the API uses. There is no write anywhere in this module, and the only reason it
can be trusted to stay that way is that it is short enough to check.

Authorization is HTTP Basic, with the Ledger token as the password, because that
is the only credential a browser can be asked for without inventing a login page
and a session cookie — both of which would be new state on a plane that
deliberately has none.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query
from fastapi.responses import HTMLResponse

from src.api.deps import Authorized, LedgerDep, authorize_env, interactive_capability
from src.api.ui.render import column, e, layout, link, short, table
from src.auth.model import NamePrefixSelector, Operation
from src.auth.tokens import Capability
from src.errors import LedgerError, NotFound
from src.format.constants import EntryKind
from src.format.model import Commit
from src.fs.blob import BlobReader
from src.fs.tree import list_dir, resolve_path
from src.ids import EnvName, ObjectName
from src.text import human_bytes, when

__all__ = ["router"]

router = APIRouter(tags=["ui"], include_in_schema=False)

#: How much of a file the preview will render. A version may hold a forty-gigabyte
#: dataset, and a browser asking for one is a mistake the server should decline
#: to make on its behalf.
PREVIEW_BYTES = 64 * 1024


def require_browse() -> object:
    """Authorize a page as an ordinary read of the environment in its path.

    The same gate the API and the registry use. A browser is the easiest place
    to probe names by hand, so the surface with the friendliest error pages is
    the last one that should have its own idea of what to say about a name the
    caller may not see.
    """

    def dependency(
        org: str,
        env: str,
        state: LedgerDep,
        capability: Annotated[Capability, Depends(interactive_capability)],
    ) -> Authorized:
        return authorize_env(state, capability, EnvName(f"{org}/{env}"), Operation.READ)

    return Depends(dependency)


BrowseAuth = Annotated[Authorized, require_browse()]


def page(html: str) -> HTMLResponse:
    return HTMLResponse(content=html)


# ─────────────────────────────────────────────────────────────────────────────
# Environments
# ─────────────────────────────────────────────────────────────────────────────


@router.get("/", response_class=HTMLResponse)
def index(
    state: LedgerDep,
    capability: Annotated[Capability, Depends(interactive_capability)],
) -> HTMLResponse:
    """Every environment **this token may read**.

    Not every environment the corpus holds. A listing is the sharpest form of
    the enumeration oracle ``authorize_env`` exists to close: a page that
    resolves each name one at a time and refuses honestly is safe, and one that
    prints the whole index to anybody holding any valid token hands over the
    corpus in a single request. The per-name gate is worth nothing if the index
    beside it is ungated.
    """
    names = _readable_environments(state, capability)
    rows = [(link(name, f"/ui/{name}"),) for name in names]
    body = (
        "<h1>Environments</h1>"
        + table([column("name")], rows, empty="no environments this token can read")
        + f'<p class="dim">{len(names)} environment(s)</p>'
    )
    return page(layout("Environments", body))


#: How many names the index will render. A browser asking for ten million rows
#: is a mistake the server should decline to make on its behalf.
INDEX_LIMIT = 500


def _readable_environments(state: LedgerDep, capability: Capability) -> list[EnvName]:
    """The names this capability permits ``env:read`` on, cheapest route first.

    A token pinned to one environment resolves exactly that one. A token
    carrying name prefixes scans each prefix, so the store filters rather than
    this function — which is also what keeps the page usable when the corpus is
    far larger than ``INDEX_LIMIT``. Anything else (label and id selectors) has
    to be decided against the environment record, so those fall back to a scan.

    Every candidate is still checked with ``permits``: the prefixes narrow *what
    to look at*, and only the scope decides what is shown.
    """
    repo = state.ledger.repo
    prefixes = [
        selector.pattern.split("*", 1)[0]
        for selector in capability.scope.selectors
        if isinstance(selector, NamePrefixSelector)
    ]
    candidates: list[EnvName] = []
    seen: set[str] = set()
    for prefix in prefixes or [""]:
        for name in repo.list_envs(prefix=prefix, limit=INDEX_LIMIT):
            if str(name) not in seen:
                seen.add(str(name))
                candidates.append(name)

    readable: list[EnvName] = []
    for name in candidates:
        try:
            env_id = repo.resolve_env_name(name)
            record = repo.get_env(env_id)
        except LedgerError:  # pragma: no cover - a name claim without a record
            continue
        if capability.scope.permits(Operation.READ, env_id, name, dict(record.labels)):
            readable.append(name)
    return readable[:INDEX_LIMIT]


@router.get("/ui/{org}/{env}", response_class=HTMLResponse)
def environment(org: str, env: str, auth: BrowseAuth) -> HTMLResponse:
    """One environment: its refs, its history, and what its version contains."""
    name = f"{org}/{env}"
    record = auth.ledger.repo.get_env(auth.env_id)
    refs = auth.ledger.repo.list_refs(auth.env_id)

    ref_rows = [
        (
            r.name,
            link(short(r.target), f"/ui/{name}/commits/{r.target}"),
            r.generation,
            r.lifecycle,
        )
        for r in refs
    ]
    body = [
        f"<h1>{e(name)}</h1>",
        f'<p class="dim">{e(record.env_id)}'
        + (f" · forked from {e(record.forked_from_env)}" if record.forked_from_env else "")
        + "</p>",
        "<h2>Refs</h2>",
        table(
            [column("ref"), column("commit"), column("generation", numeric=True), column("")],
            ref_rows,
            empty="this environment has no refs",
        ),
        "<h2>History</h2>",
        _history_table(auth, name),
        "<h2>Images</h2>",
        _images_table(auth, name),
        "<h2>Operations</h2>",
        _operations_table(auth),
    ]
    return page(layout(name, "".join(body), crumbs=[("environments", "/"), (name, "")]))


def _history_table(auth: Authorized, name: str) -> str:
    from src.service.commits import CommitService

    try:
        default = auth.ledger.repo.get_env(auth.env_id).default_ref
        entries = CommitService(auth.ledger).log(auth.env_id, default, limit=25)
    except LedgerError:
        return '<p class="empty">no history yet</p>'

    rows = [
        (
            link(short(entry.name), f"/ui/{name}/commits/{entry.name}"),
            entry.commit.message.splitlines()[0] if entry.commit.message else "",
            entry.commit.author,
            when(entry.commit.timestamp_us),
        )
        for entry in entries
    ]
    return table(
        [column("commit"), column("message"), column("author"), column("when")],
        rows,
        empty="no history yet",
    )


def _images_table(auth: Authorized, name: str) -> str:
    """What this version's images are, and how to pull each one.

    Through ``ImageService`` rather than reading the layout here, so the browser
    and the registry agree about what a version contains. A page that resolved
    images its own way would eventually show one the registry would not serve.
    """
    from src.oci.model import REF_NAME_ANNOTATION
    from src.service.images import ImageService

    try:
        default = auth.ledger.repo.get_env(auth.env_id).default_ref
        index = ImageService(auth.ledger).list_images(auth.env_id, default)
    except LedgerError:
        return '<p class="empty">no version yet</p>'
    rows = [
        (
            descriptor.annotation_map.get(REF_NAME_ANNOTATION, "?"),
            str(descriptor.digest),
            str(descriptor.platform) if descriptor.platform else "-",
            f"docker pull <host>/{name}/"
            f"{descriptor.annotation_map.get(REF_NAME_ANNOTATION, '?')}:main",
        )
        for descriptor in index.manifests
    ]
    return table(
        [column("image"), column("manifest"), column("platform"), column("pull with")],
        rows,
        empty="this version holds no container images",
    )


def _operations_table(auth: Authorized) -> str:
    entries = auth.ledger.repo.list_ops(auth.env_id, limit=15)
    rows = [
        (entry.sequence, entry.kind, entry.ref or "", entry.principal, when(entry.at_us))
        for entry in entries
    ]
    return table(
        [
            column("#", numeric=True),
            column("operation"),
            column("ref"),
            column("principal"),
            column("when"),
        ],
        rows,
        empty="nothing has happened here yet",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Commits
# ─────────────────────────────────────────────────────────────────────────────


@router.get("/ui/{org}/{env}/commits/{commit}", response_class=HTMLResponse)
def commit_page(org: str, env: str, commit: str, auth: BrowseAuth) -> HTMLResponse:
    return _tree_page(org, env, commit, "", auth)


@router.get("/ui/{org}/{env}/commits/{commit}/tree/{path:path}", response_class=HTMLResponse)
def tree_page(org: str, env: str, commit: str, path: str, auth: BrowseAuth) -> HTMLResponse:
    return _tree_page(org, env, commit, path, auth)


def _tree_page(org: str, env: str, commit: str, path: str, auth: Authorized) -> HTMLResponse:
    name = f"{org}/{env}"
    commit_name = ObjectName.parse(commit)
    auth.state.policy.authorize_commit(auth.capability, auth.env_id, commit_name)
    record = auth.ledger.store.get_as(commit_name, Commit)

    tree = record.tree
    if path:
        resolved = resolve_path(auth.ledger.store, tree, path)
        if resolved.kind is not EntryKind.TREE:
            raise NotFound("that path is a file", path=path)
        tree = resolved.target

    base = f"/ui/{name}/commits/{commit}"
    rows = []
    for entry in list_dir(auth.ledger.store, tree, limit=500).entries:
        child = entry.name.decode(errors="replace")
        full = f"{path}/{child}" if path else child
        is_tree = entry.kind is EntryKind.TREE
        target = f"{base}/tree/{full}" if is_tree else f"{base}/file/{full}"
        rows.append(
            (
                link(child + ("/" if is_tree else ""), target),
                entry.kind.name.lower(),
                "" if is_tree else human_bytes(entry.size),
                short(entry.target),
            )
        )

    body = [
        f"<h1>{e(path or '/')}</h1>",
        f'<p class="hash">{e(commit)}</p>',
        _commit_summary(record),
        table(
            [column("name"), column("kind"), column("size", numeric=True), column("object")],
            rows,
            empty="this directory is empty",
        ),
        _diff_section(auth, name, commit_name, record),
    ]
    return page(
        layout(
            f"{name} · {path or '/'}",
            "".join(body),
            crumbs=_crumbs(name, commit, path),
        )
    )


def _commit_summary(record: Commit) -> str:
    parents = ", ".join(short(p) for p in record.parents) or "none (root commit)"
    return (
        f"<p>{e(record.message.splitlines()[0] if record.message else '')}</p>"
        f'<p class="dim">{e(record.author)} · {e(when(record.timestamp_us))} · '
        f"change {e(short(record.change_id))} · parents {e(parents)}</p>"
    )


def _diff_section(auth: Authorized, name: str, commit: ObjectName, record: Commit) -> str:
    """What this version changed, against its first parent.

    Cheap for the same reason ``Diff`` is cheap everywhere else: an unchanged
    subtree has an unchanged name, so a whole branch is dismissed by comparing
    two hashes.
    """
    del commit, name
    if not record.parents:
        return ""
    from src.fs.diff import diff_trees

    parent_tree = auth.ledger.store.get_as(record.parents[0], Commit).tree
    changes: list[tuple[str, str, int]] = []
    for change in diff_trees(auth.ledger.store, parent_tree, record.tree):
        if len(changes) == 200:
            break
        changes.append((change.display_path, str(change.kind), change.size_delta))

    return "<h2>Changed in this version</h2>" + table(
        [column("path"), column("change"), column("bytes", numeric=True)],
        changes,
        empty="nothing changed against the first parent",
    )


def _crumbs(name: str, commit: str, path: str) -> list[tuple[str, str]]:
    base = f"/ui/{name}/commits/{commit}"
    crumbs = [("environments", "/"), (name, f"/ui/{name}"), (short(commit), base)]
    walked = ""
    for part in [p for p in path.split("/") if p]:
        walked = f"{walked}/{part}" if walked else part
        crumbs.append((part, f"{base}/tree/{walked}"))
    return crumbs


# ─────────────────────────────────────────────────────────────────────────────
# Files
# ─────────────────────────────────────────────────────────────────────────────


@router.get("/ui/{org}/{env}/commits/{commit}/file/{path:path}", response_class=HTMLResponse)
def file_page(
    org: str,
    env: str,
    commit: str,
    path: str,
    auth: BrowseAuth,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> HTMLResponse:
    """Preview one file, bounded.

    A version may hold a forty-gigabyte dataset. Rendering a *range* rather than
    a file is the same descent the read path uses, and it is why a browser can look
    inside one at all.
    """
    name = f"{org}/{env}"
    commit_name = ObjectName.parse(commit)
    auth.state.policy.authorize_commit(auth.capability, auth.env_id, commit_name)
    tree = auth.ledger.store.get_as(commit_name, Commit).tree

    resolved = resolve_path(auth.ledger.store, tree, path)
    if resolved.kind is EntryKind.TREE:
        raise NotFound("that path is a directory", path=path)

    reader = BlobReader(auth.ledger.store, resolved.target)
    payload = reader.read(offset, PREVIEW_BYTES)
    total = resolved.entry.size

    try:
        preview = payload.decode()
        rendered = f"<pre>{e(preview)}</pre>"
    except UnicodeDecodeError:
        rendered = (
            f'<p class="empty">binary — {e(human_bytes(total))}, '
            f"showing nothing rather than mojibake</p>"
        )

    more = ""
    if offset + len(payload) < total:
        following = offset + PREVIEW_BYTES
        more = (
            f"<p>{link('next ' + human_bytes(PREVIEW_BYTES) + ' →', f'?offset={following}')}"
            f'<span class="dim"> — bytes {offset:,}–{offset + len(payload):,} '
            f"of {total:,}</span></p>"
        )

    body = [
        f"<h1>{e(path)}</h1>",
        f'<p class="dim">{e(human_bytes(total))} · <span class="hash">'
        f"{e(resolved.target)}</span></p>",
        rendered,
        more,
    ]
    return page(layout(f"{name} · {path}", "".join(body), crumbs=_crumbs(name, commit, path)))
