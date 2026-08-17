"""Converting a git repository into Ledger.

Not a wrapper around git and not a two-way bridge — a **conversion**. Each git
object becomes the Ledger object that means the same thing, and after the import
nothing points back at git.

Three decisions are worth stating up front, because they are what makes this
honest rather than approximate.

**Blobs are re-chunked, not copied.** A git blob is stored whole and
zlib-deflated; Ledger chunks it so that editing part of a large file costs the
part. Copying git's representation across would import the
storage model this system exists to replace, and a 2 GiB dataset would still
cost 2 GiB per revision.

**Every git object is converted once.** A `git_sha → object name` mapping means a
subtree shared by a thousand commits is walked once, and re-running an import
resumes rather than restarts. That mapping is the same alternate-digest index the
container registry uses (``store.digests``) — a git SHA-1 is another ecosystem's
name for content Ledger holds, which is exactly what that index is for.

**What cannot be represented faithfully is refused, loudly.** An LFS pointer is a
128-byte file that stands for content held on a server this importer was not
given; a submodule is a commit id in another repository. Importing either as-is
would produce an environment that looks complete and is not — the single worst
outcome available here, because it fails later, elsewhere, to someone else. By
default the import stops and names the path. ``--skip-unsupported`` continues and
returns the full list, so the decision to accept an incomplete import is one
somebody makes on purpose.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Final, final

from src.errors import InvalidRequest, NotFound
from src.format.constants import MODE_EXEC, MODE_REGULAR, EntryKind
from src.format.model import Commit, TreeEntry
from src.format.shape import build_tree
from src.fs.closure import commit_closure
from src.ids import ChangeId, ObjectName
from src.runtime.ingest import IngestStats
from src.store.digests import DigestEntry

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from src.ids import EnvId, RefName
    from src.instance import Ledger

__all__ = ["GitImporter", "ImportReport", "Unsupported", "UnsupportedPath"]

#: How a git SHA-1 is recorded in the alternate-digest index. Namespaced so it
#: cannot collide with the SHA-256 an OCI layer is named by.
GIT_SHA1: Final = "git-sha1"

#: What an LFS pointer file starts with. The whole file is about 130 bytes and
#: stands for content that lives somewhere else entirely.
LFS_MAGIC: Final = b"version https://git-lfs.github.com/spec/"

#: git's own empty tree. Every repository has it whether or not it is stored.
GIT_EMPTY_TREE: Final = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"

#: git's file modes, and what each means to Ledger.
MODE_DIRECTORY: Final = "40000"
MODE_SUBMODULE: Final = "160000"
MODE_SYMLINK: Final = "120000"


class Unsupported(StrEnum):
    """Why a path could not be converted faithfully."""

    LFS_POINTER = "lfs_pointer"
    SUBMODULE = "submodule"


@final
@dataclass(frozen=True, slots=True)
class UnsupportedPath:
    path: str
    reason: Unsupported
    detail: str = ""


@final
@dataclass(frozen=True, slots=True)
class ImportReport:
    """What the import converted, and what it cost.

    The deduplication numbers are the point. The first open question
    is whether
    2 GiB of unique content per environment is the right assumption, and an
    import over real history is the only thing in this system that can answer it.
    """

    head: ObjectName | None = None
    commits: int = 0
    trees: int = 0
    blobs: int = 0
    #: git objects that had already been converted — a subtree shared by many
    #: commits, or a re-run of the same import.
    reused: int = 0
    #: Bytes of git blob content seen, before chunking and deduplication.
    git_bytes: int = 0
    stats: IngestStats = field(default_factory=IngestStats)
    unsupported: tuple[UnsupportedPath, ...] = ()

    @property
    def dedup_ratio(self) -> float:
        """How much of the history's content was already held."""
        if self.git_bytes == 0:
            return 0.0
        return 1.0 - (self.stats.bytes_stored / self.git_bytes)

    @property
    def complete(self) -> bool:
        return not self.unsupported


@final
class GitImporter:
    """Reads a git repository and writes it as Ledger commits."""

    __slots__ = ("_ledger", "_skip_unsupported")

    def __init__(self, ledger: Ledger, *, skip_unsupported: bool = False) -> None:
        self._ledger = ledger
        self._skip_unsupported = skip_unsupported

    def import_repository(
        self,
        repository: Path,
        env: EnvId,
        ref: RefName,
        *,
        source_ref: str = "HEAD",
        limit: int | None = None,
        author: str = "ledger-import",
    ) -> ImportReport:
        """Convert a repository's history and publish it at ``ref``.

        Commits are converted **oldest first**, so each one's parents already
        exist when it is written — which is also what makes a partial import
        resumable rather than merely restartable.
        """
        with _GitRepository(repository) as git:
            shas = git.rev_list(source_ref, limit=limit)
            if not shas:
                raise InvalidRequest("that ref names no commits", ref=source_ref)

            report = ImportReport()
            translated: dict[str, ObjectName] = {}
            for sha in shas:
                report = self._convert_commit(git, sha, translated, report)

            head = translated[shas[-1]]

        if report.unsupported and not self._skip_unsupported:  # pragma: no cover - guarded below
            raise AssertionError("unsupported paths should have raised already")

        self._publish(env, ref, head, author=author)
        return replace(report, head=head)

    # ── commits ──────────────────────────────────────────────────────────────

    def _convert_commit(
        self,
        git: _GitRepository,
        sha: str,
        translated: dict[str, ObjectName],
        report: ImportReport,
    ) -> ImportReport:
        header = git.read_commit(sha)
        tree, report = self._convert_tree(git, header.tree, "", report)

        parents = tuple(translated[parent] for parent in header.parents if parent in translated)
        commit = Commit(
            tree=tree,
            parents=parents,
            # A git commit has no change id. One derived from its SHA-1 is stable
            # across re-imports, so importing the same history twice produces the
            # same change identity rather than a fresh one each time.
            change_id=ChangeId(sha[:32].ljust(32, "0")),
            author=header.author,
            committer=header.committer,
            timestamp_us=header.committed_at_us,
            message=header.message,
            metadata=(("git.sha", sha),),
        )
        outcome = self._ledger.store.put_object(commit)
        translated[sha] = outcome.name
        self._remember(sha, outcome.name, len(header.message))

        return replace(
            report,
            commits=report.commits + 1,
            stats=report.stats
            + IngestStats(
                objects_offered=1,
                objects_created=int(outcome.created),
                bytes_offered=outcome.size,
                bytes_stored=outcome.size if outcome.created else 0,
                bytes_on_disk=outcome.stored_size,
            ),
        )

    # ── trees ────────────────────────────────────────────────────────────────

    def _convert_tree(
        self, git: _GitRepository, sha: str, prefix: str, report: ImportReport
    ) -> tuple[ObjectName, ImportReport]:
        known = self._recall(sha)
        if known is not None:
            return known, replace(report, reused=report.reused + 1)

        entries: list[TreeEntry] = []
        for item in git.list_tree(sha):
            path = f"{prefix}{item.name}"
            converted, report = self._convert_entry(git, item, path, report)
            if converted is not None:
                entries.append(converted)

        # git orders tree entries as though directories ended in "/", so its
        # order is not plain byte order. Re-sorting is not cosmetic: the codec
        # requires sorted entries, and a tree built in git's order would either
        # be rejected or — worse — get a different name for the same content.
        entries.sort(key=lambda entry: entry.name)

        stats = _Counter()
        name = build_tree(entries, stats.emitter(self._ledger), self._ledger.shape_params)
        self._remember(sha, name, 0)
        return name, replace(
            report,
            trees=report.trees + 1,
            stats=report.stats + stats.frozen,
        )

    def _convert_entry(
        self, git: _GitRepository, item: _TreeItem, path: str, report: ImportReport
    ) -> tuple[TreeEntry | None, ImportReport]:
        if item.mode == MODE_SUBMODULE:
            return None, self._refuse(
                report, path, Unsupported.SUBMODULE, f"points at commit {item.sha[:12]}"
            )

        if item.mode == MODE_DIRECTORY:
            subtree, report = self._convert_tree(git, item.sha, f"{path}/", report)
            return TreeEntry(item.name.encode(), EntryKind.TREE, subtree, 0, 0), report

        # Asked *before* reading. A file that survives a hundred commits
        # appears in a hundred trees, and reading it from git each time would
        # make the import cost the history's logical size rather than its
        # content's — the exact thing this system exists not to do. The size
        # comes from the index alongside the name, so a known blob costs one
        # lookup and no bytes at all.
        known = self._ledger.digests.lookup(GIT_SHA1, item.sha)
        if known is not None and not self._ledger.store.missing([known.name]):
            entry = _blob_entry(item, known.name, known.size)
            return entry, replace(
                report,
                reused=report.reused + 1,
                git_bytes=report.git_bytes + known.size,
            )

        content = git.read_blob(item.sha)
        if content.startswith(LFS_MAGIC):
            return None, self._refuse(
                report,
                path,
                Unsupported.LFS_POINTER,
                "the file is a pointer; its content lives on an LFS server",
            )

        name, converted = self._ledger.ingester.ingest_bytes(content)
        self._remember(item.sha, name, len(content))

        return _blob_entry(item, name, len(content)), replace(
            report,
            blobs=report.blobs + 1,
            git_bytes=report.git_bytes + len(content),
            stats=report.stats + converted,
        )

    # ── the git-sha index ────────────────────────────────────────────────────

    def _recall(self, sha: str) -> ObjectName | None:
        """What this git object became, if it has been converted before.

        A hint, confirmed against the store's own existence predicate — the same
        rule the container registry follows, and for the same reason: collection
        may have swept the object since the row was written, and a stale row must
        never make an import skip content it then references.
        """
        hit = self._ledger.digests.lookup(GIT_SHA1, sha)
        if hit is None or self._ledger.store.missing([hit.name]):
            return None
        return hit.name

    def _remember(self, sha: str, name: ObjectName, size: int) -> None:
        self._ledger.digests.record(
            [DigestEntry(algorithm=GIT_SHA1, encoded=sha, name=name, size=size)]
        )

    # ── refusals ─────────────────────────────────────────────────────────────

    def _refuse(
        self, report: ImportReport, path: str, reason: Unsupported, detail: str
    ) -> ImportReport:
        unsupported = UnsupportedPath(path=path, reason=reason, detail=detail)
        if not self._skip_unsupported:
            raise InvalidRequest(
                f"cannot import {path} faithfully: {detail}. Re-run with "
                f"--skip-unsupported to import the rest and get the full list",
                path=path,
                reason=str(reason),
            )
        return replace(report, unsupported=(*report.unsupported, unsupported))

    # ── publishing ───────────────────────────────────────────────────────────

    def _publish(self, env: EnvId, ref: RefName, head: ObjectName, *, author: str) -> None:
        """Point the ref at the imported head, through the ordinary write path.

        Not a special case: the objects are already durable, so this is one ref
        update, and it is a compare-and-swap like any other. An import that
        raced a writer gets the same 409 anybody else would.
        """
        current = self._ledger.repo.try_get_ref(env, ref)
        if current is None:
            self._ledger.repo.create_ref(env, ref, head, principal=author)
        else:
            self._ledger.repo.update_ref(
                env,
                ref,
                expected_generation=current.generation,
                target=head,
                principal=author,
            )
        self._ledger.gc.graduate(str(env), commit_closure(self._ledger.store, head))


def _blob_entry(item: _TreeItem, name: ObjectName, size: int) -> TreeEntry:
    """A file or a symlink. git stores a symlink's target as blob content, and so
    does Ledger, so the only difference is the entry's kind.
    """
    if item.mode == MODE_SYMLINK:
        return TreeEntry(item.name.encode(), EntryKind.SYMLINK, name, 0, size)
    mode = MODE_EXEC if item.mode.endswith("755") else MODE_REGULAR
    return TreeEntry(item.name.encode(), EntryKind.BLOB, name, mode, size)


# ─────────────────────────────────────────────────────────────────────────────
# Reading git
# ─────────────────────────────────────────────────────────────────────────────


@final
@dataclass(frozen=True, slots=True)
class _TreeItem:
    mode: str
    kind: str
    sha: str
    name: str


@final
@dataclass(frozen=True, slots=True)
class _CommitHeader:
    tree: str
    parents: tuple[str, ...]
    author: str
    committer: str
    committed_at_us: int
    message: str


@final
class _GitRepository:
    """A repository, read through ``git`` itself.

    Shelling out rather than parsing ``.git`` directly. Packfiles, deltas,
    alternates and the several object formats git has shipped are a great deal of
    surface to reimplement in order to read data that git will hand over on
    request — and every one of them is a place to be subtly wrong about somebody
    else's history.

    One long-lived ``cat-file --batch`` process serves every object read. Spawning
    a process per object turns importing a repository with a hundred thousand
    objects into a hundred thousand fork/exec pairs, which dominates everything
    else the importer does.
    """

    __slots__ = ("_batch", "_root")

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        if not (self._root / ".git").exists() and not (self._root / "HEAD").exists():
            raise NotFound("not a git repository", path=str(self._root))
        self._batch = subprocess.Popen(
            ["git", "-C", str(self._root), "cat-file", "--batch"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

    def __enter__(self) -> _GitRepository:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self._batch.stdin is not None:
            self._batch.stdin.close()
        self._batch.wait(timeout=10)

    # ── plumbing ─────────────────────────────────────────────────────────────

    def _run(self, *arguments: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(self._root), *arguments],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise InvalidRequest(
                "git could not answer that", command=" ".join(arguments), error=result.stderr[:400]
            )
        return result.stdout

    def _read_object(self, sha: str) -> tuple[str, bytes]:
        assert self._batch.stdin is not None
        assert self._batch.stdout is not None
        self._batch.stdin.write(f"{sha}\n".encode())
        self._batch.stdin.flush()

        header = self._batch.stdout.readline().decode().strip()
        if header.endswith("missing"):
            raise NotFound("git has no such object", sha=sha)
        _, kind, size = header.split()
        payload = self._batch.stdout.read(int(size))
        self._batch.stdout.read(1)  # the trailing newline git writes
        return kind, payload

    # ── the three questions the importer asks ────────────────────────────────

    def rev_list(self, ref: str, *, limit: int | None = None) -> list[str]:
        """Commits reachable from ``ref``, **oldest first**."""
        arguments = ["rev-list", "--reverse", "--topo-order"]
        if limit is not None:
            # `--max-count` counts from the newest, so it is applied before the
            # reversal — taking the most recent N and importing them in order.
            arguments = ["rev-list", "--topo-order", f"--max-count={limit}"]
            shas = self._run(*arguments, ref).split()
            return list(reversed(shas))
        return self._run(*arguments, ref).split()

    def read_commit(self, sha: str) -> _CommitHeader:
        kind, payload = self._read_object(sha)
        if kind != "commit":
            raise InvalidRequest("expected a commit", sha=sha, kind=kind)
        return _parse_commit(payload)

    def list_tree(self, sha: str) -> Iterator[_TreeItem]:
        if sha == GIT_EMPTY_TREE:
            return
        for line in self._run("ls-tree", sha).splitlines():
            meta, _, name = line.partition("\t")
            mode, kind, object_sha = meta.split()
            yield _TreeItem(mode=mode.lstrip("0"), kind=kind, sha=object_sha, name=name)

    def read_blob(self, sha: str) -> bytes:
        kind, payload = self._read_object(sha)
        if kind != "blob":
            raise InvalidRequest("expected a blob", sha=sha, kind=kind)
        return payload


def _parse_commit(payload: bytes) -> _CommitHeader:
    """Parse git's commit object format.

    Headers, a blank line, then the message. The only subtlety is that a
    signature header continues onto indented lines, which are skipped rather
    than parsed — a signature covers git's serialization, which does not survive
    the conversion anyway.
    """
    header, _, message = payload.partition(b"\n\n")
    tree = ""
    parents: list[str] = []
    author = committer = ""
    committed_at_us = 0

    for raw in header.decode(errors="replace").splitlines():
        if raw.startswith(" "):
            continue
        key, _, value = raw.partition(" ")
        match key:
            case "tree":
                tree = value
            case "parent":
                parents.append(value)
            case "author":
                author = _identity(value)
            case "committer":
                committer = _identity(value)
                committed_at_us = _timestamp_us(value)

    if not tree:
        raise InvalidRequest("a git commit with no tree")
    return _CommitHeader(
        tree=tree,
        parents=tuple(parents),
        author=author or committer,
        committer=committer or author,
        committed_at_us=committed_at_us,
        message=message.decode(errors="replace"),
    )


def _identity(value: str) -> str:
    """``Name <email> 1700000000 +0100`` → ``Name <email>``."""
    end = value.rfind(">")
    return value[: end + 1] if end != -1 else value


def _timestamp_us(value: str) -> int:
    parts = value.split()
    for part in reversed(parts):
        if part.isdigit():
            return int(part) * 1_000_000
    return 0


@final
class _Counter:
    """Tallies what building a tree wrote. Mirrors the ingester's accumulator."""

    __slots__ = ("frozen",)

    def __init__(self) -> None:
        self.frozen = IngestStats()

    def emitter(self, ledger: Ledger):  # type: ignore[no-untyped-def]
        def emit(name: ObjectName, framed: bytes) -> None:
            outcome = ledger.store.put_encoded(name, framed)
            self.frozen = self.frozen + IngestStats.counting([outcome])

        return emit


def unsupported_summary(paths: Sequence[UnsupportedPath]) -> str:
    counts: dict[str, int] = {}
    for path in paths:
        counts[str(path.reason)] = counts.get(str(path.reason), 0) + 1
    return ", ".join(f"{count} {reason}" for reason, count in sorted(counts.items()))
