"""Committing, and reading history.

This is the four-phase write protocol, executed locally rather than
over the wire:

    phase 1   BeginWrite            → a session with a lease; everything
                                      uploaded under it is a GC root
    phase 2   offer content         → only bytes Ledger has never seen move
    phase 3   write objects         → bottom-up: chunks → blobs → trees → commit
    phase 4   UpdateRef             → the single atomic point in the system

Phases 1 through 3 need no idempotency protection at all, because content
addressing makes them idempotent by construction: storing an object twice is a
no-op, since the name *is* the content. **Only phase 4 is a state transition**,
so only phase 4 carries a key — a real simplification rather than an accident.

The ordering of 3 before 4 is what makes a ref move only once everything it
reaches is durable. A crash before phase 4 leaves
unreferenced objects, which collection reclaims, and no visible change to anyone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, final

from src.format.model import Commit
from src.fs.closure import tree_closure
from src.ids import ChangeId
from src.meta.models import OpKind
from src.runtime.ingest import IngestStats

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence
    from pathlib import Path

    from src.ids import EnvId, ObjectName, RefName
    from src.instance import Ledger
    from src.meta.models import Ref

__all__ = ["CommitResult", "CommitService", "HistoryEntry", "TreeBuilder"]

#: Produces the tree to publish, given the ref's current record (``None`` when
#: the ref does not exist yet). Called inside the write session, so everything
#: it writes is covered by the session's lease.
type TreeBuilder = Callable[["Ref | None"], tuple["ObjectName", IngestStats]]


@final
@dataclass(frozen=True, slots=True)
class CommitResult:
    commit: ObjectName
    ref: RefName
    generation: int
    stats: IngestStats
    op_sequence: int
    replayed: bool = False


@final
@dataclass(frozen=True, slots=True)
class HistoryEntry:
    name: ObjectName
    commit: Commit


@final
class CommitService:
    """Turns a directory into a published version."""

    __slots__ = ("_ledger",)

    def __init__(self, ledger: Ledger) -> None:
        self._ledger = ledger

    def commit(
        self,
        env: EnvId,
        ref: RefName,
        source: Path,
        *,
        author: str,
        message: str,
        change_id: ChangeId | None = None,
        idempotency_key: str | None = None,
        metadata: Sequence[tuple[str, str]] = (),
    ) -> CommitResult:
        """Write a directory as the next version of ``ref``.

        The parent is whatever ``ref`` points at *now*, and the update compares
        on the generation read at the same moment — so a writer that raced us
        produces a 409 carrying the current state rather than a silent overwrite.
        """
        return self.publish(
            env,
            ref,
            lambda _: self._ledger.ingester.ingest_directory(source),
            author=author,
            message=message,
            change_id=change_id,
            idempotency_key=idempotency_key,
            metadata=metadata,
        )

    def publish(
        self,
        env: EnvId,
        ref: RefName,
        build: TreeBuilder,
        *,
        author: str,
        message: str,
        change_id: ChangeId | None = None,
        idempotency_key: str | None = None,
        metadata: Sequence[tuple[str, str]] = (),
    ) -> CommitResult:
        """Run ``build`` inside a write session and publish what it produced.

        The four-phase protocol with the *content* left abstract, because how a
        tree came to exist — a directory walk, an imported image, a converted git
        commit — changes nothing about how it is published.

        ``build`` receives the ref's current record so it can start from the
        version it is amending, and runs **inside** the session on purpose: a
        tree assembled before the lease exists is unprotected content during the
        window when it matters most, and a collection cycle freezing there would
        take objects the commit is about to depend on.
        """
        ledger = self._ledger
        current = ledger.repo.try_get_ref(env, ref)

        session = ledger.repo.begin_write(env, principal=author)
        try:
            tree, stats = build(current)

            # What the *caller* asked for: make this ref name a commit of this
            # tree. Whether that becomes a create or a compare-and-swap is our
            # business, and a retry must not be refused merely because the first
            # attempt created the ref and this one would update it.
            request = {"op": "Commit", "ref": str(ref), "tree": str(tree)}

            if idempotency_key is not None:
                recorded = ledger.repo.recorded_outcome(env, idempotency_key, request)
                if recorded is not None:
                    # Detected before writing a commit object, so a retry leaves
                    # no unreferenced garbage behind.
                    published = ledger.repo.get_ref(env, ref)
                    return CommitResult(
                        commit=published.target,
                        ref=ref,
                        generation=published.generation,
                        stats=stats,
                        op_sequence=int(recorded["op_sequence"]),
                        replayed=True,
                    )

            commit_object = Commit(
                tree=tree,
                parents=(current.target,) if current else (),
                change_id=change_id or ChangeId.new(),
                author=author,
                committer=author,
                timestamp_us=ledger.clock.now_us(),
                message=message,
                metadata=tuple(sorted(metadata)),
            )
            outcome = ledger.store.put_object(commit_object)

            # The commit object is part of what this write cost. Reporting only
            # the tree would show "0 new objects" for a commit that genuinely
            # created one, which is exactly the number a reader uses to decide
            # whether deduplication is working.
            stats = stats + IngestStats(
                objects_offered=1,
                objects_created=int(outcome.created),
                bytes_offered=outcome.size,
                bytes_stored=outcome.size if outcome.created else 0,
            )

            # Everything this commit reaches is recorded against the session
            # *before* the ref moves, so a collection cycle that freezes in the
            # window cannot take content the new commit depends on.
            ledger.repo.record_uploaded(
                env, session.session_id, [outcome.name, *tree_closure(self._ledger.store, tree)]
            )

            if current is None:
                update = ledger.repo.create_ref(
                    env,
                    ref,
                    outcome.name,
                    principal=author,
                    change_id=commit_object.change_id,
                    idempotency_key=idempotency_key,
                    idempotency_request=request,
                )
            else:
                update = ledger.repo.update_ref(
                    env,
                    ref,
                    expected_generation=current.generation,
                    target=outcome.name,
                    principal=author,
                    change_id=commit_object.change_id,
                    idempotency_key=idempotency_key,
                    idempotency_request=request,
                )
            # The ref has moved, so this commit's objects are live. Graduate
            # them into the keep-set *before* releasing the lease, so they are
            # covered by one guard right up to the moment the other takes over.
            # Without that overlap a collection cycle freezing in the window
            # would delete a just-published commit's content.
            ledger.gc.graduate(str(env), [outcome.name, *tree_closure(self._ledger.store, tree)])
        finally:
            # The session ends either way. On success its objects are already
            # in the keep-set; on failure they are unreferenced content on a
            # timer, and no rollback is needed.
            ledger.repo.end_session(env, session.session_id)

        return CommitResult(
            commit=update.ref.target,
            ref=ref,
            generation=update.ref.generation,
            stats=stats,
            op_sequence=update.op_sequence,
            replayed=update.replayed,
        )

    def resolve(self, env: EnvId, ref: RefName) -> ObjectName:
        """The only step that touches mutable state.

        Everything after it is addressed by content and therefore cacheable
        forever — which is why a scheduler resolves once and hands the hash to a
        thousand workers rather than resolving a thousand times.
        """
        return self._ledger.repo.get_ref(env, ref).target

    def log(self, env: EnvId, ref: RefName, *, limit: int = 50) -> list[HistoryEntry]:
        """Walk parent links from a ref.

        Commits are full snapshots over shared content, so this is a plain read
        rather than a delta replay — and reading an old version costs the same
        as reading the newest one.
        """
        return list(self.walk(self.resolve(env, ref), limit=limit))

    def walk(self, start: ObjectName, *, limit: int = 50) -> Iterator[HistoryEntry]:
        """First-parent history from a commit.

        First-parent rather than a full DAG traversal: on a merge the first
        parent is the branch that was being advanced, so this is the history a
        human means by "what happened on main".
        """
        name: ObjectName | None = start
        seen: set[ObjectName] = set()
        for _ in range(limit):
            if name is None or name in seen:
                return
            seen.add(name)
            commit = self._ledger.store.get_as(name, Commit)
            yield HistoryEntry(name=name, commit=commit)
            name = commit.parents[0] if commit.parents else None

    def revert(
        self,
        env: EnvId,
        ref: RefName,
        to: ObjectName,
        *,
        author: str,
        idempotency_key: str | None = None,
    ) -> CommitResult:
        """Go back to an earlier version.

        Because every commit is a full snapshot over shared content,
        this is **one row update and no content moves at all**. The old version's
        bytes never went anywhere.
        """
        current = self._ledger.repo.get_ref(env, ref)
        # Verify the target really is a commit before pointing a ref at it: a
        # ref that named a tree would resolve fine and fail at materialization.
        self._ledger.store.get_as(to, Commit)

        update = self._ledger.repo.update_ref(
            env,
            ref,
            expected_generation=current.generation,
            target=to,
            principal=author,
            idempotency_key=idempotency_key,
            op_kind=OpKind.UPDATE_REF,
        )
        return CommitResult(
            commit=to,
            ref=ref,
            generation=update.ref.generation,
            stats=IngestStats(),
            op_sequence=update.op_sequence,
        )
