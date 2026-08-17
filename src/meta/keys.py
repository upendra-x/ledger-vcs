"""The key grammar for the metadata store.

Everything mutable about one environment is co-located in one
partition::

    PK = env_01J8XQ7M…                    one partition per environment
      ├── SK = env                        the environment record
      ├── SK = ref#refs/heads/main        target, generation, lifecycle
      ├── SK = chg#k7m4…                  change_id → its commits, newest first
      ├── SK = note#c9f2…#qa              QA verdict for that commit
      ├── SK = op#0000000123              operation log entry
      ├── SK = seq                        the dense operation counter
      ├── SK = ws#7f3a…                   an open write session and its lease
      └── SK = idem#7f3a…                 idempotency record

That layout does two jobs at once. Because everything for an environment is
co-located, a commit that moves a ref *and* appends to the operation log *and*
records the change is one single-partition transaction. Because different
environments are different partitions, work on one can never queue behind work
on another — obtained from the data layout rather than from a lock manager.

Sort keys are built here and nowhere else. Two reasons: a prefix query is only
correct if the prefix cannot collide with a different record type (``note#`` and
``notebook#`` would), and operation sequence numbers must be **zero-padded** so
that lexicographic order equals numeric order — an unpadded ``op#10`` sorts
before ``op#9`` and the log silently reads out of order.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from src.ids import ChangeId, ObjectName, RefName, SessionId

__all__ = [
    "OP_SEQ_DIGITS",
    "GlobalSpace",
    "ItemKind",
    "build_lease_sk",
    "build_queue_prefix",
    "build_queue_sk",
    "change_sk",
    "env_sk",
    "idem_sk",
    "note_prefix",
    "note_sk",
    "op_sk",
    "ref_prefix",
    "ref_sk",
    "seq_sk",
    "session_page_prefix",
    "session_page_sk",
    "session_sk",
]

#: Ten digits addresses ten billion operations in one environment, which is
#: about six orders of magnitude more than the capacity plan projects. Padding is not
#: cosmetic: without it ``op#10`` sorts before ``op#9``.
OP_SEQ_DIGITS: Final = 10


class ItemKind:
    """Record types, used as a projected column so queries can filter cheaply."""

    ENV: Final = "env"
    REF: Final = "ref"
    CHANGE: Final = "chg"
    NOTE: Final = "note"
    OP: Final = "op"
    SEQ: Final = "seq"
    SESSION: Final = "ws"
    SESSION_PAGE: Final = "wsp"
    IDEMPOTENCY: Final = "idem"
    #: Queued builds live in the *environment's* partition, which is what makes
    #: "at most one build per environment in flight" a property of the data
    #: layout rather than of a lock manager.
    BUILD_QUEUE: Final = "bq"
    BUILD_LEASE: Final = "blease"


class GlobalSpace:
    """Keyspaces that are deliberately *not* per-environment.

    Mistaking any of these for per-environment state produces a system that is
    subtly wrong rather than obviously broken:

    * a **name** must be unique across all environments, so it cannot live in
      any one of their partitions;
    * a **grant** selects environments by prefix or label, so it belongs to no
      single one — and duplicating it into each match would make every new
      environment a fan-out write;
    * a **build result** is a pure function of a commit, so two refs, or a fork
      and its parent, must find one result rather than rebuilding per
      environment.
    """

    NAMES: Final = "names"
    GRANTS: Final = "grants"
    BUILDS: Final = "builds"
    #: Where a change-stream consumer records how far it has read. Global
    #: because a consumer reads *across* environments — its progress is not a
    #: fact about any one of them.
    CURSORS: Final = "cursors"


def env_sk() -> str:
    return ItemKind.ENV


def ref_sk(name: RefName) -> str:
    return f"{ItemKind.REF}#{name}"


def ref_prefix() -> str:
    return f"{ItemKind.REF}#"


def change_sk(change: ChangeId) -> str:
    return f"{ItemKind.CHANGE}#{change}"


def note_sk(commit: ObjectName, namespace: str) -> str:
    """Notes are namespaced by producer, so the QA pipeline, the builder and the
    platform sync never contend even while annotating the same commit.
    """
    if "#" in namespace or not namespace:
        raise ValueError(f"invalid note namespace: {namespace!r}")
    return f"{ItemKind.NOTE}#{commit.hex}#{namespace}"


def note_prefix(commit: ObjectName | None = None) -> str:
    if commit is None:
        return f"{ItemKind.NOTE}#"
    return f"{ItemKind.NOTE}#{commit.hex}#"


def op_sk(sequence: int) -> str:
    if sequence < 0:
        raise ValueError(f"operation sequence must be non-negative: {sequence}")
    return f"{ItemKind.OP}#{sequence:0{OP_SEQ_DIGITS}d}"


def seq_sk() -> str:
    return ItemKind.SEQ


def session_sk(session: SessionId) -> str:
    return f"{ItemKind.SESSION}#{session}"


def session_page_sk(session: SessionId, page: int) -> str:
    """Uploaded object names, in pages.

    A first commit of a large environment offers tens of thousands of chunks,
    and a session that recorded them all in one item body would be megabytes —
    past what any partitioned store will accept as a single item, and rewritten
    in full on every batch. Paging keeps each write bounded and append-only.
    """
    if page < 0:
        raise ValueError(f"page must be non-negative: {page}")
    return f"{ItemKind.SESSION_PAGE}#{session}#{page:06d}"


def session_page_prefix(session: SessionId) -> str:
    return f"{ItemKind.SESSION_PAGE}#{session}#"


def build_queue_sk(op_sequence: int) -> str:
    """Queued work, ordered by the operation that triggered it.

    Zero-padded for the same reason operation keys are: the queue is drained in
    sort-key order, and an unpadded ``bq#10`` would sort before ``bq#9`` — which
    is precisely how version 49 ends up syncing after version 50.
    """
    if op_sequence < 0:
        raise ValueError(f"operation sequence must be non-negative: {op_sequence}")
    return f"{ItemKind.BUILD_QUEUE}#{op_sequence:0{OP_SEQ_DIGITS}d}"


def build_queue_prefix() -> str:
    return f"{ItemKind.BUILD_QUEUE}#"


def build_lease_sk() -> str:
    """One key per environment, so holding it *is* the exclusivity."""
    return ItemKind.BUILD_LEASE


def idem_sk(key: str) -> str:
    if not key or "#" in key:
        raise ValueError(f"invalid idempotency key: {key!r}")
    return f"{ItemKind.IDEMPOTENCY}#{key}"
