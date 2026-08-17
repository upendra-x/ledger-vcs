"""Tombstones: the record that an object was swept, so dedup cannot resurrect it.

This is the subtlest guard in the whole design, and it exists because
collection does not delete an object and its parent at the same instant:

    1. tree T becomes unreachable; its exclusive chunk C is standalone
    2. sweep deletes C                    ── C was genuinely garbage
    3. T is small, so it lives in a pack, and packs are dropped only when
       they die or are repacked          ── T survives, dead but present
    4. a new commit contains an identical T
       HasObjects(T) → "present"         ── the client uploads nothing
    5. PutObject(commit) → direct child T exists ✓ → UpdateRef succeeds
    6. the new commit reaches C, which no longer exists

No other guard catches this. The epoch cutoff, the grace period and write leases
all protect *recently written* content, and C was legitimately garbage when it
was swept. The failure is **resurrection** — a dead parent handed back by
deduplication after its children are gone — and its shape is the worst kind:
``Resolve``, ``ListDir``, ``Diff`` and ``Log`` all succeed, and the rollout fails
half an hour in.

The fix is to make deletion visible to deduplication. Every swept hash is
recorded here, and the existence predicate answers *missing* for it regardless
of whether the bytes physically remain — so the client re-uploads and the object
is rewritten with its children.

The price is re-uploading content that was briefly still present. The
alternative is a commit that cannot be read.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, Protocol, final, runtime_checkable

from src.sqlite_support import ThreadLocalConnections

if TYPE_CHECKING:
    import sqlite3
    from collections.abc import Iterable, Sequence
    from pathlib import Path

    from src.ids import ObjectName

__all__ = ["NullTombstoneStore", "SqliteTombstoneStore", "TombstoneStore"]

_SCHEMA: Final = """
CREATE TABLE IF NOT EXISTS tombstone (
    digest        BLOB PRIMARY KEY,
    expires_at_us INTEGER NOT NULL
) WITHOUT ROWID;

CREATE INDEX IF NOT EXISTS tombstone_expiry ON tombstone (expires_at_us);
"""


@runtime_checkable
class TombstoneStore(Protocol):
    """Hashes that were swept and must be treated as absent."""

    def record(self, names: Iterable[ObjectName], *, expires_at_us: int) -> int:
        """Mark hashes as swept. Returns how many were newly recorded."""
        ...

    def filter_tombstoned(self, names: Sequence[ObjectName]) -> set[ObjectName]:
        """Which of ``names`` are tombstoned. Batched — this sits on the write
        path's existence check, which asks about a thousand hashes at a time.
        """
        ...

    def clear(self, names: Iterable[ObjectName]) -> int:
        """Forget tombstones for hashes that have been re-uploaded.

        Without this the store would deadlock: a re-uploaded object would stay
        permanently 'missing', so every subsequent write would upload it again
        and the commit referencing it could never be read back through the
        normal path.
        """
        ...

    def purge_expired(self, now_us: int) -> int:
        """Drop tombstones past their expiry.

        They live one full collection period past the sweep that created them —
        longer than any in-flight write session — after which no writer can
        still be holding a stale 'present' answer from before the sweep.
        """
        ...

    def count(self) -> int:
        """How many tombstones are outstanding.

        On the interface rather than only the implementation because it is an
        operational signal: tombstones growing without bound means sweeps are
        outrunning expiry, and re-uploads are being forced needlessly.
        """
        ...


@final
class SqliteTombstoneStore:
    """The real implementation. One row per swept object.

    A separate table rather than a column on the write catalog, because the two
    have opposite lifetimes: the catalog is append-only and compacted after each
    cycle, while tombstones are created and expired continuously.
    """

    __slots__ = ("_connections",)

    def __init__(self, path: Path | str) -> None:
        self._connections = ThreadLocalConnections(path, schema=_SCHEMA)

    @classmethod
    def open(cls, path: Path | str) -> SqliteTombstoneStore:
        return cls(path)

    @property
    def _connection(self) -> sqlite3.Connection:
        return self._connections.get()

    def record(self, names: Iterable[ObjectName], *, expires_at_us: int) -> int:
        rows = [(name.digest, expires_at_us) for name in names]
        if not rows:
            return 0
        cursor = self._connection.executemany(
            "INSERT OR IGNORE INTO tombstone (digest, expires_at_us) VALUES (?, ?)", rows
        )
        return cursor.rowcount

    def filter_tombstoned(self, names: Sequence[ObjectName]) -> set[ObjectName]:
        if not names:
            return set()
        by_digest = {name.digest: name for name in names}
        placeholders = ",".join("?" * len(by_digest))
        rows = self._connection.execute(
            f"SELECT digest FROM tombstone WHERE digest IN ({placeholders})",
            list(by_digest),
        ).fetchall()
        return {by_digest[row[0]] for row in rows}

    def clear(self, names: Iterable[ObjectName]) -> int:
        rows = [(name.digest,) for name in names]
        if not rows:
            return 0
        cursor = self._connection.executemany("DELETE FROM tombstone WHERE digest = ?", rows)
        return cursor.rowcount

    def purge_expired(self, now_us: int) -> int:
        cursor = self._connection.execute(
            "DELETE FROM tombstone WHERE expires_at_us <= ?", (now_us,)
        )
        return cursor.rowcount

    def count(self) -> int:
        return int(self._connection.execute("SELECT COUNT(*) FROM tombstone").fetchone()[0])

    def close(self) -> None:
        self._connections.close()


@final
class NullTombstoneStore:
    """A tombstone store that records nothing.

    Exists for exactly one purpose: to let ``test_resurrection_regression`` show
    that the guard is load-bearing by removing it and watching a commit become
    readable-but-broken. Wiring this into a running system reintroduces the bug
    in the module docstring, so ``ObjectStore`` takes **no default** for its
    tombstone store and the application factory asserts this type is not used.
    """

    __slots__ = ()

    def record(self, names: Iterable[ObjectName], *, expires_at_us: int) -> int:
        del names, expires_at_us
        return 0

    def filter_tombstoned(self, names: Sequence[ObjectName]) -> set[ObjectName]:
        del names
        return set()

    def clear(self, names: Iterable[ObjectName]) -> int:
        del names
        return 0

    def purge_expired(self, now_us: int) -> int:
        del now_us
        return 0

    def count(self) -> int:
        return 0
