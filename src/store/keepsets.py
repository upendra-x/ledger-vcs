"""Keep-sets: what each environment still needs.

Garbage collection is a set difference — *stored minus live* — and this is the
live side. The catalog (``store.catalog``) is the stored side.

**Maintained, not recomputed**. A commit's closure is fixed
forever, so *addition* is free: when a ref update succeeds, the objects the
write session just recorded join that environment's keep-set, and the write path
already knows what they are. Only *subtraction* costs anything, and only on ref
deletion, where the set is rebuilt from that environment's remaining roots.

**Objects that deduplicated away are recorded too.** A hash the client offered
and was told it already had is still content this commit will reach, and the
collector has to know that before the ref moves — otherwise a sweep between the
offer and the commit takes content the new commit depends on.

The representation is deliberately exact: full 32-byte digests, no Bloom filter
and no probabilistic retention anywhere. Partitioning by hash prefix keeps each
shard's live set small enough to compare in memory, which is what lets the diff
be exact rather than merely safe.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, Protocol, final, runtime_checkable

from src.ids import ObjectName
from src.sqlite_support import ThreadLocalConnections
from src.store.sharding import shard_bounds

if TYPE_CHECKING:
    import sqlite3
    from collections.abc import Iterable, Iterator, Sequence
    from pathlib import Path

__all__ = ["KeepSetStore", "SqliteKeepSetStore"]

_SCHEMA: Final = """
CREATE TABLE IF NOT EXISTS keepset (
    env    TEXT NOT NULL,
    digest BLOB NOT NULL,
    PRIMARY KEY (env, digest)
) WITHOUT ROWID;

-- The collector reads by digest, not by environment: it asks "is this object
-- live anywhere", which under global deduplication is the only question that
-- matters. A chunk may be reached by thousands of unrelated environments.
CREATE INDEX IF NOT EXISTS keepset_by_digest ON keepset (digest);
"""


@runtime_checkable
class KeepSetStore(Protocol):
    def merge(self, env: str, names: Iterable[ObjectName]) -> int:
        """Add objects to an environment's keep-set. Idempotent."""
        ...

    def replace(self, env: str, names: Iterable[ObjectName]) -> int:
        """Rebuild an environment's keep-set from scratch. Used after a deletion."""
        ...

    def drop(self, env: str) -> int: ...

    def live(self, names: Sequence[ObjectName]) -> set[ObjectName]:
        """Which of ``names`` any environment still needs."""
        ...

    def holds(self, env: str, names: Sequence[ObjectName]) -> set[ObjectName]:
        """Which of ``names`` **this** environment reaches.

        The collector never asks this — it only cares whether an object is live
        somewhere. Authorization does: under global deduplication an object name
        is a corpus-wide address, so "may this caller read this hash" cannot be
        answered from the hash alone. This is the per-environment half of the
        same index, and it is what stops a token for one environment reading
        another's content by knowing a name.
        """
        ...

    def iter_shard(self, prefix_bits: int, shard: int) -> Iterator[ObjectName]: ...

    def size(self, env: str | None = None) -> int: ...


@final
class SqliteKeepSetStore:
    __slots__ = ("_connections",)

    def __init__(self, path: Path | str) -> None:
        self._connections = ThreadLocalConnections(path, schema=_SCHEMA)

    @classmethod
    def open(cls, path: Path | str) -> SqliteKeepSetStore:
        return cls(path)

    @property
    def _connection(self) -> sqlite3.Connection:
        return self._connections.get()

    def merge(self, env: str, names: Iterable[ObjectName]) -> int:
        rows = [(env, name.digest) for name in names]
        if not rows:
            return 0
        cursor = self._connection.executemany(
            "INSERT OR IGNORE INTO keepset (env, digest) VALUES (?, ?)", rows
        )
        return cursor.rowcount

    def replace(self, env: str, names: Iterable[ObjectName]) -> int:
        connection = self._connection
        connection.execute("BEGIN IMMEDIATE")
        try:
            connection.execute("DELETE FROM keepset WHERE env = ?", (env,))
            rows = [(env, name.digest) for name in names]
            if rows:
                connection.executemany(
                    "INSERT OR IGNORE INTO keepset (env, digest) VALUES (?, ?)", rows
                )
        except BaseException:
            connection.execute("ROLLBACK")
            raise
        connection.execute("COMMIT")
        return len(rows)

    def drop(self, env: str) -> int:
        cursor = self._connection.execute("DELETE FROM keepset WHERE env = ?", (env,))
        return cursor.rowcount

    def live(self, names: Sequence[ObjectName]) -> set[ObjectName]:
        if not names:
            return set()
        by_digest = {name.digest: name for name in names}
        placeholders = ",".join("?" * len(by_digest))
        rows = self._connection.execute(
            f"SELECT DISTINCT digest FROM keepset WHERE digest IN ({placeholders})",
            list(by_digest),
        ).fetchall()
        return {by_digest[row[0]] for row in rows}

    def holds(self, env: str, names: Sequence[ObjectName]) -> set[ObjectName]:
        if not names:
            return set()
        by_digest = {name.digest: name for name in names}
        placeholders = ",".join("?" * len(by_digest))
        rows = self._connection.execute(
            f"SELECT digest FROM keepset WHERE env = ? AND digest IN ({placeholders})",
            [env, *by_digest],
        ).fetchall()
        return {by_digest[row[0]] for row in rows}

    def iter_shard(self, prefix_bits: int, shard: int) -> Iterator[ObjectName]:
        low, high = shard_bounds(prefix_bits, shard)
        rows = self._connection.execute(
            "SELECT DISTINCT digest FROM keepset WHERE digest >= ? AND digest < ? ORDER BY digest",
            (low, high),
        )
        for (digest,) in rows:
            yield ObjectName(digest)

    def size(self, env: str | None = None) -> int:
        if env is None:
            row = self._connection.execute("SELECT COUNT(DISTINCT digest) FROM keepset").fetchone()
        else:
            row = self._connection.execute(
                "SELECT COUNT(*) FROM keepset WHERE env = ?", (env,)
            ).fetchone()
        return int(row[0])

    def close(self) -> None:
        self._connections.close()
