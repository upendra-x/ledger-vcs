"""The write catalog: an append-only record of every object ever stored.

Garbage collection is a set difference — *stored minus live* — and this is the
stored side of it. It exists because the alternative is
enumerating the object store itself, which at 21 billion keys is not a listing
anyone wants to do daily.

Read sequentially and never randomly, so it stays cheap: the collector streams
it in hash order per shard, diffs against that shard's live set, and compacts
afterwards as sweeps append deletion records.

It also answers a question nothing else can: *how much would this reclaim?*
Sizes live here rather than being re-derived by stat-ing every candidate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Protocol, final, runtime_checkable

from src.ids import ObjectName
from src.sqlite_support import ThreadLocalConnections
from src.store.sharding import shard_bounds, shard_of

if TYPE_CHECKING:
    import sqlite3
    from collections.abc import Iterable, Iterator
    from pathlib import Path

__all__ = ["CatalogEntry", "InMemoryWriteCatalog", "SqliteWriteCatalog", "WriteCatalog"]

_SCHEMA: Final = """
CREATE TABLE IF NOT EXISTS write_catalog (
    digest        BLOB PRIMARY KEY,
    size          INTEGER NOT NULL,
    kind          INTEGER NOT NULL,
    written_at_us INTEGER NOT NULL
) WITHOUT ROWID;
"""


@final
@dataclass(frozen=True, slots=True)
class CatalogEntry:
    name: ObjectName
    size: int
    #: Kind tag, kept so the collector can report what it is about to delete
    #: without fetching and decoding every candidate.
    kind: int
    written_at_us: int


@runtime_checkable
class WriteCatalog(Protocol):
    def record(self, entries: Iterable[CatalogEntry]) -> int:
        """Append entries. Idempotent by hash — re-storing an object is a no-op,
        so its catalog row must not be duplicated either.
        """
        ...

    def forget(self, names: Iterable[ObjectName]) -> int:
        """Remove entries for swept objects, so the next cycle's diff is smaller."""
        ...

    def iter_shard(self, prefix_bits: int, shard: int) -> Iterator[CatalogEntry]:
        """Stream one shard of the catalog in hash order.

        Sharding by hash prefix is what makes the diff exact rather than
        probabilistic: each shard's live set is small enough to sort in memory,
        so no Bloom filter and no approximate retention is needed anywhere.
        """
        ...

    def total(self) -> tuple[int, int]:
        """(object count, total bytes). Feeds the circuit breaker, which aborts a
        cycle proposing to delete more than a configured fraction of the corpus.
        """
        ...


@final
class SqliteWriteCatalog:
    __slots__ = ("_connections",)

    def __init__(self, path: Path | str) -> None:
        self._connections = ThreadLocalConnections(path, schema=_SCHEMA)

    @classmethod
    def open(cls, path: Path | str) -> SqliteWriteCatalog:
        return cls(path)

    @property
    def _connection(self) -> sqlite3.Connection:
        return self._connections.get()

    def record(self, entries: Iterable[CatalogEntry]) -> int:
        rows = [(e.name.digest, e.size, e.kind, e.written_at_us) for e in entries]
        if not rows:
            return 0
        cursor = self._connection.executemany(
            "INSERT OR IGNORE INTO write_catalog (digest, size, kind, written_at_us) "
            "VALUES (?, ?, ?, ?)",
            rows,
        )
        return cursor.rowcount

    def forget(self, names: Iterable[ObjectName]) -> int:
        rows = [(name.digest,) for name in names]
        if not rows:
            return 0
        cursor = self._connection.executemany("DELETE FROM write_catalog WHERE digest = ?", rows)
        return cursor.rowcount

    def iter_shard(self, prefix_bits: int, shard: int) -> Iterator[CatalogEntry]:
        low, high = shard_bounds(prefix_bits, shard)
        rows = self._connection.execute(
            "SELECT digest, size, kind, written_at_us FROM write_catalog "
            "WHERE digest >= ? AND digest < ? ORDER BY digest",
            (low, high),
        )
        for digest, size, kind, written_at_us in rows:
            yield CatalogEntry(ObjectName(digest), size, kind, written_at_us)

    def total(self) -> tuple[int, int]:
        count, size = self._connection.execute(
            "SELECT COUNT(*), COALESCE(SUM(size), 0) FROM write_catalog"
        ).fetchone()
        return int(count), int(size)

    def close(self) -> None:
        self._connections.close()


@final
class InMemoryWriteCatalog:
    """For tests. Same semantics, no file."""

    __slots__ = ("_entries",)

    def __init__(self) -> None:
        self._entries: dict[ObjectName, CatalogEntry] = {}

    def record(self, entries: Iterable[CatalogEntry]) -> int:
        added = 0
        for entry in entries:
            if entry.name not in self._entries:
                self._entries[entry.name] = entry
                added += 1
        return added

    def forget(self, names: Iterable[ObjectName]) -> int:
        return sum(self._entries.pop(name, None) is not None for name in names)

    def iter_shard(self, prefix_bits: int, shard: int) -> Iterator[CatalogEntry]:
        for name in sorted(self._entries):
            if shard_of(name, prefix_bits) == shard:
                yield self._entries[name]

    def total(self) -> tuple[int, int]:
        return len(self._entries), sum(e.size for e in self._entries.values())

    def __len__(self) -> int:
        return len(self._entries)
