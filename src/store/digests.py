"""The alternate-digest index: *other* content addresses for content we hold.

Ledger names objects by BLAKE3. Some ecosystems that hand us content name it by
something else — an OCI layer is identified by its SHA-256 — and the format
asks
for a small index recording the correspondence, so a pull by SHA-256 does not
require a scan.

Nothing about this module knows what OCI is. It maps *(algorithm, hex) → object
name*, which is a fact about bytes, and the layer above decides what the
algorithm means.

**It is a hint, never a proof of existence.** The one predicate that decides
whether an object is present is ``ObjectStore.missing``, and a row here can
outlive the object it names — collection sweeps content, and a stale row must
not resurrect it. Every caller therefore
confirms with the store before using a hit, and the collector forgets rows for
what it swept so the staleness stays rare rather than permanent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Protocol, final, runtime_checkable

from src.ids import ObjectName
from src.sqlite_support import ThreadLocalConnections

if TYPE_CHECKING:
    import sqlite3
    from collections.abc import Iterable
    from pathlib import Path

__all__ = ["DigestEntry", "DigestIndex", "SqliteDigestIndex"]

_SCHEMA: Final = """
CREATE TABLE IF NOT EXISTS alternate_digest (
    algorithm   TEXT NOT NULL,
    encoded     BLOB NOT NULL,
    object_name BLOB NOT NULL,
    size        INTEGER NOT NULL,
    PRIMARY KEY (algorithm, encoded)
) WITHOUT ROWID;

-- The collector deletes by object name, which is the reverse of how the index
-- is read. Without this it would be a full scan per sweep.
CREATE INDEX IF NOT EXISTS alternate_digest_by_object ON alternate_digest (object_name);
"""


@final
@dataclass(frozen=True, slots=True)
class DigestEntry:
    """One correspondence: content named ``algorithm:encoded`` is object ``name``.

    ``size`` is the length of the *content*, not of the stored object — a caller
    that found a hit needs to describe the content to whoever asked, and re-deriving
    the length would mean reading every chunk.
    """

    algorithm: str
    encoded: str
    name: ObjectName
    size: int


@runtime_checkable
class DigestIndex(Protocol):
    def record(self, entries: Iterable[DigestEntry]) -> int:
        """Add correspondences. Idempotent — the mapping is a fact about content."""
        ...

    def lookup(self, algorithm: str, encoded: str) -> DigestEntry | None:
        """The object holding that content, *if we ever stored it*.

        A hit is a hint. Confirm with ``ObjectStore.missing`` before relying on
        it: an object can be swept between the write that recorded this row and
        the read that finds it.
        """
        ...

    def forget(self, names: Iterable[ObjectName]) -> int:
        """Drop rows naming these objects. Called after a sweep."""
        ...

    def size(self) -> int: ...


@final
class SqliteDigestIndex:
    __slots__ = ("_connections",)

    def __init__(self, path: Path | str) -> None:
        self._connections = ThreadLocalConnections(path, schema=_SCHEMA)

    @classmethod
    def open(cls, path: Path | str) -> SqliteDigestIndex:
        return cls(path)

    @property
    def _connection(self) -> sqlite3.Connection:
        return self._connections.get()

    def record(self, entries: Iterable[DigestEntry]) -> int:
        rows = [
            (entry.algorithm, bytes.fromhex(entry.encoded), entry.name.digest, entry.size)
            for entry in entries
        ]
        if not rows:
            return 0
        cursor = self._connection.executemany(
            "INSERT OR REPLACE INTO alternate_digest "
            "(algorithm, encoded, object_name, size) VALUES (?, ?, ?, ?)",
            rows,
        )
        return cursor.rowcount

    def lookup(self, algorithm: str, encoded: str) -> DigestEntry | None:
        try:
            key = bytes.fromhex(encoded)
        except ValueError:
            return None
        row = self._connection.execute(
            "SELECT object_name, size FROM alternate_digest WHERE algorithm = ? AND encoded = ?",
            (algorithm, key),
        ).fetchone()
        if row is None:
            return None
        return DigestEntry(
            algorithm=algorithm, encoded=encoded, name=ObjectName(row[0]), size=int(row[1])
        )

    def forget(self, names: Iterable[ObjectName]) -> int:
        rows = [(name.digest,) for name in names]
        if not rows:
            return 0
        cursor = self._connection.executemany(
            "DELETE FROM alternate_digest WHERE object_name = ?", rows
        )
        return cursor.rowcount

    def size(self) -> int:
        row = self._connection.execute("SELECT COUNT(*) FROM alternate_digest").fetchone()
        return int(row[0])

    def close(self) -> None:
        self._connections.close()
