"""One SQLite connection per thread, per database.

A ``sqlite3.Connection`` is not safe for concurrent use, even with
``check_same_thread=False`` — that flag disables the *check*, not the hazard. Two
threads touching one handle produce ``InterfaceError: bad parameter or other API
misuse`` at unpredictable moments, which is exactly the kind of failure that
shows up first under the concurrency the system is built for.

Sharing one connection and guarding it with a lock is worse than it looks: it
serialises every reader behind the current writer, across every partition, which
throws away the per-partition parallelism the write path depends on. WAL mode plus
per-thread connections lets readers proceed while a writer holds its database,
and ``BEGIN IMMEDIATE`` plus ``busy_timeout`` provides the mutual exclusion that
actually matters.

This is the one place that knowledge lives, so the object store's catalog, its
tombstones and the metadata shards cannot each get it subtly differently.
"""

from __future__ import annotations

import secrets
import sqlite3
import threading
from pathlib import Path
from typing import Final, final

__all__ = ["ThreadLocalConnections", "configure"]

#: Long enough to ride out a contended write, short enough that a genuine
#: deadlock surfaces rather than hanging a request forever.
BUSY_TIMEOUT_MS: Final = 30_000


def configure(connection: sqlite3.Connection) -> sqlite3.Connection:
    """Apply the pragmas every Ledger database wants."""
    # WAL: readers do not block the writer, and the writer does not block
    # readers — the property that makes per-thread connections worth having.
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
    # NORMAL rather than FULL: a crash can lose the last transactions, and
    # everything here is either reconstructible (the catalog, the tombstones,
    # rebuilt by a scan) or mirrored (the operation log).
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.row_factory = sqlite3.Row
    return connection


@final
class ThreadLocalConnections:
    """Hands each thread its own connection to one database file."""

    __slots__ = ("_keepalive", "_local", "_target", "_uri")

    def __init__(self, path: Path | str, *, schema: str = "") -> None:
        # ":memory:" needs care. A plain in-memory database is private to one
        # connection, so per-thread connections would each get their own empty
        # one — the schema would appear to vanish. A uniquely-named shared-cache
        # URI gives every connection in this process the *same* database, which
        # is what a caller asking for ":memory:" actually means. It is unique
        # per instance so two stores cannot collide.
        self._uri = str(path) == ":memory:"
        if self._uri:
            self._target = f"file:ledger-{secrets.token_hex(8)}?mode=memory&cache=shared"
        else:
            self._target = str(path)

        self._local = threading.local()

        # A shared in-memory database lives only while some connection holds it
        # open, so keep one for the lifetime of this pool. For a file database
        # this is simply the connection the schema was created on.
        self._keepalive = self._open()
        if schema:
            self._keepalive.executescript(schema)

    def _open(self) -> sqlite3.Connection:
        return configure(
            sqlite3.connect(
                self._target,
                uri=self._uri,
                # Transactions are opened explicitly with BEGIN IMMEDIATE;
                # leaving autocommit to sqlite3 lets it choose DEFERRED for us,
                # which is the lost-update hazard this guards against.
                isolation_level=None,
                check_same_thread=False,
                timeout=BUSY_TIMEOUT_MS / 1000,
            )
        )

    @property
    def path(self) -> Path:
        return Path(self._target)

    def get(self) -> sqlite3.Connection:
        connection: sqlite3.Connection | None = getattr(self._local, "connection", None)
        if connection is None:
            connection = self._open()
            self._local.connection = connection
        return connection

    def close(self) -> None:
        """Close this thread's connection.

        Other threads' connections go when their threads do. Keeping a registry
        of every handle purely to close it on shutdown would be a second thing
        to get wrong, and SQLite releases them anyway.
        """
        connection: sqlite3.Connection | None = getattr(self._local, "connection", None)
        if connection is not None:
            connection.close()
            self._local.connection = None
        self._keepalive.close()

    def __repr__(self) -> str:
        return f"ThreadLocalConnections({self._target})"
