"""The metadata store: item-level primitives, and nothing else.

**Why the interface is item-level and not domain-level.** The tempting interface
has ``update_ref(env, name, expected_generation, target)`` on it. That is wrong:
it forces every backend to re-implement the generation compare-and-swap, the
dense operation counter and the idempotency triple-outcome — so per-partition
compare-and-swap and idempotent replay would be proven once *per backend*
instead of once. The primitive
interface below is exactly the shape DynamoDB ``TransactWriteItems``, Postgres
``BEGIN…COMMIT`` and FoundationDB all present, so the semantics live once in
``meta.repository`` and are tested once.

Five capabilities are required, and they map one to one:

    conditional write on a single item   → Put/Delete + Condition
    transaction across items in ONE      → transact_write, which refuses to span
      partition                            more than one partition key
    strongly consistent single-item read → get / batch_get
    secondary index (name → env_id)      → the global keyspaces
    change stream / CDC                  → Emit, written inside the transaction

**Sharding is not decoration.** One SQLite file globally serialises writers,
which would make *no operation on one environment can block one on another*
both untestable and untrue. Partitioning across N database files by
``hash(pk)`` gives genuine per-partition parallelism on a laptop and is a
faithful scale model of what a partitioned store does. Since a transaction is
scoped to one partition key, it always lands in exactly one shard.

**``BEGIN IMMEDIATE``, never ``BEGIN DEFERRED``.** This is the single most
load-bearing implementation detail in the file. Deferred takes the write lock
lazily, so two threads can both read ``generation = 41`` and only *then* contend;
a developer who retries just the write half re-applies a stale condition
evaluation and reintroduces the exact lost update compare-and-swap exists to prevent.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, final

from blake3 import blake3

from src.errors import BackendUnavailable, Conflict
from src.sqlite_support import ThreadLocalConnections

if TYPE_CHECKING:
    import sqlite3
    from collections.abc import Iterator, Mapping, Sequence


__all__ = [
    "Absent",
    "Condition",
    "Delete",
    "Emit",
    "Event",
    "GenerationIs",
    "Item",
    "Key",
    "MetadataStore",
    "Put",
    "ShardedSqliteMetadataStore",
    "VersionIs",
    "WriteOp",
    "canonical_json",
]


def canonical_json(body: Mapping[str, Any]) -> str:
    """One encoding per logical body.

    Idempotency detects "same key, different payload" by fingerprinting the
    request, and a fingerprint over a non-canonical encoding would
    report a mismatch for two spellings of the same thing — turning a correct
    retry into a 422.
    """
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


# ─────────────────────────────────────────────────────────────────────────────
# Values
# ─────────────────────────────────────────────────────────────────────────────


@final
@dataclass(frozen=True, slots=True, order=True)
class Key:
    pk: str
    sk: str


@final
@dataclass(frozen=True, slots=True)
class Item:
    key: Key
    kind: str
    body: Mapping[str, Any]
    #: Bumped on every write. The generic optimistic-concurrency handle, used
    #: wherever a record has no domain-level counter of its own.
    version: int
    created_at_us: int
    updated_at_us: int
    #: Projected out of the body so a condition can be a column comparison
    #: rather than a JSON extraction. ``None`` for records that have no such
    #: concept.
    generation: int | None = None
    expires_at_us: int | None = None


@final
@dataclass(frozen=True, slots=True)
class Event:
    """One entry of the change stream."""

    sequence: int
    shard: int
    partition: str
    event_type: str
    payload: Mapping[str, Any]
    created_at_us: int


@final
@dataclass(frozen=True, slots=True)
class StreamCursor:
    """How far a consumer has read — **one position per shard**.

    A single number will not do, and the failure is silent. Each shard numbers
    its own events from one, so a consumer that collapsed them into one scalar
    and advanced it to shard 0's position would skip every event a quieter shard
    had not yet reached — the environments on that shard would simply stop being
    built, with nothing anywhere reporting an error.

    This is also what a real partitioned stream offers. Kinesis checkpoints per
    shard for exactly this reason, so a consumer written against this cursor
    ports without changing its notion of progress.
    """

    positions: tuple[int, ...]

    @classmethod
    def start(cls, shard_count: int) -> StreamCursor:
        return cls(positions=(0,) * shard_count)

    def position(self, shard: int) -> int:
        return self.positions[shard] if shard < len(self.positions) else 0

    def advanced(self, events: Sequence[Event]) -> StreamCursor:
        """The cursor after consuming ``events``. Monotonic per shard."""
        positions = list(self.positions)
        for event in events:
            if event.shard < len(positions):
                positions[event.shard] = max(positions[event.shard], event.sequence)
        return StreamCursor(positions=tuple(positions))

    def __str__(self) -> str:
        return ",".join(str(p) for p in self.positions)


# ─────────────────────────────────────────────────────────────────────────────
# Conditions
# ─────────────────────────────────────────────────────────────────────────────


class Condition:
    """A precondition evaluated inside the transaction, never before it.

    A closed union rather than an extension point: the store must be able to
    express every condition on a backend that is not SQLite, and an open
    hierarchy would let a condition be added here that DynamoDB cannot evaluate.
    """

    __slots__ = ()


@final
@dataclass(frozen=True, slots=True)
class Absent(Condition):
    """Create-if-absent. Claims a name, or creates a tag exactly once."""


@final
@dataclass(frozen=True, slots=True)
class VersionIs(Condition):
    version: int


@final
@dataclass(frozen=True, slots=True)
class GenerationIs(Condition):
    """Compare on the ref's generation counter — **not** on its target.

    The lost update, in one sentence: a ref that moves away and back
    is indistinguishable by target alone, so a writer holding a stale read would
    succeed and silently erase every update in between. The counter makes each
    update a decision about one specific prior state.
    """

    generation: int


# ─────────────────────────────────────────────────────────────────────────────
# Write operations
# ─────────────────────────────────────────────────────────────────────────────


class WriteOp(ABC):
    @property
    @abstractmethod
    def partition(self) -> str: ...


@final
@dataclass(frozen=True, slots=True)
class Put(WriteOp):
    key: Key
    kind: str
    body: Mapping[str, Any]
    condition: Condition | None = None
    generation: int | None = None
    expires_at_us: int | None = None

    @property
    def partition(self) -> str:
        return self.key.pk


@final
@dataclass(frozen=True, slots=True)
class Delete(WriteOp):
    key: Key
    condition: Condition | None = None

    @property
    def partition(self) -> str:
        return self.key.pk


@final
@dataclass(frozen=True, slots=True)
class Emit(WriteOp):
    """Append to the change stream, inside the same transaction.

    **This is not a dual write.** The row lands in the same database, in the same
    transaction, guarded by the same commit — so an event exists exactly when the
    mutation it describes was published. A separate publish step
    would be a second thing to keep consistent, and it is the classic place a
    system starts lying about what happened.
    """

    partition_key: str
    event_type: str
    payload: Mapping[str, Any] = field(default_factory=dict)

    @property
    def partition(self) -> str:
        return self.partition_key


# ─────────────────────────────────────────────────────────────────────────────
# The interface
# ─────────────────────────────────────────────────────────────────────────────


class MetadataStore(ABC):
    """Primitives over conditioned items. No domain knowledge whatsoever."""

    @abstractmethod
    def get(self, key: Key) -> Item | None:
        """Strongly consistent single-item read.

        Strong rather than eventual because read-your-writes is required: an
        automation that just committed must see its own commit, or every write
        becomes a poll.
        """

    @abstractmethod
    def query(
        self,
        pk: str,
        sk_prefix: str,
        *,
        after: str | None = None,
        limit: int = 1000,
        descending: bool = False,
    ) -> list[Item]: ...

    @abstractmethod
    def transact_write(self, writes: Sequence[WriteOp]) -> None:
        """Apply every write, or none. Raises ``Conflict`` if a condition fails.

        Refuses at runtime if the writes span more than one partition, so
        partition isolation cannot be lost by accident rather than by decision.
        """

    @abstractmethod
    def claim_global(self, space: str, key: str, body: Mapping[str, Any]) -> bool:
        """Create-if-absent in a global keyspace. False if already claimed."""

    @abstractmethod
    def read_global(self, space: str, key: str) -> Item | None: ...

    @abstractmethod
    def put_global(self, space: str, key: str, body: Mapping[str, Any]) -> None: ...

    @abstractmethod
    def delete_global(self, space: str, key: str) -> bool: ...

    @abstractmethod
    def scan_global(self, space: str, *, prefix: str = "", limit: int = 1000) -> list[Item]: ...

    @abstractmethod
    def read_events(self, cursor: StreamCursor | None = None, *, limit: int = 100) -> list[Event]:
        """Tail the change stream. Ordered per partition, which is all that is needed."""

    @abstractmethod
    def scan_kind(self, kind: str, *, limit: int = 1000) -> list[Item]:
        """Every item of one record type, across every partition.

        The one query that deliberately crosses partitions, and it exists for the
        work queue: a build worker has to find work *somewhere* without knowing
        which environment produced it. Everything else in this interface is
        single-partition on purpose, so this is the exception that
        has to justify itself — and it does, because the alternative is polling
        ten million partitions.
        """

    @abstractmethod
    def close(self) -> None: ...


# ─────────────────────────────────────────────────────────────────────────────
# SQLite implementation
# ─────────────────────────────────────────────────────────────────────────────

_ITEMS_DDL: Final = """
CREATE TABLE IF NOT EXISTS items (
    pk            TEXT    NOT NULL,
    sk            TEXT    NOT NULL,
    kind          TEXT    NOT NULL,
    body          TEXT    NOT NULL,
    version       INTEGER NOT NULL,
    generation    INTEGER,
    expires_at_us INTEGER,
    created_at_us INTEGER NOT NULL,
    updated_at_us INTEGER NOT NULL,
    PRIMARY KEY (pk, sk)
) WITHOUT ROWID;

CREATE INDEX IF NOT EXISTS items_kind    ON items (kind, pk);
CREATE INDEX IF NOT EXISTS items_expiry  ON items (expires_at_us) WHERE expires_at_us IS NOT NULL;

CREATE TABLE IF NOT EXISTS outbox (
    seq           INTEGER PRIMARY KEY AUTOINCREMENT,
    pk            TEXT    NOT NULL,
    event_type    TEXT    NOT NULL,
    payload       TEXT    NOT NULL,
    created_at_us INTEGER NOT NULL
);
"""

_GLOBALS_DDL: Final = """
CREATE TABLE IF NOT EXISTS globals (
    space         TEXT    NOT NULL,
    key           TEXT    NOT NULL,
    body          TEXT    NOT NULL,
    version       INTEGER NOT NULL,
    created_at_us INTEGER NOT NULL,
    updated_at_us INTEGER NOT NULL,
    PRIMARY KEY (space, key)
) WITHOUT ROWID;
"""


@final
class ShardedSqliteMetadataStore(MetadataStore):
    """N SQLite files, partitioned by ``hash(pk)``.

    The shard count is fixed at creation and recorded, because changing it
    re-partitions every key — the same class of decision as a chunking
    parameter, and it fails the same quiet way if it drifts.
    """

    __slots__ = ("_globals_pool", "_pools", "_root", "_shard_count")

    def __init__(self, root: Path | str, *, shard_count: int = 16) -> None:
        if shard_count < 1:
            raise ValueError("shard_count must be positive")
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._shard_count = shard_count

        # Connections are per thread, per database — see ledger.sqlite_support
        # for why sharing one would both misbehave and undo the per-partition
        # parallelism.
        self._pools = [
            ThreadLocalConnections(self._root / f"shard-{i:02d}.db", schema=_ITEMS_DDL)
            for i in range(shard_count)
        ]
        self._globals_pool = ThreadLocalConnections(self._root / "global.db", schema=_GLOBALS_DDL)

    def _shard_connection(self, shard: int) -> sqlite3.Connection:
        return self._pools[shard].get()

    @property
    def _globals(self) -> sqlite3.Connection:
        return self._globals_pool.get()

    @property
    def _shards(self) -> list[sqlite3.Connection]:
        return [pool.get() for pool in self._pools]

    @property
    def shard_count(self) -> int:
        return self._shard_count

    def shard_of(self, pk: str) -> int:
        return blake3(pk.encode()).digest(length=4)[0] % self._shard_count

    def _connection_for(self, pk: str) -> sqlite3.Connection:
        return self._shard_connection(self.shard_of(pk))

    # ── reads ────────────────────────────────────────────────────────────────

    def get(self, key: Key) -> Item | None:
        row = (
            self._connection_for(key.pk)
            .execute("SELECT * FROM items WHERE pk = ? AND sk = ?", (key.pk, key.sk))
            .fetchone()
        )
        return _row_to_item(row) if row else None

    def query(
        self,
        pk: str,
        sk_prefix: str,
        *,
        after: str | None = None,
        limit: int = 1000,
        descending: bool = False,
    ) -> list[Item]:
        # GLOB rather than LIKE: LIKE is case-insensitive for ASCII by default in
        # SQLite, which would make `ref#Main` match a query for `ref#main`.
        clauses = ["pk = ?", "sk GLOB ?"]
        params: list[Any] = [pk, _glob_escape(sk_prefix) + "*"]
        if after is not None:
            clauses.append("sk < ?" if descending else "sk > ?")
            params.append(after)
        order = "DESC" if descending else "ASC"
        params.append(limit)

        rows = (
            self._connection_for(pk)
            .execute(
                f"SELECT * FROM items WHERE {' AND '.join(clauses)} ORDER BY sk {order} LIMIT ?",
                params,
            )
            .fetchall()
        )
        return [_row_to_item(row) for row in rows]

    # ── the transaction ──────────────────────────────────────────────────────

    def transact_write(self, writes: Sequence[WriteOp]) -> None:
        if not writes:
            return

        partitions = {write.partition for write in writes}
        if len(partitions) > 1:
            # Partition isolation made structural. A cross-partition transaction is
            # exactly what a real partitioned store cannot offer, so allowing one
            # here would let code be written that cannot be deployed.
            raise ValueError(
                f"a transaction may not span partitions: {sorted(partitions)}. "
                f"Operations that appear to need one (a rename, a fork) are "
                f"deliberately modelled as a sequence with a sweeper — see MetadataRepository."
            )

        pk = next(iter(partitions))
        connection = self._connection_for(pk)
        now = _now_us()

        # IMMEDIATE takes the write lock up front, so two writers cannot both
        # evaluate their conditions against the same pre-state and then both
        # proceed. See the module docstring. No Python-level lock: that would
        # serialise across shards and undo the per-partition parallelism.
        connection.execute("BEGIN IMMEDIATE")
        try:
            for write in writes:
                self._apply(connection, write, now)
        except BaseException:
            connection.execute("ROLLBACK")
            raise
        connection.execute("COMMIT")

    def _apply(self, connection: sqlite3.Connection, write: WriteOp, now: int) -> None:
        match write:
            case Emit():
                connection.execute(
                    "INSERT INTO outbox (pk, event_type, payload, created_at_us) "
                    "VALUES (?, ?, ?, ?)",
                    (write.partition_key, write.event_type, canonical_json(write.payload), now),
                )
            case Delete():
                existing = _fetch(connection, write.key)
                _require(write.condition, existing, write.key)
                connection.execute(
                    "DELETE FROM items WHERE pk = ? AND sk = ?", (write.key.pk, write.key.sk)
                )
            case Put():
                existing = _fetch(connection, write.key)
                _require(write.condition, existing, write.key)
                connection.execute(
                    "INSERT INTO items "
                    "(pk, sk, kind, body, version, generation, expires_at_us, "
                    " created_at_us, updated_at_us) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT (pk, sk) DO UPDATE SET "
                    "  kind = excluded.kind, body = excluded.body, "
                    "  version = excluded.version, generation = excluded.generation, "
                    "  expires_at_us = excluded.expires_at_us, "
                    "  updated_at_us = excluded.updated_at_us",
                    (
                        write.key.pk,
                        write.key.sk,
                        write.kind,
                        canonical_json(write.body),
                        (existing["version"] + 1) if existing else 1,
                        write.generation,
                        write.expires_at_us,
                        existing["created_at_us"] if existing else now,
                        now,
                    ),
                )

    # ── global keyspaces ─────────────────────────────────────────────────────

    def claim_global(self, space: str, key: str, body: Mapping[str, Any]) -> bool:
        now = _now_us()
        connection = self._globals
        connection.execute("BEGIN IMMEDIATE")
        try:
            cursor = connection.execute(
                "INSERT OR IGNORE INTO globals "
                "(space, key, body, version, created_at_us, updated_at_us) "
                "VALUES (?, ?, ?, 1, ?, ?)",
                (space, key, canonical_json(body), now, now),
            )
            claimed = cursor.rowcount == 1
        except BaseException:
            connection.execute("ROLLBACK")
            raise
        connection.execute("COMMIT")
        return claimed

    def read_global(self, space: str, key: str) -> Item | None:
        row = self._globals.execute(
            "SELECT * FROM globals WHERE space = ? AND key = ?", (space, key)
        ).fetchone()
        return _global_row_to_item(row) if row else None

    def put_global(self, space: str, key: str, body: Mapping[str, Any]) -> None:
        now = _now_us()
        self._globals.execute(
            "INSERT INTO globals (space, key, body, version, created_at_us, updated_at_us) "
            "VALUES (?, ?, ?, 1, ?, ?) "
            "ON CONFLICT (space, key) DO UPDATE SET "
            "  body = excluded.body, version = globals.version + 1, "
            "  updated_at_us = excluded.updated_at_us",
            (space, key, canonical_json(body), now, now),
        )

    def delete_global(self, space: str, key: str) -> bool:
        cursor = self._globals.execute(
            "DELETE FROM globals WHERE space = ? AND key = ?", (space, key)
        )
        return cursor.rowcount > 0

    def scan_global(self, space: str, *, prefix: str = "", limit: int = 1000) -> list[Item]:
        rows = self._globals.execute(
            "SELECT * FROM globals WHERE space = ? AND key GLOB ? ORDER BY key LIMIT ?",
            (space, _glob_escape(prefix) + "*", limit),
        ).fetchall()
        return [_global_row_to_item(row) for row in rows]

    # ── the change stream ────────────────────────────────────────────────────

    def read_events(self, cursor: StreamCursor | None = None, *, limit: int = 100) -> list[Event]:
        """Merge the per-shard outboxes, resuming each shard from its own position.

        Ordering is *per partition*, not global — which is exactly what the
        build pipeline requires ("events are ordered per environment") and
        exactly what a real partitioned stream offers. Promising a global order here would
        invent a guarantee the production system cannot keep.

        The cursor is per shard because the sequences are per shard. Resuming
        every shard from one number would skip whatever the quieter shards had
        not yet reached, and the environments living there would silently stop
        being built.
        """
        position = cursor or StreamCursor.start(self._shard_count)
        events: list[Event] = []
        for shard, connection in enumerate(self._shards):
            rows = connection.execute(
                "SELECT * FROM outbox WHERE seq > ? ORDER BY seq LIMIT ?",
                (position.position(shard), limit),
            ).fetchall()
            events.extend(
                Event(
                    sequence=row["seq"],
                    shard=shard,
                    partition=row["pk"],
                    event_type=row["event_type"],
                    payload=json.loads(row["payload"]),
                    created_at_us=row["created_at_us"],
                )
                for row in rows
            )
        # Sorted by *time*, so a consumer draining several shards sees causally
        # later events later. Within one partition this is exactly the publish
        # order, which is the only ordering required.
        events.sort(key=lambda e: (e.created_at_us, e.shard, e.sequence))
        return events[:limit]

    def scan_kind(self, kind: str, *, limit: int = 1000) -> list[Item]:
        items: list[Item] = []
        for connection in self._shards:
            rows = connection.execute(
                "SELECT * FROM items WHERE kind = ? ORDER BY pk, sk LIMIT ?", (kind, limit)
            ).fetchall()
            items.extend(_row_to_item(row) for row in rows)
        return items[:limit]

    def iter_expired(self, kind: str, now_us: int, *, limit: int = 1000) -> Iterator[Item]:
        """Items past their TTL. Drives the ephemeral-ref and idempotency sweepers."""
        for connection in self._shards:
            rows = connection.execute(
                "SELECT * FROM items WHERE kind = ? AND expires_at_us IS NOT NULL "
                "AND expires_at_us <= ? LIMIT ?",
                (kind, now_us, limit),
            ).fetchall()
            for row in rows:
                yield _row_to_item(row)

    def close(self) -> None:
        """Close this thread's connections.

        Other threads' connections are closed when their thread ends; SQLite
        cleans up a connection whose thread has gone, and holding a registry of
        every thread's handles just to close them would be a second thing to get
        wrong on shutdown.
        """
        for pool in self._pools:
            pool.close()
        self._globals_pool.close()

    def __enter__(self) -> ShardedSqliteMetadataStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _now_us() -> int:
    """Row timestamps only — never a correctness input.

    Anything that *decides* something from time (a lease expiring, a TTL
    elapsing) takes an injected ``Clock``, so it can be tested without sleeping.
    """
    import time

    return time.time_ns() // 1000


def _fetch(connection: sqlite3.Connection, key: Key) -> sqlite3.Row | None:
    row: sqlite3.Row | None = connection.execute(
        "SELECT * FROM items WHERE pk = ? AND sk = ?", (key.pk, key.sk)
    ).fetchone()
    return row


def _require(condition: Condition | None, existing: sqlite3.Row | None, key: Key) -> None:
    if condition is None:
        return
    match condition:
        case Absent():
            if existing is not None:
                raise Conflict("item already exists", pk=key.pk, sk=key.sk)
        case VersionIs(version=expected):
            actual = existing["version"] if existing else None
            if actual != expected:
                raise Conflict(
                    "item version does not match",
                    pk=key.pk,
                    sk=key.sk,
                    expected=expected,
                    actual=actual,
                )
        case GenerationIs(generation=expected):
            actual = existing["generation"] if existing else None
            if actual != expected:
                raise Conflict(
                    "generation does not match",
                    pk=key.pk,
                    sk=key.sk,
                    expected=expected,
                    actual=actual,
                )
        case _:  # pragma: no cover - the union is closed
            raise BackendUnavailable(f"unknown condition: {condition!r}")


def _glob_escape(value: str) -> str:
    """GLOB treats ``*``, ``?`` and ``[`` specially; keys may legitimately not."""
    return value.replace("[", "[[]").replace("*", "[*]").replace("?", "[?]")


def _row_to_item(row: sqlite3.Row) -> Item:
    return Item(
        key=Key(row["pk"], row["sk"]),
        kind=row["kind"],
        body=json.loads(row["body"]),
        version=row["version"],
        created_at_us=row["created_at_us"],
        updated_at_us=row["updated_at_us"],
        generation=row["generation"],
        expires_at_us=row["expires_at_us"],
    )


def _global_row_to_item(row: sqlite3.Row) -> Item:
    return Item(
        key=Key(row["space"], row["key"]),
        kind=row["space"],
        body=json.loads(row["body"]),
        version=row["version"],
        created_at_us=row["created_at_us"],
        updated_at_us=row["updated_at_us"],
    )
