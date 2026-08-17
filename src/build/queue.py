"""The build queue: ordered per environment, exclusive per environment.

Two properties that sound like they need a scheduler and do not: *events are
ordered per environment*, and *at most one build per environment is in flight*.
Both fall out of where the records are put.

**Ordering** is free because a queue entry's sort key is the operation sequence
that triggered it. The operation log already assigns a dense, gap-free counter
per environment, so draining an environment's entries in sort-key order *is*
draining them in publish order. There is no second notion of "which came first"
that could disagree with the first one — which is how version 49 would otherwise
come to sync after version 50.

**Exclusivity** is one key. A worker must hold ``blease`` in the environment's
partition to build anything in it, and there is exactly one such key, so holding
it is the mutual exclusion. No lock manager, no fencing token to compare — the
same trick the write path uses, applied again: partitioning rather than
locking, so environments never queue behind each other.

Leases expire. A worker that dies must not stop an environment building forever,
and a worker that comes back from a pause must not still believe it owns one —
so every write a worker makes on completion is conditioned on the lease it still
thinks it holds, and a stale worker's write fails rather than corrupting the
queue.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Final, final

from src.build.manifest import MAX_TIMEOUT_SECONDS
from src.build.models import QueuedBuild
from src.errors import Conflict
from src.meta import keys
from src.meta.keys import ItemKind
from src.meta.store import Absent, Delete, Key, Put, VersionIs

if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.clock import Clock
    from src.ids import EnvId, ObjectName, RefName
    from src.meta.store import Condition, Item, MetadataStore

__all__ = ["DEFAULT_LEASE_US", "LEASE_MARGIN_US", "BuildLease", "BuildQueue"]

#: How long a worker owns an environment.
#:
#: **Derived from the longest build a manifest may declare, not chosen.** A lease
#: that can expire while its build is still running is not a safety net, it is a
#: livelock: the lease is taken over, the original worker's ``complete`` fails
#: its version check, the work goes back on the queue, and the next worker
#: repeats it — taking just as long, and losing it at exactly the same point.
#: Nothing escalates, because that path never reaches ``fail`` and so never
#: increments ``attempts``; ``MAX_ATTEMPTS`` cannot end what it never counts.
#:
#: It bit at fifteen minutes against an hour of permitted build time: every build
#: over fifteen minutes span forever, while a typical one — which the pool sizing
#: assumes takes about five — was fine. That is the shape of it: invisible in
#: tests and demos, certain in production, and worst for the heaviest
#: environments.
#:
#: The margin covers setup and the sync note either side of the run itself.
#: Renewal would let this be shorter, but only with a heartbeat running alongside
#: a blocking subprocess; a constant that cannot be wrong is worth more than a
#: thread that must not fail.
#:
#: The cost is stated rather than hidden: an environment whose worker *dies* is
#: unavailable for this long. Without heartbeats, a dead worker and a slow one
#: are indistinguishable before the longest legitimate build has had time to
#: finish, so this is the soonest an honest takeover can happen.
LEASE_MARGIN_US: Final = 5 * 60 * 1_000_000
DEFAULT_LEASE_US: Final = MAX_TIMEOUT_SECONDS * 1_000_000 + LEASE_MARGIN_US

#: How many times a build is retried before its failure is recorded and the entry
#: is dropped. A failed build **never blocks or reverses the commit that
#: triggered it**, so the queue has to give up rather than spin.
MAX_ATTEMPTS: Final = 3


@final
@dataclass(frozen=True, slots=True)
class BuildLease:
    """Ownership of one environment's builds, for a bounded time."""

    env_id: str
    worker: str
    expires_at_us: int
    #: The item version the lease was claimed at. Every later write is
    #: conditioned on it, so a worker whose lease was taken over cannot write.
    version: int
    work: QueuedBuild

    @property
    def sort_key(self) -> str:
        return keys.build_queue_sk(self.work.op_sequence)


@final
class BuildQueue:
    """Enqueue work, lease an environment, finish or fail."""

    __slots__ = ("_clock", "_lease_us", "_store")

    def __init__(
        self, store: MetadataStore, *, clock: Clock, lease_us: int = DEFAULT_LEASE_US
    ) -> None:
        self._store = store
        self._clock = clock
        self._lease_us = lease_us

    # ── producing ────────────────────────────────────────────────────────────

    def enqueue(
        self,
        env: EnvId | str,
        commit: ObjectName | str,
        *,
        ref: RefName | str,
        op_sequence: int,
        manual: bool = False,
    ) -> bool:
        """Add work. Idempotent on the operation that triggered it.

        Delivery from the change stream is at-least-once, so the same ref update
        will sometimes arrive twice. Keying the entry on the operation sequence
        makes the second arrival a no-op rather than a second build — the effect
        is exactly-once without anything having to deduplicate.
        """
        partition = str(env)
        entry = QueuedBuild(
            env_id=partition,
            commit=str(commit),
            ref=str(ref),
            op_sequence=op_sequence,
            enqueued_at_us=self._clock.now_us(),
            manual=manual,
        )
        try:
            self._store.transact_write(
                [
                    Put(
                        Key(partition, keys.build_queue_sk(op_sequence)),
                        ItemKind.BUILD_QUEUE,
                        entry.to_body(),
                        condition=Absent(),
                    )
                ]
            )
        except Conflict:
            return False
        return True

    # ── consuming ────────────────────────────────────────────────────────────

    def lease(self, *, worker: str, limit: int = 200) -> BuildLease | None:
        """Take the oldest unleased environment's oldest queued build.

        Scans across partitions — the one query in the system that does — because
        a worker has to find work without knowing which environment produced it.
        Everything else stays single-partition, so this is the exception, and the
        alternative is polling ten million partitions.
        """
        now = self._clock.now_us()
        pending = self._store.scan_kind(ItemKind.BUILD_QUEUE, limit=limit)
        for partition, entry in _oldest_per_partition(pending):
            claimed = self._claim(partition, worker=worker, now=now)
            if claimed is None:
                continue  # someone else owns this environment right now
            return BuildLease(
                env_id=partition,
                worker=worker,
                expires_at_us=now + self._lease_us,
                version=claimed,
                work=QueuedBuild.from_body(entry.body),
            )
        return None

    def _claim(self, partition: str, *, worker: str, now: int) -> int | None:
        """Claim the environment's build lease. Returns the new item version."""
        key = Key(partition, keys.build_lease_sk())
        current = self._store.get(key)
        body = {"worker": worker, "expires_at_us": now + self._lease_us}
        condition: Condition

        if current is None:
            condition = Absent()
            expected_version = 1
        elif int(current.body.get("expires_at_us", 0)) > now:
            return None  # a live lease, held by someone else
        else:
            # Expired. Taking it over is conditioned on the version we read, so
            # two workers racing to reclaim the same dead lease cannot both win.
            condition = VersionIs(current.version)
            expected_version = current.version + 1

        try:
            self._store.transact_write(
                [
                    Put(
                        key,
                        ItemKind.BUILD_LEASE,
                        body,
                        condition=condition,
                        expires_at_us=now + self._lease_us,
                    )
                ]
            )
        except Conflict:
            return None
        return expected_version

    # ── finishing ────────────────────────────────────────────────────────────

    def complete(self, lease: BuildLease, *, extra: Sequence[Put] | None = None) -> None:
        """Remove the entry and release the lease, in one transaction.

        ``extra`` rides along so the sync note is written by the same commit that
        finishes the work: either the environment records what happened and the
        work disappears, or neither. Both land in the environment's partition, so
        this stays a single-partition transaction and no environment can block another.
        """
        self._store.transact_write(
            [
                Delete(Key(lease.env_id, lease.sort_key)),
                Delete(
                    Key(lease.env_id, keys.build_lease_sk()),
                    condition=VersionIs(lease.version),
                ),
                *(extra or ()),
            ]
        )

    def fail(self, lease: BuildLease) -> bool:
        """Record an attempt and release the lease.

        Returns True when the entry was dropped for good. A failing build must
        never hold its environment: a failure neither blocks nor reverses the
        commit that triggered it, so after ``MAX_ATTEMPTS`` the entry goes and
        the failure lives in the result store where it can be queried in
        aggregate.
        """
        attempts = lease.work.attempts + 1
        exhausted = attempts >= MAX_ATTEMPTS
        entry_key = Key(lease.env_id, lease.sort_key)
        release = Delete(
            Key(lease.env_id, keys.build_lease_sk()), condition=VersionIs(lease.version)
        )

        if exhausted:
            self._store.transact_write([Delete(entry_key), release])
            return True

        self._store.transact_write(
            [
                Put(
                    entry_key,
                    ItemKind.BUILD_QUEUE,
                    replace(lease.work, attempts=attempts).to_body(),
                ),
                release,
            ]
        )
        return False

    def release(self, lease: BuildLease) -> None:
        """Give the environment back without touching the entry — the work stays
        queued and someone else will take it.
        """
        try:
            self._store.transact_write(
                [
                    Delete(
                        Key(lease.env_id, keys.build_lease_sk()),
                        condition=VersionIs(lease.version),
                    )
                ]
            )
        except Conflict:
            return  # already taken over; nothing of ours to give back

    # ── inspection ───────────────────────────────────────────────────────────

    def depth(self, *, limit: int = 10_000) -> int:
        return len(self._store.scan_kind(ItemKind.BUILD_QUEUE, limit=limit))

    def pending(self, env: EnvId | str, *, limit: int = 100) -> list[QueuedBuild]:
        items = self._store.query(str(env), keys.build_queue_prefix(), limit=limit)
        return [QueuedBuild.from_body(item.body) for item in items]


def _oldest_per_partition(items: Sequence[Item]) -> list[tuple[str, Item]]:
    """One candidate per environment: its earliest queued operation.

    Taking only the earliest is what preserves per-environment order. Taking the
    globally earliest across environments as well would be a fairness policy, and
    a wrong one — a busy environment would starve every other.
    """
    oldest: dict[str, Item] = {}
    for item in items:
        current = oldest.get(item.key.pk)
        if current is None or item.key.sk < current.key.sk:
            oldest[item.key.pk] = item
    return sorted(oldest.items(), key=lambda pair: pair[1].body.get("enqueued_at_us", 0))
