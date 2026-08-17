"""Garbage collection: reclaiming exactly what nothing needs.

An object is retained exactly while it is reachable from a live ref, a retained
operation-log entry, or an open write lease. Everything else is
garbage. The question is how to find it without ever deleting something in use.

**Why reference counting loses.** It is the obvious answer and it is wrong here.
Under global deduplication a single chunk may be referenced by thousands of
unrelated environments, so every commit fans out into thousands of counter
updates — and the failure modes are asymmetric: a lost increment deletes live
data, a lost decrement leaks forever, and neither is detectable without a full
recount. That puts a distributed-correctness problem on the commit path, which
is exactly where the read/write ratio says we cannot afford one.

**What happens instead — epoch-based mark and sweep with per-environment
keep-sets:**

    (1) freeze an epoch    record cutoff T; nothing written after T is a
         │                 deletion candidate in this cycle
         ▼
    (2) gather roots       live refs, retained op-log commits, open leases
         │
         ▼
    (3) shard and diff     partition all hashes by prefix; per shard, compare
         │                 the live set against the stored set. Exact.
         ▼
    (4) sweep              delete candidates, and record a tombstone for each
         │
         ▼
    (5) reclaim            report what came back

**Deletion has six independent guards**, because it is the only destructive
operation in the system. Each is implemented as a named ``Guard`` so that
"which guard stopped this?" has an answer, and so each has its own test.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Final, Protocol, final, runtime_checkable

from src.errors import LedgerError
from src.format.model import Commit
from src.fs.closure import tree_closure
from src.ids import EnvName, EpochId, ObjectName
from src.metrics import COUNTERS

if TYPE_CHECKING:
    from collections.abc import Iterable

    from src.clock import Clock
    from src.ids import EnvId, SessionId
    from src.meta.models import OpLogEntry, Ref, WriteSession
    from src.store.cas import ObjectStore
    from src.store.digests import DigestIndex
    from src.store.keepsets import KeepSetStore

__all__ = ["GarbageCollector", "GcConfig", "GcPlan", "GcReport", "GuardName", "RootSource"]

logger = logging.getLogger("ledger.gc")

#: Seven days past the cutoff. Protects content a slow or
#: paused writer is still assembling, beyond what a write lease covers.
DEFAULT_GRACE_US: Final = 7 * 86_400 * 1_000_000

#: A tombstone outlives the sweep that created it by one full collection period,
#: which is longer than any in-flight write session can hold a stale "present".
DEFAULT_TOMBSTONE_TTL_US: Final = 2 * 86_400 * 1_000_000

#: How far back the operation log is retained.
#:
#: **This is a storage decision, not a logging one, and nothing else does
#: not settle it.** A retained op-log entry is a retention root — it has
#: to, because undo restores a ref to an old commit and cannot resurrect objects
#: collection has already reclaimed — but it never says how long an entry is
#: retained. That omission is the whole decision: log retention is *exactly* how
#: far back undo is guaranteed to work, and *exactly* how long a discarded
#: branch's content stays alive in order to make that guarantee true.
#:
#: So the two properties are one number, and they pull in opposite directions.
#: Ninety days is the safe end of it — deep undo, slow reclamation. Shortening it
#: is a deliberate trade of undo depth for storage returned sooner, which is why
#: it is a knob on ``GcConfig`` rather than a constant anyone has to patch.
DEFAULT_OP_LOG_RETENTION_US: Final = 90 * 86_400 * 1_000_000

#: How many commits one keep-set rebuild will walk before giving up.
#:
#: Generous against the capacity plan's fifty versions per environment, and
#: bounded so a pathological history cannot make a maintenance job run forever.
#: Hitting it does **not** shrink the keep-set — see ``rebuild_keep_set``, where
#: the safe response to a partial answer is to keep the stale set.
MAX_HISTORY_WALK: Final = 100_000


class GuardName(StrEnum):
    """The six guards, named so a refusal can say which
    fired."""

    EPOCH_CUTOFF = "epoch_cutoff"
    GRACE_PERIOD = "grace_period"
    WRITE_LEASE = "write_lease"
    KEEP_SET = "keep_set"
    TOMBSTONE = "tombstone"
    CIRCUIT_BREAKER = "circuit_breaker"


@final
@dataclass(frozen=True, slots=True)
class GcConfig:
    grace_us: int = DEFAULT_GRACE_US
    tombstone_ttl_us: int = DEFAULT_TOMBSTONE_TTL_US
    #: See DEFAULT_OP_LOG_RETENTION_US: this directly bounds how soon a
    #: discarded branch's storage can come back.
    op_log_retention_us: int = DEFAULT_OP_LOG_RETENTION_US
    #: Hash-prefix bits used to partition the diff. Four (sixteen shards) is
    #: ample on one machine; production uses twelve. The algorithm is identical.
    shard_bits: int = 4
    #: Abort a cycle that proposes deleting more than this fraction of the corpus.
    max_delete_fraction: float = 0.05
    #: …but only once the corpus is at least this large. Without a floor the
    #: breaker fires on every integration test and gets disabled — which is how
    #: circuit breakers die.
    min_corpus_objects: int = 1000


@final
@dataclass(frozen=True, slots=True)
class GcPlan:
    epoch: EpochId
    cutoff_us: int
    candidates: tuple[ObjectName, ...]
    bytes_reclaimable: int
    corpus_objects: int
    corpus_bytes: int
    live_objects: int
    protected: dict[str, int] = field(default_factory=dict)
    aborted_by: GuardName | None = None
    abort_reason: str = ""

    @property
    def aborted(self) -> bool:
        return self.aborted_by is not None


@final
@dataclass(frozen=True, slots=True)
class GcReport:
    plan: GcPlan
    enforced: bool
    deleted: int = 0
    bytes_freed: int = 0
    tombstones_written: int = 0

    @property
    def aborted(self) -> bool:
        return self.plan.aborted


@runtime_checkable
class RootSource(Protocol):
    """Where retention roots come from — refs, the operation log, open leases.

    Named as a Protocol here rather than taking the metadata repository directly
    because ``store`` sits **below** ``meta``: this package's own docstring says
    it knows nothing about environments or refs, and importing the repository to
    ask it for refs would make that false. ``MetadataRepository`` satisfies this
    structurally, so the wiring is unchanged and the dependency points the way
    the layering says it does.

    It is deliberately the *smallest* surface that answers "what is still
    reachable": five reads, no writes. A collector that could mutate metadata
    would be a collector that could lose a ref.
    """

    def list_envs(self, *, prefix: str = "", limit: int = 100) -> list[EnvName]: ...

    def resolve_env_name(self, name: EnvName) -> EnvId: ...

    def list_refs(self, env: EnvId) -> list[Ref]: ...

    def list_ops(self, env: EnvId, *, limit: int = 100) -> list[OpLogEntry]: ...

    def list_sessions(self, env: EnvId) -> list[WriteSession]: ...

    def uploaded_objects(self, env: EnvId, session: SessionId) -> list[ObjectName]: ...


@final
class GarbageCollector:
    """The epoch state machine. Orchestration only — every mechanism is a
    collaborator, which is what keeps report-only mode and crash-resumption from
    being bolted on.
    """

    __slots__ = ("_clock", "_config", "_digests", "_keepsets", "_roots", "_store")

    def __init__(
        self,
        store: ObjectStore,
        keepsets: KeepSetStore,
        roots: RootSource,
        *,
        clock: Clock,
        digests: DigestIndex,
        config: GcConfig | None = None,
    ) -> None:
        self._store = store
        self._keepsets = keepsets
        self._roots = roots
        self._clock = clock
        self._digests = digests
        self._config = config or GcConfig()

    # ── the cycle ────────────────────────────────────────────────────────────

    def plan(self) -> GcPlan:
        """Decide what would be deleted, without deleting anything.

        Safe to run at any time, and the default mode: a collector that can only
        be run destructively is a collector nobody runs.
        """
        now = self._clock.now_us()
        cutoff = now - self._config.grace_us
        corpus_objects, corpus_bytes = self._store.catalog.total()

        # Held whole, deliberately: leases cover only what open write sessions
        # have uploaded, which is bounded by concurrency rather than by corpus
        # size. The keep-sets are the opposite and are read one shard at a time.
        leased = self._leased()

        candidates: list[ObjectName] = []
        reclaimable = 0
        too_recent = 0
        live_objects = 0

        for shard in range(1 << self._config.shard_bits):
            # **One shard's live set at a time.** Both sides are partitioned by
            # the same hash prefix, so an object in this shard of the catalog can
            # only be kept alive by this shard of the keep-sets — which is the
            # entire reason the diff is sharded, and the reason the catalog can
            # promise an exact answer with no Bloom filter anywhere.
            #
            # Holding every shard's live set at once, as this once did, gives the
            # right answer and quietly costs O(corpus) memory: at the capacity plan's
            # year-10 numbers that is tens of gigabytes in one process, and the
            # collector becomes the largest consumer of memory in a system whose
            # entire read path is designed to stream.
            live = frozenset(self._keepsets.iter_shard(self._config.shard_bits, shard))
            live_objects += len(live)
            for entry in self._store.catalog.iter_shard(self._config.shard_bits, shard):
                if entry.name in live or entry.name in leased:
                    continue
                # Guards 1 and 2: nothing written after the cutoff is a candidate,
                # which covers both the epoch freeze and the grace period. They
                # are one comparison because the cutoff already has the grace
                # subtracted — but they are distinct reasons, so both are counted.
                if entry.written_at_us > cutoff:
                    too_recent += 1
                    continue
                candidates.append(entry.name)
                reclaimable += entry.size

        protected = {
            GuardName.KEEP_SET.value: live_objects,
            GuardName.WRITE_LEASE.value: len(leased),
            GuardName.GRACE_PERIOD.value: too_recent,
        }

        plan = GcPlan(
            epoch=EpochId(now),
            cutoff_us=cutoff,
            candidates=tuple(candidates),
            bytes_reclaimable=reclaimable,
            corpus_objects=corpus_objects,
            corpus_bytes=corpus_bytes,
            live_objects=live_objects,
            protected=protected,
        )
        return self._apply_circuit_breaker(plan)

    def run(self, *, enforce: bool = False) -> GcReport:
        """Execute a cycle.

        ``enforce`` defaults to False. Deletion is the only destructive
        operation in the system, so making the destructive mode the one you have
        to ask for is worth the extra flag.
        """
        plan = self.plan()
        if plan.aborted or not enforce:
            return GcReport(plan=plan, enforced=False)

        # Guard 5: every swept hash is tombstoned *before* the bytes go, so a
        # concurrent HasObjects can never answer "present" for an object that is
        # about to vanish. See store.tombstone for why this is load-bearing.
        expiry = self._clock.now_us() + self._config.tombstone_ttl_us
        tombstones = self._store.tombstones.record(plan.candidates, expires_at_us=expiry)

        deleted = self._store.delete(plan.candidates)

        # The alternate-digest index is a hint, and every reader confirms a hit
        # against the store — but leaving rows pointing at swept objects would
        # make every re-ingest of that content pay for a lookup that always
        # misses. Dropped here rather than expired on a timer, because the sweep
        # is the moment the fact stops being true.
        self._digests.forget(plan.candidates)

        return GcReport(
            plan=plan,
            enforced=True,
            deleted=deleted,
            bytes_freed=plan.bytes_reclaimable,
            tombstones_written=tombstones,
        )

    # ── roots ────────────────────────────────────────────────────────────────

    def _leased(self) -> set[ObjectName]:
        """Guard 3: objects uploaded under a session whose lease is still open.

        A session's objects are GC roots until it ends, which is
        what makes an abandoned write need no cleanup by anyone — the bytes are
        just unreferenced content on a timer.
        """
        now = self._clock.now_us()
        leased: set[ObjectName] = set()
        for name in self._roots.list_envs(limit=10_000):
            env = self._roots.resolve_env_name(name)
            for session in self._roots.list_sessions(env):
                if session.expires_at_us > now:
                    leased.update(self._roots.uploaded_objects(env, session.session_id))
        return leased

    # ── keep-set maintenance ─────────────────────────────────────────────────

    def graduate(self, env: str, names: Iterable[ObjectName]) -> int:
        """Move a completed write's objects into an environment's keep-set.

        Called after the ref has moved and *before* the write session ends, so
        the objects are covered by the lease right up to the moment they are
        covered by the keep-set. Without that overlap a collection cycle
        freezing in the window would delete a just-published commit's content.
        """
        return self._keepsets.merge(env, names)

    def refresh_keep_sets(self, *, limit: int = 10_000) -> int:
        """Recompute every environment's keep-set. Returns how many were rebuilt.

        A keep-set is a *time-dependent* answer, and that is easy to miss. It
        holds what an environment's refs and its **retained** operation log
        reach, so it goes stale two ways: when a ref is deleted, and — later, on
        a timer nobody triggers — when the operation-log entry that was keeping a
        discarded branch alive finally ages out.

        Only the first has an obvious moment to act on. Without the second,
        deleting a branch shrinks nothing (undo can still reach it, correctly),
        the log ages out an hour or a month later, and no code runs at that
        instant — so the content stays in the keep-set forever and the claim that
        discarding a branch frees its storage is never true in a running system.

        **This walks every environment**, which is fine at one-machine scale and
        is not the shape of the eventual answer. Production tracks environments
        whose operation log has entries crossing the horizon and rebuilds only
        those; the walk is what makes the property true today, and the narrowing
        is an optimisation of *which* environments rather than of what is done to
        them.
        """
        rebuilt = 0
        for name in self._roots.list_envs(limit=limit):
            self.rebuild_keep_set(str(name))
            rebuilt += 1
        return rebuilt

    def rebuild_keep_set(self, env_name: str) -> int:
        """Recompute one environment's keep-set from its remaining roots.

        The only expensive direction. Addition is free because a commit's closure
        is fixed forever; subtraction is not, because an object may have been
        reachable through the ref that just went — or through an operation-log
        entry that has since aged out.

        **A rebuild walks history, not just ref tips**, and the difference is
        data loss. Maintenance never removes anything: every published commit
        graduates its closure into the keep-set and the set only grows, so a
        running environment accumulates all of its ancestors. A rebuild has to
        arrive at the same answer minus what genuinely became unreachable — and
        a rebuild that walked only the tips would arrive at something strictly
        smaller. Once the operation log aged past its horizon, every earlier
        version of every environment would fall out of the keep-set and the next
        sweep would take it, quietly breaking ``log``, ``diff`` and restoring an
        earlier version — which are four fifths of what a version control system
        is for.

        Ancestry is exactly what "reachable from a live ref" has always meant.
        The reason for not crawling history is that the *steady state*
        never rebuilds; it is not a licence for the rebuild to be wrong when it
        does run.
        """
        env = self._roots.resolve_env_name(EnvName(env_name))
        reachable: set[ObjectName] = set()
        visited: set[ObjectName] = set()
        complete = True

        for ref in self._roots.list_refs(env):
            complete &= self._walk_history(ref.target, reachable, visited)

        # Commits named in the *retained* operation log are roots too: undo
        # restores a ref to an old commit, and it cannot resurrect objects
        # collection has already reclaimed.
        #
        # The age filter is what makes discarded content eventually collectable.
        # Without it a DeleteRef entry keeps the branch it deleted alive forever,
        # and "discarding a branch frees its storage" is never true.
        horizon = self._clock.now_us() - self._config.op_log_retention_us
        for entry in self._roots.list_ops(env, limit=10_000):
            if entry.at_us < horizon:
                continue
            for commit in (entry.before, entry.after):
                if commit is not None:
                    complete &= self._walk_history(commit, reachable, visited)

        if not complete:
            # The walk hit its bound, so ``reachable`` is a *subset* of what this
            # environment reaches. Replacing the keep-set with a subset is the
            # one mistake here that deletes live data, and leaving a stale
            # keep-set in place only delays reclamation. So the stale one stands,
            # loudly.
            logger.error(
                "keep-set rebuild for %s walked %d commits without finishing; "
                "leaving the existing keep-set in place rather than shrinking it "
                "to a partial answer",
                env_name,
                MAX_HISTORY_WALK,
            )
            return self._keepsets.size(str(env))

        return self._keepsets.replace(str(env), reachable)

    def _walk_history(
        self,
        head: ObjectName,
        reachable: set[ObjectName],
        visited: set[ObjectName],
    ) -> bool:
        """Add ``head`` and every ancestor's closure to ``reachable``.

        ``visited`` spans the whole rebuild, so a commit shared by several refs —
        the overwhelmingly common case, since a branch and its trunk share all
        but a handful — is walked once. Returns whether the walk finished; a
        ``False`` means the caller must not shrink anything.

        The commit is decoded **once** and used for both its tree and its
        parents. Asking ``commit_closure`` for the closure and then asking the
        store again for the parents would double every fetch in the walk, which
        on the environment with the deepest history is the difference between a
        maintenance job and an incident.
        """
        frontier = [head]
        walked = 0
        while frontier:
            commit = frontier.pop()
            if commit in visited:
                continue
            if walked >= MAX_HISTORY_WALK:
                return False
            visited.add(commit)
            walked += 1

            node = self._commit(commit)
            if node is None:
                # A commit this store does not hold — a ref restored from a
                # backup, or an ancestor a previous sweep already took. It
                # contributes nothing and names no parents; it must not stop the
                # walk finding the rest of the history.
                continue
            reachable.add(commit)
            reachable.update(self._closure(node.tree))
            frontier.extend(node.parents)
        return True

    def _commit(self, name: ObjectName) -> Commit | None:
        try:
            return self._store.get_as(name, Commit)
        except LedgerError:
            return None

    def _closure(self, tree: ObjectName) -> set[ObjectName]:
        """Every object a tree reaches, or nothing if it cannot be read.

        A rebuild walks whatever the refs and the retained operation log name,
        and either may reach a node this store does not hold. Refusing to
        rebuild because one root is unreadable would leave the *whole*
        environment's keep-set stale, which is far worse than under-counting one
        root: the diff would then propose deleting everything else the
        environment reaches.
        """
        try:
            return set(tree_closure(self._store, tree))
        except LedgerError:
            return set()

    # ── the circuit breaker ──────────────────────────────────────────────────

    def _apply_circuit_breaker(self, plan: GcPlan) -> GcPlan:
        """Guard 6: refuse a cycle that proposes deleting an implausible share.

        The floor matters as much as the fraction. Without ``min_corpus_objects``
        the breaker fires on every small corpus — including every integration
        test — and the first thing anyone does with a breaker that cries wolf is
        turn it off.
        """
        from dataclasses import replace as _replace

        if plan.corpus_objects < self._config.min_corpus_objects:
            return plan
        if not plan.candidates:
            return plan

        fraction = len(plan.candidates) / plan.corpus_objects
        if fraction <= self._config.max_delete_fraction:
            return plan

        # The other signal must never be silent. A breaker that
        # trips and is only visible in a return value is a breaker nobody notices
        # until the backlog does.
        COUNTERS.increment("ledger_gc_circuit_breaker_trips_total")
        return _replace(
            plan,
            aborted_by=GuardName.CIRCUIT_BREAKER,
            abort_reason=(
                f"this cycle proposes deleting {len(plan.candidates):,} of "
                f"{plan.corpus_objects:,} objects ({fraction:.1%}), above the "
                f"{self._config.max_delete_fraction:.0%} limit — refusing"
            ),
        )
