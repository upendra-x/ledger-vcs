"""Garbage collection: reclaiming exactly what nothing needs.

This file carries the requirement nothing else can demonstrate — *discarding
a branch frees whatever storage it used* — and the subtlest bug in the whole
design, resurrection by deduplication.

Every test that matters here is a *negative* one: not "the collector reclaims
things" but "the collector refuses to reclaim this specific thing, for this
specific reason". Deletion is the only destructive operation in the system, so
the guards are the product.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any, cast

import pytest

from src.clock import ManualClock
from src.errors import CorruptObject, ObjectNotFound
from src.format.cdc import ChunkParams
from src.format.model import Chunk
from src.format.shape import ShapeParams
from src.fs.closure import commit_closure
from src.ids import EnvName, RefName
from src.instance import Ledger
from src.maintenance.gc import GarbageCollector, GcConfig, GuardName
from src.service.commits import CommitService
from src.store.tombstone import NullTombstoneStore

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence
    from pathlib import Path

    from src.ids import EnvId, ObjectName
    from src.store.catalog import CatalogEntry, WriteCatalog
    from src.store.keepsets import KeepSetStore

MAIN = RefName("refs/heads/main")
BRANCH = RefName("refs/heads/exp/lr-3e4")

#: A short grace period and log retention, so tests can step past both without
#: sleeping. The relationship they encode is the real one: a discarded branch's
#: content stays alive exactly as long as the operation log can still undo the
#: deletion.
FAST_GC = GcConfig(
    grace_us=60 * 1_000_000,
    op_log_retention_us=300 * 1_000_000,
    shard_bits=2,
    min_corpus_objects=1000,
)


@pytest.fixture
def clock() -> ManualClock:
    return ManualClock(start_us=1_700_000_000_000_000)


@pytest.fixture
def ledger(tmp_path: Path, clock: ManualClock) -> Iterator[Ledger]:
    with Ledger(
        tmp_path / "ledger",
        clock=clock,
        chunk_params=ChunkParams.for_average(4096),
        shape_params=ShapeParams(
            domain=b"ledger.tree.split.v1",
            period=32,
            min_entries=4,
            max_entries=64,
            max_node_bytes=16 * 1024,
        ),
        shard_count=4,
    ) as opened:
        yield opened


@pytest.fixture
def collector(ledger: Ledger, clock: ManualClock) -> GarbageCollector:
    return GarbageCollector(
        ledger.store,
        ledger.keepsets,
        ledger.repo,
        clock=clock,
        digests=ledger.digests,
        config=FAST_GC,
    )


@pytest.fixture
def commits(ledger: Ledger) -> CommitService:
    return CommitService(ledger)


@pytest.fixture
def env(ledger: Ledger) -> EnvId:
    return ledger.repo.create_env(EnvName("proximal/demo"), owner="agent-17").env_id


@pytest.fixture
def source(tmp_path: Path) -> Path:
    root = tmp_path / "env"
    (root / "task").mkdir(parents=True)
    (root / "data").mkdir()
    (root / "harbor.yaml").write_text("name: demo\n")
    (root / "task" / "prompt.md").write_text("Solve the failing test.\n")
    (root / "data" / "train.bin").write_bytes(random.Random(42).randbytes(120_000))
    return root


def past_grace(clock: ManualClock) -> None:
    """Step beyond the grace period so recent writes stop being protected."""
    clock.advance_seconds(120)


class _CountingCommitReads:
    """Records which commits a walk decoded. Forwards everything else."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.commits: list[ObjectName] = []

    def get_as(self, name: ObjectName, expected: Any) -> Any:
        from src.format.model import Commit

        if expected is Commit:
            self.commits.append(name)
        return self._inner.get_as(name, expected)

    def __getattr__(self, item: str) -> Any:
        return getattr(self._inner, item)


class _Trace:
    """The order in which a collection cycle touched each side of the diff.

    Both sides are decorated because the defect this catches is invisible in
    either one alone: reading every keep-set shard up front and reading them one
    at a time make exactly the same calls, in the same number, and produce
    exactly the same plan. Only the *interleaving* differs — and the
    interleaving is the whole memory bound.
    """

    def __init__(self) -> None:
        self.events: list[tuple[str, int]] = []

    def keepsets(self, inner: KeepSetStore) -> _WatchedKeepSets:
        return _WatchedKeepSets(inner, self)

    def catalog(self, inner: WriteCatalog) -> _WatchedCatalog:
        return _WatchedCatalog(inner, self)

    @property
    def shape(self) -> list[str]:
        """The sequence with shard numbers dropped — ``["live", "stored", …]``."""
        return [side for side, _ in self.events]


class _WatchedKeepSets:
    """A keep-set store that reports when each shard was read.

    A decorator rather than a patched method: ``SqliteKeepSetStore`` uses
    ``__slots__``, and reaching past that to rewrite an attribute would be
    testing a hole in the class rather than the collector's behaviour.
    """

    def __init__(self, inner: KeepSetStore, trace: _Trace) -> None:
        self._inner = inner
        self._trace = trace

    def iter_shard(self, prefix_bits: int, shard: int) -> Iterator[ObjectName]:
        self._trace.events.append(("live", shard))
        return self._inner.iter_shard(prefix_bits, shard)

    def merge(self, env: str, names: Iterable[ObjectName]) -> int:
        return self._inner.merge(env, names)

    def replace(self, env: str, names: Iterable[ObjectName]) -> int:
        return self._inner.replace(env, names)

    def drop(self, env: str) -> int:
        return self._inner.drop(env)

    def live(self, names: Sequence[ObjectName]) -> set[ObjectName]:
        return self._inner.live(names)

    def holds(self, env: str, names: Sequence[ObjectName]) -> set[ObjectName]:
        return self._inner.holds(env, names)

    def size(self, env: str | None = None) -> int:
        return self._inner.size(env)


class _WatchedCatalog:
    """The stored side of the diff, reporting the same way.

    Swapped onto the object store rather than passed in, because the store takes
    its catalog by injection and this is that injection happening late. The
    attribute is in ``__slots__``, so this is assignment to a declared field, not
    a hole punched in the class.
    """

    def __init__(self, inner: WriteCatalog, trace: _Trace) -> None:
        self._inner = inner
        self._trace = trace

    def iter_shard(self, prefix_bits: int, shard: int) -> Iterator[CatalogEntry]:
        self._trace.events.append(("stored", shard))
        return self._inner.iter_shard(prefix_bits, shard)

    def record(self, entries: Sequence[CatalogEntry]) -> None:
        self._inner.record(entries)

    def forget(self, names: Iterable[ObjectName]) -> int:
        return self._inner.forget(names)

    def total(self) -> tuple[int, int]:
        return self._inner.total()


class TestReportOnlyByDefault:
    def test_planning_deletes_nothing(
        self,
        ledger: Ledger,
        collector: GarbageCollector,
        commits: CommitService,
        env: EnvId,
        source: Path,
        clock: ManualClock,
    ) -> None:
        """A collector that can only be run destructively is one nobody runs."""
        commits.commit(env, MAIN, source, author="a", message="v1")
        before = ledger.store.catalog.total()

        past_grace(clock)
        plan = collector.plan()

        assert ledger.store.catalog.total() == before
        assert not plan.aborted

    def test_run_without_enforce_deletes_nothing(
        self,
        ledger: Ledger,
        collector: GarbageCollector,
        commits: CommitService,
        env: EnvId,
        source: Path,
        clock: ManualClock,
    ) -> None:
        commits.commit(env, MAIN, source, author="a", message="v1")
        before = ledger.store.catalog.total()
        past_grace(clock)

        report = collector.run()
        assert not report.enforced
        assert report.deleted == 0
        assert ledger.store.catalog.total() == before


class TestLiveContentIsNeverCollected:
    def test_a_committed_environment_survives(
        self,
        ledger: Ledger,
        collector: GarbageCollector,
        commits: CommitService,
        env: EnvId,
        source: Path,
        clock: ManualClock,
    ) -> None:
        head = commits.commit(env, MAIN, source, author="a", message="v1")
        past_grace(clock)

        report = collector.run(enforce=True)
        assert report.deleted == 0, "nothing reachable from a live ref may be collected"

        # And it is still readable, end to end.
        from src.format.model import Commit
        from src.fs.tree import resolve_path

        tree = ledger.store.get_as(head.commit, Commit).tree
        assert resolve_path(ledger.store, tree, "task/prompt.md") is not None

    def test_undo_still_works_after_a_sweep(
        self,
        ledger: Ledger,
        collector: GarbageCollector,
        commits: CommitService,
        env: EnvId,
        source: Path,
        clock: ManualClock,
    ) -> None:
        """The coupling made explicit, end to end.

        Undo restores a ref to an old commit, so as long as the log can undo an
        operation, the content that operation touched must still be there.
        """
        first = commits.commit(env, MAIN, source, author="a", message="v1")
        (source / "task" / "prompt.md").write_text("v2\n")
        second = commits.commit(env, MAIN, source, author="a", message="v2")

        past_grace(clock)
        collector.run(enforce=True)

        ledger.repo.undo(env, second.op_sequence, principal="operator")
        assert commits.resolve(env, MAIN) == first.commit

        from src.format.model import Commit
        from src.fs.tree import resolve_path

        tree = ledger.store.get_as(first.commit, Commit).tree
        assert resolve_path(ledger.store, tree, "task/prompt.md") is not None

    def test_an_old_version_survives_because_the_op_log_names_it(
        self,
        ledger: Ledger,
        collector: GarbageCollector,
        commits: CommitService,
        env: EnvId,
        source: Path,
        clock: ManualClock,
    ) -> None:
        """Commits named in the retained log are GC roots, because
        undo restores a ref to an old commit and cannot resurrect objects
        collection has already reclaimed.
        """
        first = commits.commit(env, MAIN, source, author="a", message="v1")
        (source / "task" / "prompt.md").write_text("v2\n")
        commits.commit(env, MAIN, source, author="a", message="v2")

        past_grace(clock)
        collector.run(enforce=True)

        from src.format.model import Commit

        assert ledger.store.get_as(first.commit, Commit).message == "v1"

    def test_history_survives_the_operation_log_ageing_out(
        self,
        ledger: Ledger,
        collector: GarbageCollector,
        commits: CommitService,
        env: EnvId,
        source: Path,
        clock: ManualClock,
    ) -> None:
        """The op log is *one* root, not the only one keeping history alive.

        The test above holds only while the log still names the old commit. Once
        it ages out, the ref is the only root left — and a rebuild that walked
        ref *tips* rather than ref *history* would drop every earlier version of
        every environment, and the next sweep would take them.

        That is the difference between a rebuild and maintenance. Maintenance
        only ever adds, so a running environment accumulates all of its
        ancestors; a rebuild has to arrive at the same answer. Getting it wrong
        breaks ``log``, ``diff`` and restoring an earlier version — silently, and
        only on environments old enough that nobody is watching.
        """
        first = commits.commit(env, MAIN, source, author="a", message="v1")
        (source / "task" / "prompt.md").write_text("v2\n")
        second = commits.commit(env, MAIN, source, author="a", message="v2")

        # Past the grace period *and* past log retention, so the operation log
        # protects nothing and the ref is the only root that remains.
        past_grace(clock)
        clock.advance_seconds(400)
        collector.rebuild_keep_set("proximal/demo")
        report = collector.run(enforce=True)

        assert report.deleted == 0, "an environment's own history is reachable from its ref"

        assert [entry.commit.message for entry in commits.log(env, MAIN)] == ["v2", "v1"]
        commits.revert(env, MAIN, first.commit, author="a")
        assert commits.resolve(env, MAIN) == first.commit
        assert second.commit != first.commit

    def test_a_shared_ancestor_is_walked_once(
        self,
        ledger: Ledger,
        collector: GarbageCollector,
        commits: CommitService,
        env: EnvId,
        source: Path,
    ) -> None:
        """Two refs over one history must not cost two walks of it.

        Branches share all but a handful of commits, so a per-ref walk with no
        shared memo would make a rebuild quadratic in the number of branches —
        which is how a correctness fix becomes an outage on the environment with
        the most branches.
        """
        del collector
        base = commits.commit(env, MAIN, source, author="a", message="base")
        (source / "task" / "prompt.md").write_text("v2\n")
        commits.commit(env, MAIN, source, author="a", message="v2")
        # A branch off the first version: two refs whose histories overlap.
        ledger.repo.create_ref(env, BRANCH, base.commit, principal="a")

        # ``ObjectStore`` is deliberately final — there must be one place
        # verification happens — so counting its calls is a proxy and a cast
        # rather than a subclass, exactly as ``demo/e2e.py`` does it.
        watched = _CountingCommitReads(ledger.store)
        GarbageCollector(
            cast("Any", watched),
            ledger.keepsets,
            ledger.repo,
            clock=ledger.clock,
            digests=ledger.digests,
            config=FAST_GC,
        ).rebuild_keep_set("proximal/demo")

        assert len(watched.commits) == len(set(watched.commits)), (
            f"a commit was decoded more than once during one rebuild: {watched.commits}"
        )


class TestDiscardingABranchFreesItsStorage:
    """Branches get discarded, and discarding one frees whatever
    storage it used — *exactly* what it alone held.
    """

    def test_exclusive_content_is_reclaimed_and_shared_content_is_not(
        self,
        ledger: Ledger,
        collector: GarbageCollector,
        commits: CommitService,
        env: EnvId,
        source: Path,
        clock: ManualClock,
    ) -> None:
        base = commits.commit(env, MAIN, source, author="a", message="base")
        ledger.repo.create_ref(env, BRANCH, base.commit, principal="a")

        # The branch adds content nothing else holds.
        exclusive = random.Random(99).randbytes(80_000)
        (source / "data" / "branch-only.bin").write_bytes(exclusive)
        commits.commit(env, BRANCH, source, author="a", message="branch work")

        before_objects, before_bytes = ledger.store.catalog.total()

        current = ledger.repo.get_ref(env, BRANCH)
        ledger.repo.delete_ref(env, BRANCH, expected_generation=current.generation, principal="a")

        # The DeleteRef entry still *names* the branch's commit, and the rule
        # makes retained-log commits GC roots — so nothing is collectable yet.
        # That is not a bug: it is exactly the guarantee that undo still works.
        collector.rebuild_keep_set("proximal/demo")
        past_grace(clock)
        assert collector.run(enforce=True).deleted == 0

        # Past log retention, undo can no longer reach it, and only now does the
        # storage come back.
        clock.advance_seconds(400)
        collector.rebuild_keep_set("proximal/demo")
        report = collector.run(enforce=True)

        assert report.deleted > 0, "the branch's exclusive content should be reclaimed"
        assert report.bytes_freed > 50_000

        after_objects, after_bytes = ledger.store.catalog.total()
        assert after_objects < before_objects
        assert after_bytes < before_bytes

        # main is untouched and still fully readable.
        from src.format.model import Commit
        from src.fs.blob import BlobReader
        from src.fs.tree import resolve_path

        tree = ledger.store.get_as(commits.resolve(env, MAIN), Commit).tree
        dataset = resolve_path(ledger.store, tree, "data/train.bin")
        assert len(BlobReader(ledger.store, dataset.target).read()) == 120_000

    def test_content_shared_with_another_environment_is_kept(
        self,
        ledger: Ledger,
        collector: GarbageCollector,
        commits: CommitService,
        env: EnvId,
        source: Path,
        clock: ManualClock,
    ) -> None:
        """Global deduplication means an object belongs to no one environment.

        Deleting a ref in one must not take content an unrelated environment
        reaches — the failure mode reference counting gets wrong.
        """
        commits.commit(env, MAIN, source, author="a", message="v1")

        other = ledger.repo.create_env(EnvName("proximal/unrelated")).env_id
        commits.commit(other, MAIN, source, author="a", message="same content")

        current = ledger.repo.get_ref(env, MAIN)
        ledger.repo.delete_ref(env, MAIN, expected_generation=current.generation, principal="a")

        clock.advance_seconds(400)  # past both the grace period and log retention
        collector.rebuild_keep_set("proximal/demo")
        collector.run(enforce=True)

        from src.format.model import Commit
        from src.fs.blob import BlobReader
        from src.fs.tree import resolve_path

        tree = ledger.store.get_as(commits.resolve(other, MAIN), Commit).tree
        dataset = resolve_path(ledger.store, tree, "data/train.bin")
        assert len(BlobReader(ledger.store, dataset.target).read()) == 120_000

    def test_deleting_the_ref_is_instant(
        self, ledger: Ledger, commits: CommitService, env: EnvId, source: Path
    ) -> None:
        """Deleting the ref is instant and synchronous — no
        automation ever waits on collection.
        """
        base = commits.commit(env, MAIN, source, author="a", message="base")
        ledger.repo.create_ref(env, BRANCH, base.commit, principal="a")

        current = ledger.repo.get_ref(env, BRANCH)
        ledger.repo.delete_ref(env, BRANCH, expected_generation=current.generation, principal="a")

        assert [r.name for r in ledger.repo.list_refs(env)] == [MAIN]


class TestGuards:
    def test_the_grace_period_protects_recent_writes(
        self, ledger: Ledger, collector: GarbageCollector, env: EnvId
    ) -> None:
        """Guard 2: content a slow or paused writer is still assembling."""
        orphan = ledger.store.put_object(Chunk(b"written just now, referenced by nothing"))

        report = collector.run(enforce=True)
        assert report.deleted == 0
        assert report.plan.protected[GuardName.GRACE_PERIOD.value] >= 1
        assert ledger.store.get(orphan.name)  # still there

    def test_an_open_write_lease_protects_its_objects(
        self, ledger: Ledger, collector: GarbageCollector, env: EnvId, clock: ManualClock
    ) -> None:
        """Guard 3: a session's objects are roots until it ends.

        This is what makes an abandoned write need no cleanup by anyone — the
        bytes are unreferenced content on a timer, not state to repair.
        """
        session = ledger.repo.begin_write(env, principal="agent-17", ttl_us=3600 * 1_000_000)
        uploaded = ledger.store.put_object(Chunk(b"uploaded but not yet committed"))
        ledger.repo.record_uploaded(env, session.session_id, [uploaded.name])

        past_grace(clock)
        report = collector.run(enforce=True)

        assert report.deleted == 0
        assert report.plan.protected[GuardName.WRITE_LEASE.value] >= 1
        assert ledger.store.get(uploaded.name)

    def test_an_expired_lease_stops_protecting(
        self, ledger: Ledger, collector: GarbageCollector, env: EnvId, clock: ManualClock
    ) -> None:
        session = ledger.repo.begin_write(env, principal="agent-17", ttl_us=60 * 1_000_000)
        uploaded = ledger.store.put_object(Chunk(b"abandoned upload"))
        ledger.repo.record_uploaded(env, session.session_id, [uploaded.name])

        clock.advance_seconds(3600)
        report = collector.run(enforce=True)

        assert report.deleted >= 1
        with pytest.raises(ObjectNotFound):
            ledger.store.get(uploaded.name)

    def test_the_circuit_breaker_aborts_an_implausible_cycle(
        self, ledger: Ledger, clock: ManualClock, env: EnvId
    ) -> None:
        """Guard 6: refuse a cycle proposing to delete a large share of the corpus."""
        for i in range(1200):
            ledger.store.put_object(Chunk(f"orphan {i}".encode()))

        breaker = GarbageCollector(
            ledger.store,
            ledger.keepsets,
            ledger.repo,
            clock=clock,
            digests=ledger.digests,
            config=GcConfig(grace_us=60 * 1_000_000, shard_bits=2, min_corpus_objects=100),
        )
        past_grace(clock)
        report = breaker.run(enforce=True)

        assert report.aborted
        assert report.plan.aborted_by is GuardName.CIRCUIT_BREAKER
        assert report.deleted == 0
        assert "refusing" in report.plan.abort_reason

    def test_the_breaker_has_a_floor_so_it_does_not_cry_wolf(
        self, ledger: Ledger, collector: GarbageCollector, clock: ManualClock, env: EnvId
    ) -> None:
        """Without ``min_corpus_objects`` the breaker fires on every small
        corpus — including every integration test — and the first thing anyone
        does with a breaker that cries wolf is turn it off.
        """
        ledger.store.put_object(Chunk(b"one lonely orphan"))
        past_grace(clock)

        report = collector.run(enforce=True)
        assert not report.aborted
        assert report.deleted == 1


class TestResurrection:
    """**The subtlest bug in the whole system**.

    Collection does not delete an object and its parent at the same instant:

        1. tree T becomes unreachable; its exclusive chunk C is standalone
        2. sweep deletes C                    ── C was genuinely garbage
        3. T survives a while longer          ── dead, but still present
        4. a new commit contains an identical T
           HasObjects(T) → "present"          ── the client uploads nothing
        5. the commit is published
        6. the new commit reaches C, which no longer exists

    No other guard catches it: the cutoff, the grace period and leases all
    protect *recently written* content, and C was legitimately garbage when it
    was swept. The failure is readable-but-broken — Resolve, ListDir, Diff and
    Log all succeed, and the rollout fails half an hour in.
    """

    def test_a_swept_object_is_reported_missing_so_the_client_re_uploads(
        self, ledger: Ledger, collector: GarbageCollector, clock: ManualClock, env: EnvId
    ) -> None:
        """The fix: deletion is made visible to deduplication."""
        chunk = Chunk(b"exclusive content that will be swept")
        name = ledger.store.put_object(chunk).name

        past_grace(clock)
        report = collector.run(enforce=True)
        assert report.deleted >= 1
        assert report.tombstones_written >= 1

        # The client asks whether it must upload. Even though the *bytes* may
        # linger in a pack, the answer must be "yes".
        assert ledger.store.missing([name]) == frozenset({name})

    def test_re_uploading_a_swept_object_clears_its_tombstone(
        self, ledger: Ledger, collector: GarbageCollector, clock: ManualClock, env: EnvId
    ) -> None:
        """Without this the store deadlocks: the object stays permanently
        'missing', so every write uploads it again and never converges.
        """
        chunk = Chunk(b"swept, then written again")
        name = ledger.store.put_object(chunk).name

        past_grace(clock)
        collector.run(enforce=True)
        assert ledger.store.missing([name]) == frozenset({name})

        ledger.store.put_object(chunk)
        assert ledger.store.missing([name]) == frozenset()
        assert ledger.store.get(name)

    def test_without_tombstones_the_bug_reappears(self, tmp_path: Path, clock: ManualClock) -> None:
        """**Proves the guard is load-bearing rather than decorative.**

        Same sequence with a no-op tombstone store: the swept object is reported
        *present*, so a client would upload nothing and publish a commit whose
        content is gone. This is the test that fails if anyone ever decides
        tombstones are an optimisation.
        """
        from src.store.backend import InMemoryBackend
        from src.store.cas import ObjectStore
        from src.store.catalog import InMemoryWriteCatalog

        backend = InMemoryBackend()
        unguarded = ObjectStore(
            backend,
            catalog=InMemoryWriteCatalog(),
            tombstones=NullTombstoneStore(),
            clock=clock,
        )
        chunk = Chunk(b"exclusive content")
        name = unguarded.put_object(chunk).name

        # Simulate the sweep deleting the bytes' *catalog* entry while the bytes
        # themselves linger — exactly step 3 of the sequence above.
        unguarded.catalog.forget([name])

        assert unguarded.missing([name]) == frozenset(), (
            "with no tombstone the store reports the swept object as present — "
            "a client would upload nothing and publish a broken commit"
        )

    def test_a_tombstone_expires(
        self, ledger: Ledger, collector: GarbageCollector, clock: ManualClock, env: EnvId
    ) -> None:
        """Tombstones live one collection period past the sweep — longer than any
        in-flight write session can hold a stale 'present' — then go.
        """
        name = ledger.store.put_object(Chunk(b"transient")).name
        past_grace(clock)
        collector.run(enforce=True)
        assert ledger.store.tombstones.count() >= 1

        clock.advance_days(3)
        assert ledger.store.tombstones.purge_expired(clock.now_us()) >= 1
        assert ledger.store.tombstones.count() == 0
        del name


class TestKeepSets:
    def test_a_commit_graduates_into_the_keep_set(
        self, ledger: Ledger, commits: CommitService, env: EnvId, source: Path
    ) -> None:
        assert ledger.keepsets.size(str(env)) == 0
        commits.commit(env, MAIN, source, author="a", message="v1")
        assert ledger.keepsets.size(str(env)) > 5

    def test_deduplicated_objects_are_recorded_too(
        self, ledger: Ledger, commits: CommitService, env: EnvId, source: Path
    ) -> None:
        """A hash the client offered and was told it already had is still content
        this commit reaches. Omitting it would let a sweep between the offer and
        the commit take content the new commit depends on.
        """
        commits.commit(env, MAIN, source, author="a", message="v1")
        other = ledger.repo.create_env(EnvName("proximal/second")).env_id

        # Every object already exists, so this commit uploads almost nothing —
        # and must still record the full closure.
        commits.commit(other, MAIN, source, author="a", message="identical")
        assert ledger.keepsets.size(str(other)) > 5

    def test_rebuilding_recovers_the_same_set(
        self,
        ledger: Ledger,
        collector: GarbageCollector,
        commits: CommitService,
        env: EnvId,
        source: Path,
    ) -> None:
        """The maintained set and the recomputed one must agree, or maintenance
        is silently drifting from truth.
        """
        commits.commit(env, MAIN, source, author="a", message="v1")
        maintained = ledger.keepsets.size(str(env))

        collector.rebuild_keep_set("proximal/demo")
        assert ledger.keepsets.size(str(env)) == maintained

    def test_a_cycle_never_holds_more_than_one_shard_of_the_live_set(
        self,
        ledger: Ledger,
        clock: ManualClock,
        commits: CommitService,
        env: EnvId,
        source: Path,
    ) -> None:
        """The diff is sharded so memory is O(shard), not O(corpus).

        Sharding both sides by the same hash prefix is what makes the comparison
        exact without a Bloom filter — but the saving is only real if the live
        side is read one shard at a time. Reading it whole gives the identical
        answer, makes the identical calls, and costs the whole corpus in memory,
        so nothing about the *result* can tell the two apart. The access pattern
        can: the two sides have to advance together, shard by shard.

        At the capacity plan's year-10 numbers the difference is tens of gigabytes held in one
        process, in the one component whose job is to run unattended.
        """
        commits.commit(env, MAIN, source, author="a", message="v1")

        trace = _Trace()
        ledger.store._catalog = trace.catalog(ledger.store.catalog)  # type: ignore[assignment]
        plan = GarbageCollector(
            ledger.store,
            trace.keepsets(ledger.keepsets),
            ledger.repo,
            clock=clock,
            digests=ledger.digests,
            config=FAST_GC,
        ).plan()

        shards = 1 << FAST_GC.shard_bits
        assert trace.shape == ["live", "stored"] * shards, (
            f"the two sides must be walked shard by shard, together; got {trace.events}"
        )
        assert [shard for _, shard in trace.events] == [s for s in range(shards) for _ in range(2)]
        # The counts still have to be right: sharding is how the total is
        # computed, and a total that drifts is worse than a large one.
        assert plan.live_objects == ledger.keepsets.size()


class TestReporting:
    def test_a_plan_reports_what_it_would_do(
        self,
        ledger: Ledger,
        collector: GarbageCollector,
        commits: CommitService,
        env: EnvId,
        source: Path,
        clock: ManualClock,
    ) -> None:
        commits.commit(env, MAIN, source, author="a", message="v1")
        ledger.store.put_object(Chunk(b"an orphan"))
        past_grace(clock)

        plan = collector.plan()
        assert plan.corpus_objects > 0
        assert plan.live_objects > 0
        assert len(plan.candidates) == 1
        assert plan.bytes_reclaimable > 0

    def test_delivery_verification_still_holds_after_a_sweep(
        self,
        ledger: Ledger,
        collector: GarbageCollector,
        commits: CommitService,
        env: EnvId,
        source: Path,
        clock: ManualClock,
    ) -> None:
        """A sweep must not leave anything that fails verification on read —
        one of the two signals that must never be silent.
        """
        head = commits.commit(env, MAIN, source, author="a", message="v1")
        ledger.store.put_object(Chunk(b"an orphan"))
        past_grace(clock)
        collector.run(enforce=True)

        for name in commit_closure(ledger.store, head.commit):
            try:
                ledger.store.get(name)
            except CorruptObject:  # pragma: no cover - would be a real defect
                pytest.fail(f"{name} failed verification after a sweep")
