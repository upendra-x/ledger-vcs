"""The build and sync pipeline.

The headline is one assertion, and it is the reason build results are keyed by
commit at all: **fork an environment, trigger a build, and the runner is not
invoked**. A fork that changed nothing inherits its parent's result, because a
build is a pure function of a commit and the two environments share the commit.
Get that wrong and forking costs a full rebuild of a forty-gigabyte environment
to discover it is identical.

Everything else here is the machinery that makes that safe to rely on:

* events come from the outbox the ref update wrote **in its own transaction**,
  so a build can never describe a commit that was not published;
* delivery is at-least-once and the effect is exactly-once, because both the
  queue entry and the platform sync are keyed by things that do not change on a
  retry;
* one environment at a time, in publish order, and never across environments —
  the same isolation the write path has, obtained the same way;
* a failed build records a failure and lets go. It never blocks or reverses the
  commit that triggered it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from src.build.manifest import parse_manifest
from src.build.models import BuildStatus
from src.build.pipeline import SYNC_NAMESPACE, BuildWorker, Dispatcher, trigger
from src.build.queue import MAX_ATTEMPTS, BuildQueue
from src.build.results import BuildResults
from src.build.runner import RecordingRunner, RunOutcome, SubprocessRunner
from src.build.sync import RecordingPlatform
from src.clock import ManualClock
from src.errors import InvalidRequest
from src.format.cdc import ChunkParams
from src.format.shape import ShapeParams
from src.ids import EnvName, RefName
from src.instance import Ledger
from src.service.commits import CommitService
from src.service.environments import EnvironmentService

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from src.build.runner import RunRequest
    from src.ids import EnvId, ObjectName

ENV = "proximal/demo"
MAIN = RefName("refs/heads/main")

MANIFEST = b"""
name: demo-environment
version: 1
build:
  command: ["/bin/sh", "-c", "echo built $LEDGER_COMMIT > out.txt"]
  timeout_seconds: 30
images:
  - app
sync:
  team: rl-infra
"""


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


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
def env_id(ledger: Ledger) -> EnvId:
    return ledger.repo.create_env(EnvName(ENV)).env_id


@pytest.fixture
def source(tmp_path: Path) -> Path:
    root = tmp_path / "src"
    (root / "task").mkdir(parents=True)
    (root / "harbor.yaml").write_bytes(MANIFEST)
    (root / "task" / "prompt.md").write_text("solve it\n")
    return root


@pytest.fixture
def runner() -> RecordingRunner:
    return RecordingRunner()


@pytest.fixture
def platform() -> RecordingPlatform:
    return RecordingPlatform()


@pytest.fixture
def queue(ledger: Ledger) -> BuildQueue:
    return BuildQueue(ledger.meta, clock=ledger.clock)


@pytest.fixture
def worker(
    ledger: Ledger, runner: RecordingRunner, platform: RecordingPlatform, queue: BuildQueue
) -> BuildWorker:
    return BuildWorker(ledger, runner=runner, platform=platform, queue=queue)


@pytest.fixture
def dispatcher(ledger: Ledger, queue: BuildQueue) -> Dispatcher:
    return Dispatcher(ledger, queue)


def commit(ledger: Ledger, env_id: EnvId, source: Path, message: str = "v1") -> ObjectName:
    return (
        CommitService(ledger)
        .commit(env_id, MAIN, source, author="agent-17", message=message)
        .commit
    )


# ─────────────────────────────────────────────────────────────────────────────
# The manifest — the only thing in Ledger that parses one
# ─────────────────────────────────────────────────────────────────────────────


class TestManifest:
    def test_reads_a_build_step(self) -> None:
        manifest = parse_manifest(MANIFEST)
        assert manifest.name == "demo-environment"
        assert manifest.builds
        assert manifest.build is not None
        assert manifest.build.command[0] == "/bin/sh"
        assert manifest.images == ("app",)
        assert manifest.sync_metadata == {"team": "rl-infra"}

    def test_an_environment_with_no_manifest_is_not_an_error(self) -> None:
        """Most environments do not build. That is a state, not a failure."""
        assert parse_manifest(b"").builds is False

    def test_a_string_command_runs_through_a_shell(self) -> None:
        manifest = parse_manifest(b"build:\n  command: make all\n")
        assert manifest.build is not None
        assert manifest.build.command == ("/bin/sh", "-c", "make all")

    def test_a_timeout_cannot_exceed_the_ceiling(self) -> None:
        """One environment must not be able to hold a worker indefinitely."""
        from src.build.manifest import MAX_TIMEOUT_SECONDS

        manifest = parse_manifest(b"build:\n  command: [true]\n  timeout_seconds: 999999\n")
        assert manifest.build is not None
        assert manifest.build.timeout_seconds == MAX_TIMEOUT_SECONDS

    def test_malformed_yaml_names_itself(self) -> None:
        with pytest.raises(InvalidRequest, match="not valid YAML"):
            parse_manifest(b"build: [unclosed\n")

    def test_a_build_step_without_a_command_is_refused(self) -> None:
        with pytest.raises(InvalidRequest, match="command"):
            parse_manifest(b"build:\n  timeout_seconds: 10\n")

    def test_the_summary_carries_no_command_or_environment(self) -> None:
        """A build result is world-readable within the corpus, and a manifest's
        environment map is exactly where someone will eventually put a token.
        """
        from src.build.manifest import describe

        summary = describe(parse_manifest(b"build:\n  command: [x]\n  env:\n    TOKEN: hunter2\n"))
        assert "hunter2" not in str(summary)
        assert "command" not in summary


# ─────────────────────────────────────────────────────────────────────────────
# Dispatch
# ─────────────────────────────────────────────────────────────────────────────


class TestDispatch:
    def test_a_commit_queues_a_build(
        self, ledger: Ledger, env_id: EnvId, source: Path, dispatcher: Dispatcher, queue: BuildQueue
    ) -> None:
        """The event was written inside the transaction that published the commit,
        so there is no window in which the two disagree.
        """
        commit(ledger, env_id, source)
        report = dispatcher.poll()

        assert report.enqueued == 1
        pending = queue.pending(env_id)
        assert len(pending) == 1
        assert pending[0].ref == str(MAIN)

    def test_polling_twice_queues_the_work_once(
        self, ledger: Ledger, env_id: EnvId, source: Path, dispatcher: Dispatcher, queue: BuildQueue
    ) -> None:
        """At-least-once delivery, exactly-once effect.

        Not by deduplicating: the entry is keyed on the operation that triggered
        it, so a second delivery has nowhere new to write.
        """
        commit(ledger, env_id, source)
        dispatcher.poll()
        second = dispatcher.poll()

        assert second.enqueued == 0
        assert queue.depth() == 1

    def test_the_cursor_survives_a_restart(
        self, ledger: Ledger, env_id: EnvId, source: Path, queue: BuildQueue
    ) -> None:
        commit(ledger, env_id, source)
        Dispatcher(ledger, queue).poll()

        fresh = Dispatcher(ledger, queue)  # as if the process had restarted
        assert fresh.poll().events == 0

    def test_a_non_default_branch_does_not_build(
        self, ledger: Ledger, env_id: EnvId, source: Path, dispatcher: Dispatcher, queue: BuildQueue
    ) -> None:
        """A build is triggered by a ref *configured* to build."""
        head = commit(ledger, env_id, source)
        dispatcher.poll()
        depth = queue.depth()

        ledger.repo.create_ref(env_id, RefName("refs/heads/wip"), head, principal="agent-17")
        report = dispatcher.poll()

        assert report.ignored >= 1
        assert queue.depth() == depth

    def test_events_from_every_shard_are_delivered(self, ledger: Ledger, queue: BuildQueue) -> None:
        """The bug a single-integer cursor would cause, made impossible.

        Each shard numbers its own events from one. A consumer that collapsed
        them into one scalar would advance past the quieter shards and stop
        delivering their events entirely — and the environments living there
        would simply never build again, with nothing reporting an error.
        """
        from pathlib import Path as _Path

        dispatcher = Dispatcher(ledger, queue)
        envs = []
        for index in range(8):
            env = ledger.repo.create_env(EnvName(f"proximal/e{index}")).env_id
            root = _Path(ledger.root) / f"src{index}"
            root.mkdir()
            (root / "file.txt").write_text(f"env {index}\n")
            commit(ledger, env, root)
            envs.append(env)

        # More than one shard must actually be in play, or this proves nothing.
        shards = {ledger.meta.shard_of(str(env)) for env in envs}
        assert len(shards) > 1, "the fixture did not spread environments across shards"

        dispatcher.poll()
        assert queue.depth() == len(envs), "an entire shard's events went undelivered"


# ─────────────────────────────────────────────────────────────────────────────
# Building
# ─────────────────────────────────────────────────────────────────────────────


class TestBuild:
    def test_commit_then_build_then_sync_note(
        self,
        ledger: Ledger,
        env_id: EnvId,
        source: Path,
        dispatcher: Dispatcher,
        worker: BuildWorker,
        runner: RecordingRunner,
        platform: RecordingPlatform,
    ) -> None:
        """The whole loop, in order."""
        head = commit(ledger, env_id, source)
        dispatcher.poll()
        outcome = worker.run_once()

        assert outcome is not None
        assert outcome.result.status is BuildStatus.SUCCEEDED
        assert not outcome.cache_hit
        assert runner.count == 1

        # The platform was told about it, under a key derived from the commit.
        assert platform.entry_for(str(env_id), str(head)) is not None

        # And the environment records which platform entry this version became.
        note = ledger.repo.get_note(env_id, head, SYNC_NAMESPACE)
        assert note.body["commit"] == str(head)
        assert note.body["platform_id"]

    def test_the_runner_sees_the_materialized_environment(
        self,
        ledger: Ledger,
        env_id: EnvId,
        source: Path,
        dispatcher: Dispatcher,
        worker: BuildWorker,
        runner: RecordingRunner,
    ) -> None:
        """Byte-identical inputs, including anything the commit pinned."""
        commit(ledger, env_id, source)
        dispatcher.poll()
        worker.run_once()

        request = runner.invocations[0]
        assert request.manifest.name == "demo-environment"
        assert request.env_name == ENV

    def test_the_work_is_gone_when_the_build_finishes(
        self,
        ledger: Ledger,
        env_id: EnvId,
        source: Path,
        dispatcher: Dispatcher,
        worker: BuildWorker,
        queue: BuildQueue,
    ) -> None:
        commit(ledger, env_id, source)
        dispatcher.poll()
        worker.run_once()

        assert queue.depth() == 0
        assert worker.run_once() is None

    def test_a_real_command_runs_in_the_checkout(
        self,
        ledger: Ledger,
        env_id: EnvId,
        source: Path,
        dispatcher: Dispatcher,
        platform: RecordingPlatform,
    ) -> None:
        """The subprocess runner against a real materialized commit.

        This is the one test that proves the workspace is a genuine directory
        containing the version's own bytes, rather than a description of one.
        """
        commit(ledger, env_id, source)
        dispatcher.poll()
        outcome = BuildWorker(ledger, runner=SubprocessRunner(), platform=platform).run_once()

        assert outcome is not None
        assert outcome.result.status is BuildStatus.SUCCEEDED
        assert outcome.result.exit_code == 0


# ─────────────────────────────────────────────────────────────────────────────
# a build is a pure function of a commit
# ─────────────────────────────────────────────────────────────────────────────


class TestBuildIsAPureFunctionOfACommit:
    def test_a_fork_inherits_its_parents_build_without_rebuilding(
        self,
        ledger: Ledger,
        env_id: EnvId,
        source: Path,
        dispatcher: Dispatcher,
        worker: BuildWorker,
        runner: RecordingRunner,
        platform: RecordingPlatform,
    ) -> None:
        """**The whole contract, in five lines.**

        Forking copies no bytes, and this is the other half of what that has to
        mean: it copies no *work* either. The fork and its parent share the
        commit, the build result is keyed by the commit, so the fork's build is
        already done before it is asked for.

        The assertion that matters is the invocation count. A pipeline that
        rebuilt here would still look correct — same result, same note — and
        would cost a full rebuild of every forked environment in the corpus.
        """
        head = commit(ledger, env_id, source)
        dispatcher.poll()
        worker.run_once()
        assert runner.count == 1

        forked = (
            EnvironmentService(ledger)
            .fork(env_id, EnvName("proximal/fork"), from_ref=MAIN, principal="agent-17")
            .environment.env_id
        )
        dispatcher.poll()
        outcome = worker.run_once()

        assert outcome is not None
        assert outcome.cache_hit, "the fork rebuilt a commit that was already built"
        assert runner.count == 1, "forking cost a rebuild"
        assert outcome.result.commit == str(head)

        # It is a *fork*, not a copy: it syncs to its own platform entry, from a
        # build it never ran.
        assert platform.entry_for(str(forked), str(head)) is not None
        assert ledger.repo.get_note(forked, head, SYNC_NAMESPACE).body["commit"] == str(head)

    def test_two_refs_at_one_commit_build_once(
        self,
        ledger: Ledger,
        env_id: EnvId,
        source: Path,
        dispatcher: Dispatcher,
        worker: BuildWorker,
        runner: RecordingRunner,
    ) -> None:
        head = commit(ledger, env_id, source)
        dispatcher.poll()
        worker.run_once()

        # Move the default ref away and back — a different operation, the same
        # commit. A target-compare cache would be fooled; a commit-keyed one is not.
        second = commit(ledger, env_id, source, message="v2")
        assert second != head
        CommitService(ledger).revert(env_id, MAIN, head, author="agent-17")
        dispatcher.poll()
        worker.drain()

        assert runner.count == 2, "the same commit was built more than once"

    def test_retriggering_an_unchanged_commit_returns_the_existing_result(
        self,
        ledger: Ledger,
        env_id: EnvId,
        source: Path,
        dispatcher: Dispatcher,
        worker: BuildWorker,
        runner: RecordingRunner,
    ) -> None:
        commit(ledger, env_id, source)
        dispatcher.poll()
        worker.run_once()

        trigger(ledger, env_id, MAIN, queue=worker.queue)
        outcome = worker.run_once()

        assert outcome is not None
        assert outcome.cache_hit
        assert runner.count == 1

    def test_asking_for_a_rebuild_forgets_the_result_first(
        self,
        ledger: Ledger,
        env_id: EnvId,
        source: Path,
        dispatcher: Dispatcher,
        worker: BuildWorker,
        runner: RecordingRunner,
    ) -> None:
        """ "Build it again" means the previous *answer* is no longer trusted, and
        that has to be said where every consumer sees it.
        """
        commit(ledger, env_id, source)
        dispatcher.poll()
        worker.run_once()

        trigger(ledger, env_id, MAIN, queue=worker.queue, rebuild=True)
        outcome = worker.run_once()

        assert outcome is not None
        assert not outcome.cache_hit
        assert runner.count == 2


# ─────────────────────────────────────────────────────────────────────────────
# Ordering and exclusivity
# ─────────────────────────────────────────────────────────────────────────────


class TestOrderingAndExclusivity:
    def test_one_environment_builds_in_publish_order(
        self,
        ledger: Ledger,
        env_id: EnvId,
        source: Path,
        dispatcher: Dispatcher,
        worker: BuildWorker,
    ) -> None:
        """What stops version 49 from syncing after version 50.

        The queue's sort key *is* the operation sequence, so there is no second
        notion of order that could disagree with the first.
        """
        first = commit(ledger, env_id, source, message="v1")
        (source / "task" / "prompt.md").write_text("solve it differently\n")
        second = commit(ledger, env_id, source, message="v2")
        dispatcher.poll()

        built = [outcome.commit for outcome in worker.drain()]
        assert built == [str(first), str(second)]

    def test_a_second_worker_cannot_take_a_leased_environment(
        self, ledger: Ledger, env_id: EnvId, source: Path, dispatcher: Dispatcher, queue: BuildQueue
    ) -> None:
        """Exclusivity is one key, not a lock manager."""
        commit(ledger, env_id, source)
        dispatcher.poll()

        held = queue.lease(worker="worker-a")
        assert held is not None
        assert queue.lease(worker="worker-b") is None

    def test_two_environments_never_block_each_other(
        self, ledger: Ledger, source: Path, dispatcher: Dispatcher, queue: BuildQueue
    ) -> None:
        """Partition isolation in the build layer: isolation from partitioning."""
        first = ledger.repo.create_env(EnvName("proximal/one")).env_id
        second = ledger.repo.create_env(EnvName("proximal/two")).env_id
        commit(ledger, first, source)
        commit(ledger, second, source)
        dispatcher.poll()

        a = queue.lease(worker="worker-a")
        b = queue.lease(worker="worker-b")
        assert a is not None
        assert b is not None
        assert a.env_id != b.env_id

    def test_an_expired_lease_is_taken_over(
        self,
        ledger: Ledger,
        clock: ManualClock,
        env_id: EnvId,
        source: Path,
        dispatcher: Dispatcher,
        queue: BuildQueue,
    ) -> None:
        """A worker that dies must not stop an environment building forever."""
        from src.build.queue import DEFAULT_LEASE_US

        commit(ledger, env_id, source)
        dispatcher.poll()
        assert queue.lease(worker="doomed") is not None

        clock.advance_us(DEFAULT_LEASE_US + 1)
        assert queue.lease(worker="successor") is not None

    def test_a_worker_that_lost_its_lease_cannot_finish(
        self,
        ledger: Ledger,
        clock: ManualClock,
        env_id: EnvId,
        source: Path,
        dispatcher: Dispatcher,
        queue: BuildQueue,
    ) -> None:
        """Every completion write is conditioned on the lease still being ours.

        Without that, a worker that paused past its lease would come back and
        delete work another worker is actively doing.
        """
        from src.build.queue import DEFAULT_LEASE_US
        from src.errors import Conflict

        commit(ledger, env_id, source)
        dispatcher.poll()
        stale = queue.lease(worker="paused")
        assert stale is not None

        clock.advance_us(DEFAULT_LEASE_US + 1)
        assert queue.lease(worker="successor") is not None

        with pytest.raises(Conflict):
            queue.complete(stale)

    def test_a_lease_outlives_the_longest_build_a_manifest_may_declare(self) -> None:
        """Otherwise a slow build is a livelock, not a slow build.

        The takeover above is correct for a worker that *died*. Applied to one
        that is merely still working, it is a loop with no exit: the lease is
        taken over mid-build, the original worker's ``complete`` fails its
        version check, the work returns to the queue, and the next worker takes
        exactly as long and loses it at exactly the same point. Nothing
        escalates — that path never reaches ``_give_up``, so ``attempts`` never
        rises and ``MAX_ATTEMPTS`` never fires.

        A manifest may declare up to an hour; the lease was fifteen minutes. So
        every build over fifteen minutes span forever, and every build under it
        was fine — which is why nothing here caught it: the suite's builds take
        milliseconds and the typical build is five minutes.

        Asserted as a relationship between the two constants rather than a
        literal, because the bug is that they were chosen independently.
        """
        from src.build.manifest import MAX_TIMEOUT_SECONDS
        from src.build.queue import DEFAULT_LEASE_US

        assert DEFAULT_LEASE_US > MAX_TIMEOUT_SECONDS * 1_000_000, (
            "a build may run longer than its worker owns the environment; "
            "the takeover will loop forever"
        )

    def test_a_build_that_runs_the_longest_permitted_time_still_completes(
        self,
        ledger: Ledger,
        clock: ManualClock,
        env_id: EnvId,
        source: Path,
        dispatcher: Dispatcher,
        queue: BuildQueue,
    ) -> None:
        """The same thing as behaviour, so the constants cannot drift apart quietly.

        Time moves by the maximum a manifest may ask for, and the worker that
        started the build must still be the one allowed to finish it.
        """
        from src.build.manifest import MAX_TIMEOUT_SECONDS

        commit(ledger, env_id, source)
        dispatcher.poll()
        lease = queue.lease(worker="slow-but-alive")
        assert lease is not None

        clock.advance_us(MAX_TIMEOUT_SECONDS * 1_000_000)

        assert queue.lease(worker="impatient") is None, "the environment was stolen mid-build"
        queue.complete(lease)


# ─────────────────────────────────────────────────────────────────────────────
# Failure
# ─────────────────────────────────────────────────────────────────────────────


class _FailingRunner:
    """A runner whose builds do not work. Counts attempts."""

    def __init__(self) -> None:
        self.count = 0

    def run(self, request: RunRequest) -> RunOutcome:
        del request
        self.count += 1
        return RunOutcome(exit_code=1, output="build.sh: line 3: boom")


class TestFailure:
    def test_a_failed_build_never_touches_the_commit(
        self,
        ledger: Ledger,
        env_id: EnvId,
        source: Path,
        dispatcher: Dispatcher,
        platform: RecordingPlatform,
    ) -> None:
        """The claim, stated as strongly as it deserves.

        The commit is already published — that decision was made by the ref
        update, and a build has no vote. A pipeline that could reverse it would
        make publishing conditional on a build succeeding, which is exactly the
        coupling this removes.
        """
        head = commit(ledger, env_id, source)
        dispatcher.poll()
        outcome = BuildWorker(ledger, runner=_FailingRunner(), platform=platform).run_once()

        assert outcome is not None
        assert outcome.result.status is BuildStatus.FAILED
        assert ledger.repo.get_ref(env_id, MAIN).target == head

    def test_it_retries_and_then_gives_up(
        self,
        ledger: Ledger,
        env_id: EnvId,
        source: Path,
        dispatcher: Dispatcher,
        platform: RecordingPlatform,
        queue: BuildQueue,
    ) -> None:
        """Retries, then a recorded failure, then stop.

        The subtle part is *when* the failure becomes durable. Results are cached
        by commit, so a failure recorded on the first attempt would be found by
        the retry and returned as a cache hit — and the retries required
        for would silently never happen. The claim is released between attempts
        instead, and only made durable when there will be no further one.
        """
        commit(ledger, env_id, source)
        dispatcher.poll()
        runner = _FailingRunner()
        outcomes = BuildWorker(ledger, runner=runner, platform=platform, queue=queue).drain(
            limit=10
        )

        assert runner.count == MAX_ATTEMPTS, "the build did not retry"
        assert len(outcomes) == MAX_ATTEMPTS
        assert queue.depth() == 0, "a failing build held its environment"

        recorded = BuildResults(ledger.meta, clock=ledger.clock).get(
            ledger.repo.get_ref(env_id, MAIN).target
        )
        assert recorded is not None
        assert recorded.status is BuildStatus.FAILED
        assert recorded.attempts == MAX_ATTEMPTS

    def test_a_failure_is_not_synced(
        self,
        ledger: Ledger,
        env_id: EnvId,
        source: Path,
        dispatcher: Dispatcher,
        platform: RecordingPlatform,
    ) -> None:
        commit(ledger, env_id, source)
        dispatcher.poll()
        BuildWorker(ledger, runner=_FailingRunner(), platform=platform).run_once()

        assert platform.deliveries == []

    def test_failures_are_queryable_in_aggregate(
        self,
        ledger: Ledger,
        env_id: EnvId,
        source: Path,
        dispatcher: Dispatcher,
        platform: RecordingPlatform,
    ) -> None:
        """One signal rather than ten million silent ones.

        Only after the retries are exhausted, which is the point: a failure that
        became durable on the first attempt would be cached and returned to the
        retry, so nothing would ever be retried at all.
        """
        commit(ledger, env_id, source)
        dispatcher.poll()
        results = BuildResults(ledger.meta, clock=ledger.clock)
        worker = BuildWorker(ledger, runner=_FailingRunner(), platform=platform)

        worker.run_once()
        assert results.failures() == [], "a failure was recorded before retries ran out"

        worker.drain(limit=10)
        failures = results.failures()
        assert len(failures) == 1
        assert "boom" in failures[0].log_excerpt


# ─────────────────────────────────────────────────────────────────────────────
# Sync
# ─────────────────────────────────────────────────────────────────────────────


class TestSync:
    def test_a_duplicated_delivery_updates_rather_than_duplicates(
        self, platform: RecordingPlatform
    ) -> None:
        """The key is *derived* from the environment and the commit.

        A generated key would differ between the first delivery and its retry,
        which is precisely the case it exists to handle.
        """
        from src.build.sync import SyncRequest

        request = SyncRequest(commit="b3:aa", env_id="env_1", env_name=ENV)
        first = platform.sync(request, now_us=1)
        second = platform.sync(request, now_us=2)

        assert len(platform.deliveries) == 2
        assert len(platform.entries) == 1
        assert second.platform_id == first.platform_id
        assert second.deduplicated

    def test_the_note_lives_in_the_environment_not_the_commit(
        self,
        ledger: Ledger,
        env_id: EnvId,
        source: Path,
        dispatcher: Dispatcher,
        worker: BuildWorker,
    ) -> None:
        """Which platform entry a commit corresponds to is a fact about the
        *environment* — a fork syncs to a different one from the same bytes.
        """
        head = commit(ledger, env_id, source)
        dispatcher.poll()
        worker.run_once()

        assert ledger.repo.get_note(env_id, head, SYNC_NAMESPACE).body["platform_id"]
        # And the commit object itself is untouched: its name is the hash of its
        # content, so recording an outcome inside it would change its identity.
        assert ledger.store.get(head)


# ─────────────────────────────────────────────────────────────────────────────
# The HTTP surface — the Build row
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def app(ledger: Ledger) -> Any:
    from src.api.app import build_app

    return build_app(ledger=ledger, clock=ledger.clock)


@pytest.fixture
def state(app: Any) -> Any:
    return app.state.ledger_state


@pytest.fixture
def client(app: Any) -> Iterator[Any]:
    from fastapi.testclient import TestClient

    with TestClient(app) as opened:
        yield opened


def token_for(state: Any, *operations: Any, principal: str = "builder") -> str:
    from src.auth.model import NamePrefixSelector, Operation, Principal, Scope

    combined = Operation(0)
    for operation in operations:
        combined |= operation
    scope = Scope(operations=combined, selectors=(NamePrefixSelector("proximal/*"),))
    minted: str = state.signer.mint(Principal(principal), scope, ttl_us=3600 * 1_000_000)
    return minted


def bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


class TestBuildApi:
    def test_a_build_result_is_read_through_an_environment(
        self,
        ledger: Ledger,
        env_id: EnvId,
        source: Path,
        dispatcher: Dispatcher,
        worker: BuildWorker,
        client: Any,
        state: Any,
    ) -> None:
        """Keyed by commit globally, but reached through an environment.

        A commit hash alone must never be enough to read anything, and a build result is no
        exception — it names images and a
        worker, which is exactly the kind of thing that leaks.
        """
        from src.auth.model import Operation

        head = commit(ledger, env_id, source)
        dispatcher.poll()
        worker.run_once()

        response = client.get(
            f"/v1/envs/{ENV}/commits/{head}/build",
            headers=bearer(token_for(state, Operation.READ)),
        )
        assert response.status_code == 200, response.text
        assert response.json()["status"] == "succeeded"

    def test_a_commit_from_another_environment_is_refused(
        self,
        ledger: Ledger,
        env_id: EnvId,
        source: Path,
        dispatcher: Dispatcher,
        worker: BuildWorker,
        client: Any,
        state: Any,
    ) -> None:
        from src.auth.model import Operation

        head = commit(ledger, env_id, source)
        dispatcher.poll()
        worker.run_once()
        ledger.repo.create_env(EnvName("proximal/other"))

        response = client.get(
            f"/v1/envs/proximal/other/commits/{head}/build",
            headers=bearer(token_for(state, Operation.READ)),
        )
        assert response.status_code == 403

    def test_triggering_needs_build_authority(
        self, ledger: Ledger, env_id: EnvId, source: Path, client: Any, state: Any
    ) -> None:
        """``env:read`` does not let you spend a build worker's five minutes."""
        from src.auth.model import Operation

        commit(ledger, env_id, source)
        reader = client.post(
            f"/v1/envs/{ENV}/builds",
            json={"ref": str(MAIN)},
            headers=bearer(token_for(state, Operation.READ)),
        )
        assert reader.status_code == 403

        builder = client.post(
            f"/v1/envs/{ENV}/builds",
            json={"ref": str(MAIN)},
            headers=bearer(token_for(state, Operation.BUILD)),
        )
        assert builder.status_code == 202
        assert builder.json()["queued"] is True

    def test_the_aggregate_failure_query_is_administrative(self, client: Any, state: Any) -> None:
        """It deliberately crosses every environment, so no environment selector
        can scope it.
        """
        from src.auth.model import Operation

        assert (
            client.get(
                "/v1/builds/failures", headers=bearer(token_for(state, Operation.READ))
            ).status_code
            == 403
        )
        allowed = client.get(
            "/v1/builds/failures", headers=bearer(token_for(state, Operation.ADMIN))
        )
        assert allowed.status_code == 200
        assert allowed.json() == {"failures": []}
