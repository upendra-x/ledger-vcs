"""Storage that is supposed to come back, coming back.

Three mechanisms existed, were tested in isolation, and were wired to nothing.
Each test here goes through the surface a caller actually uses — the ref service,
the maintenance pass — rather than calling the mechanism directly, because
"implemented" and "reachable" were exactly the two things that had come apart.

The load-bearing one is the first: *discarding a branch frees whatever storage it
alone was using*. Deleting the ref was only ever half of that. The other half —
shrinking the environment's keep-set — was left to the caller, no caller did it,
and the property quietly did not hold through the API or the CLI.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import pytest

from src.clock import ManualClock
from src.format.cdc import ChunkParams
from src.format.shape import ShapeParams
from src.ids import EnvName, RefName
from src.instance import Ledger
from src.maintenance.gc import GcConfig
from src.maintenance.reclaim import run_maintenance
from src.service.commits import CommitService
from src.service.refs import RefService

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from src.ids import EnvId

MAIN = RefName("refs/heads/main")
BRANCH = RefName("refs/heads/exp/throwaway")

#: Short enough that a test can step past both without sleeping. The relationship
#: is the real one: a discarded branch's content stays alive exactly as long as
#: the operation log can still undo the deletion.
FAST_GC = GcConfig(
    grace_us=60 * 1_000_000,
    op_log_retention_us=300 * 1_000_000,
    shard_bits=2,
    min_corpus_objects=1_000_000,
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
        gc_config=FAST_GC,
    ) as opened:
        yield opened


@pytest.fixture
def env(ledger: Ledger) -> EnvId:
    return ledger.repo.create_env(EnvName("proximal/demo"), owner="agent-17").env_id


@pytest.fixture
def source(tmp_path: Path) -> Path:
    root = tmp_path / "env"
    root.mkdir()
    (root / "harbor.yaml").write_text("name: demo\n")
    (root / "shared.bin").write_bytes(random.Random(1).randbytes(80_000))
    return root


class TestDiscardingABranchFreesItsStorage:
    def test_undo_holds_the_branch_until_the_log_ages_out(
        self, ledger: Ledger, env: EnvId, source: Path, clock: ManualClock
    ) -> None:
        """Deleting a branch must *not* free it immediately, and this says why.

        The operation log still names the branch's commit, so undo can restore
        it — and undo cannot resurrect objects collection has already taken. The
        keep-set is therefore expected to be unchanged right after the delete,
        and to shrink only once the log ages past its retention.
        """
        commits = CommitService(ledger)
        refs = RefService(ledger)
        commits.commit(env, MAIN, source, author="a", message="v1")

        refs.create(env, BRANCH, principal="agent-17")
        (source / "only-on-the-branch.bin").write_bytes(random.Random(2).randbytes(120_000))
        commits.commit(env, BRANCH, source, author="a", message="experiment")
        with_branch = ledger.keepsets.size(str(env))

        immediately = refs.delete(env, BRANCH, principal="agent-17").keep_set_size
        assert immediately == with_branch, "undo could no longer reach the branch it just deleted"

        clock.advance_days(30)
        ledger.gc.refresh_keep_sets()

        assert ledger.keepsets.size(str(env)) < with_branch, (
            "the operation log aged out and nothing recomputed the keep-set, so "
            "the branch's exclusive content can never be collected"
        )

    def test_the_content_actually_comes_back(
        self, ledger: Ledger, env: EnvId, source: Path, clock: ManualClock
    ) -> None:
        """And the whole way through: sweep, and the bytes are returned."""
        commits = CommitService(ledger)
        refs = RefService(ledger)
        commits.commit(env, MAIN, source, author="a", message="v1")

        refs.create(env, BRANCH, principal="agent-17")
        (source / "only-on-the-branch.bin").write_bytes(random.Random(3).randbytes(200_000))
        commits.commit(env, BRANCH, source, author="a", message="experiment")
        (source / "only-on-the-branch.bin").unlink()

        refs.delete(env, BRANCH, principal="agent-17")
        # Past the grace period *and* past operation-log retention: until the log
        # ages out, undo can still reach the branch and nothing may be taken.
        clock.advance_days(30)
        ledger.gc.refresh_keep_sets()

        report = ledger.gc.run(enforce=True)
        assert report.deleted > 0, "nothing was reclaimed after a branch was discarded"
        assert report.bytes_freed > 0

    def test_content_shared_with_main_is_never_taken(
        self, ledger: Ledger, env: EnvId, source: Path, clock: ManualClock
    ) -> None:
        """The half that matters more. A collector that freed the branch and
        broke ``main`` would satisfy every assertion above.
        """
        commits = CommitService(ledger)
        refs = RefService(ledger)
        head = commits.commit(env, MAIN, source, author="a", message="v1")

        refs.create(env, BRANCH, principal="agent-17")
        (source / "only-on-the-branch.bin").write_bytes(random.Random(4).randbytes(200_000))
        commits.commit(env, BRANCH, source, author="a", message="experiment")
        refs.delete(env, BRANCH, principal="agent-17")
        clock.advance_days(30)
        ledger.gc.refresh_keep_sets()
        ledger.gc.run(enforce=True)

        # main's version still reads, byte for byte.
        restored = source.parent / "restored"
        ledger.materializer.materialize_commit(head.commit, restored)
        assert (restored / "shared.bin").stat().st_size == 80_000


class TestEphemeralRefsExpire:
    def test_a_branch_past_its_ttl_is_discarded_by_maintenance(
        self, ledger: Ledger, env: EnvId, source: Path, clock: ManualClock
    ) -> None:
        """An ephemeral ref carried an ``expires_at`` that nothing ever read.

        At the projected hundred thousand actively-iterated environments running
        ten experiments each, that is a million refs pinning content nobody
        wants — and every one of them a live retention root.
        """
        commits = CommitService(ledger)
        refs = RefService(ledger)
        commits.commit(env, MAIN, source, author="a", message="v1")
        refs.create(env, BRANCH, principal="agent-17", ephemeral=True, ttl_days=14)

        assert any(r.name == BRANCH for r in ledger.repo.list_refs(env))

        clock.advance_days(20)
        report = run_maintenance(
            ledger.repo,
            ledger.store.tombstones,
            clock=clock,
            discard=lambda e, n: refs.delete(e, n, principal="ledger-maintenance"),
        )

        assert report.expired_refs == 1
        assert not any(r.name == BRANCH for r in ledger.repo.list_refs(env))

    def test_a_branch_inside_its_ttl_is_left_alone(
        self, ledger: Ledger, env: EnvId, source: Path, clock: ManualClock
    ) -> None:
        commits = CommitService(ledger)
        refs = RefService(ledger)
        commits.commit(env, MAIN, source, author="a", message="v1")
        refs.create(env, BRANCH, principal="agent-17", ephemeral=True, ttl_days=14)

        clock.advance_days(3)
        report = run_maintenance(
            ledger.repo,
            ledger.store.tombstones,
            clock=clock,
            discard=lambda e, n: refs.delete(e, n, principal="ledger-maintenance"),
        )

        assert report.expired_refs == 0
        assert any(r.name == BRANCH for r in ledger.repo.list_refs(env))

    def test_a_permanent_branch_is_never_expired(
        self, ledger: Ledger, env: EnvId, source: Path, clock: ManualClock
    ) -> None:
        """``permanent`` means permanent. A maintenance pass that took these
        would be deleting people's work on a timer.
        """
        commits = CommitService(ledger)
        refs = RefService(ledger)
        commits.commit(env, MAIN, source, author="a", message="v1")
        refs.create(env, BRANCH, principal="agent-17", ephemeral=False)

        clock.advance_days(400)
        report = run_maintenance(
            ledger.repo,
            ledger.store.tombstones,
            clock=clock,
            discard=lambda e, n: refs.delete(e, n, principal="ledger-maintenance"),
        )

        assert report.expired_refs == 0
        assert any(r.name == BRANCH for r in ledger.repo.list_refs(env))


class TestMaintenanceIsSafeToRepeat:
    def test_a_second_pass_finds_nothing(
        self, ledger: Ledger, env: EnvId, source: Path, clock: ManualClock
    ) -> None:
        """Every step is independently idempotent, so an interrupted pass leaves
        the next one something to finish rather than something to undo.
        """
        commits = CommitService(ledger)
        refs = RefService(ledger)
        commits.commit(env, MAIN, source, author="a", message="v1")
        refs.create(env, BRANCH, principal="agent-17", ephemeral=True, ttl_days=1)
        clock.advance_days(5)

        discard = lambda e, n: refs.delete(e, n, principal="ledger-maintenance")  # noqa: E731
        first = run_maintenance(ledger.repo, ledger.store.tombstones, clock=clock, discard=discard)
        second = run_maintenance(ledger.repo, ledger.store.tombstones, clock=clock, discard=discard)

        assert first.expired_refs == 1
        assert second.expired_refs == 0

    def test_it_runs_on_an_empty_ledger(self, ledger: Ledger, clock: ManualClock) -> None:
        report = run_maintenance(
            ledger.repo,
            ledger.store.tombstones,
            clock=clock,
            discard=lambda e, n: None,
        )
        assert report.total == 0
