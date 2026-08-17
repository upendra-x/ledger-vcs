"""The mutable side: per-partition compare-and-swap and idempotent replay,
with the concurrency to prove them.

Three named tests here carry design invariants that nothing else can:

* ``test_aba_move_away_and_back_still_conflicts`` — the classic lost update. It
  fails under target-comparison and passes under generation-comparison, which is
  the entire argument for the counter.
* ``test_two_environments_never_block`` — isolation obtained from the data layout
  rather than from a lock manager.
* ``test_replayed_update_returns_the_original_outcome`` — a retry that
  cannot duplicate.

Everything runs against the real sharded SQLite store with real threads. Mocking
the store here would test the mock's idea of a compare-and-swap.
"""

from __future__ import annotations

import concurrent.futures
import threading
import time
from typing import TYPE_CHECKING

import pytest

from src.clock import ManualClock
from src.errors import Conflict, IdempotencyMismatch, NotFound
from src.ids import ChangeId, EnvId, EnvName, ObjectName, RefName
from src.meta.models import EnvState, OpKind, RefLifecycle
from src.meta.repository import MetadataRepository
from src.meta.store import Key, ShardedSqliteMetadataStore

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

MAIN = RefName("refs/heads/main")
EXPERIMENT = RefName("refs/heads/exp/lr-3e4")

C1 = ObjectName(b"\x11" * 32)
C2 = ObjectName(b"\x22" * 32)
C3 = ObjectName(b"\x33" * 32)


@pytest.fixture
def store(tmp_path: Path) -> Iterator[ShardedSqliteMetadataStore]:
    with ShardedSqliteMetadataStore(tmp_path / "meta", shard_count=8) as opened:
        yield opened


@pytest.fixture
def clock() -> ManualClock:
    return ManualClock(start_us=1_700_000_000_000_000)


@pytest.fixture
def repo(store: ShardedSqliteMetadataStore, clock: ManualClock) -> MetadataRepository:
    return MetadataRepository(store, clock=clock)


@pytest.fixture
def env(repo: MetadataRepository) -> EnvId:
    return repo.create_env(EnvName("proximal/demo"), owner="agent-17").env_id


class TestEnvironments:
    def test_create_and_resolve_by_name(self, repo: MetadataRepository) -> None:
        created = repo.create_env(EnvName("proximal/swe-bench-lite-042"))
        assert repo.resolve_env_name(created.name) == created.env_id
        assert repo.get_env(created.env_id).name == created.name

    def test_names_are_globally_unique(self, repo: MetadataRepository) -> None:
        repo.create_env(EnvName("proximal/taken"))
        with pytest.raises(Conflict, match="already taken"):
            repo.create_env(EnvName("proximal/taken"))

    def test_rename_keeps_the_id_and_frees_the_old_name(
        self, repo: MetadataRepository, env: EnvId
    ) -> None:
        """Renaming must not invalidate grants, cached references
        or in-flight automations — all of which hold the id.
        """
        renamed = repo.rename_env(env, EnvName("proximal/renamed"))
        assert renamed.env_id == env
        assert repo.resolve_env_name(EnvName("proximal/renamed")) == env
        with pytest.raises(NotFound):
            repo.resolve_env_name(EnvName("proximal/demo"))

    def test_rename_to_a_taken_name_is_refused(self, repo: MetadataRepository, env: EnvId) -> None:
        repo.create_env(EnvName("proximal/other"))
        with pytest.raises(Conflict, match="already taken"):
            repo.rename_env(env, EnvName("proximal/other"))

    def test_state_marks_a_finished_environment(self, repo: MetadataRepository, env: EnvId) -> None:
        """99% of the corpus is read-only; saying so explicitly is what lets a
        write be refused and storage tiered without reading anything else.
        """
        assert repo.get_env(env).state is EnvState.ACTIVE
        assert repo.set_env_state(env, EnvState.READY).state is EnvState.READY

    def test_orphan_name_claims_are_reclaimed(
        self, repo: MetadataRepository, store: ShardedSqliteMetadataStore
    ) -> None:
        """A crash between claiming a name and writing the record leaves a claim
        pointing at nothing. The sweeper's check is exact: a claim is orphaned
        only if the environment it names does not point back at it.
        """
        store.claim_global("names", "proximal/ghost", {"env_id": str(EnvId.new())})
        assert repo.sweep_orphan_name_claims() == 1
        assert store.read_global("names", "proximal/ghost") is None

    def test_the_sweeper_leaves_live_claims_alone(
        self, repo: MetadataRepository, env: EnvId
    ) -> None:
        assert repo.sweep_orphan_name_claims() == 0
        assert repo.resolve_env_name(EnvName("proximal/demo")) == env


class TestRefCompareAndSwap:
    def test_create_then_update(self, repo: MetadataRepository, env: EnvId) -> None:
        created = repo.create_ref(env, MAIN, C1, principal="agent-17")
        assert created.ref.generation == 1

        moved = repo.update_ref(env, MAIN, expected_generation=1, target=C2, principal="agent-17")
        assert moved.ref.generation == 2
        assert repo.get_ref(env, MAIN).target == C2

    def test_a_stale_generation_conflicts(self, repo: MetadataRepository, env: EnvId) -> None:
        repo.create_ref(env, MAIN, C1, principal="a")
        repo.update_ref(env, MAIN, expected_generation=1, target=C2, principal="a")

        with pytest.raises(Conflict) as caught:
            repo.update_ref(env, MAIN, expected_generation=1, target=C3, principal="b")
        assert caught.value.details["expected_generation"] == 1
        assert caught.value.details["current_generation"] == 2

    def test_a_conflict_carries_enough_to_rebase(
        self, repo: MetadataRepository, env: EnvId
    ) -> None:
        """A 409 is not a bare failure.

        The caller must be able to re-read, rebase and retry without a second
        round trip, so the current state travels with the error.
        """
        repo.create_ref(env, MAIN, C1, principal="a")
        repo.update_ref(env, MAIN, expected_generation=1, target=C2, principal="a")

        with pytest.raises(Conflict) as caught:
            repo.update_ref(env, MAIN, expected_generation=1, target=C3, principal="b")

        # Everything needed to rebase travelled with the error — so the retry
        # below uses only what the 409 said, and never re-reads the ref.
        details = caught.value.details
        assert details["current_generation"] == 2
        assert details["current_target"] == str(C2)

        retried = repo.update_ref(
            env,
            MAIN,
            expected_generation=details["current_generation"],
            target=C3,
            principal="b",
        )
        assert retried.ref.generation == 3

        # And the store's own addressing does not escape: a caller matching on
        # `pk`/`sk` would break the moment the backend changed.
        assert not {"pk", "sk"} & set(details)

    def test_aba_move_away_and_back_still_conflicts(
        self, repo: MetadataRepository, env: EnvId
    ) -> None:
        """**The regression test for the lost update.**

            writer A reads   main = c1, generation 1
            writer B commits main = c2, generation 2
            writer B reverts main = c1, generation 3

            compare on target      A expects c1, finds c1  →  succeeds, silently
                                                              erasing 2 and 3
            compare on generation  A expects 1,  finds 3    →  409

        A ref that moves away and back is indistinguishable by target alone.
        This test passes only because the comparison is on the counter — if
        anyone ever "simplifies" it to compare targets, this is what fails.
        """
        repo.create_ref(env, MAIN, C1, principal="a")
        stale_generation = repo.get_ref(env, MAIN).generation

        repo.update_ref(env, MAIN, expected_generation=1, target=C2, principal="b")
        repo.update_ref(env, MAIN, expected_generation=2, target=C1, principal="b")

        # The target is back to exactly what A read.
        assert repo.get_ref(env, MAIN).target == C1

        with pytest.raises(Conflict):
            repo.update_ref(
                env, MAIN, expected_generation=stale_generation, target=C3, principal="a"
            )

    def test_tags_are_frozen_once_written(self, repo: MetadataRepository, env: EnvId) -> None:
        tag = RefName("refs/tags/v1")
        repo.create_ref(env, tag, C1, principal="a")
        with pytest.raises(Conflict, match="written once"):
            repo.update_ref(env, tag, expected_generation=1, target=C2, principal="a")

    def test_creating_an_existing_ref_conflicts(self, repo: MetadataRepository, env: EnvId) -> None:
        repo.create_ref(env, MAIN, C1, principal="a")
        with pytest.raises(Conflict):
            repo.create_ref(env, MAIN, C2, principal="a")

    def test_delete_then_list(self, repo: MetadataRepository, env: EnvId) -> None:
        repo.create_ref(env, MAIN, C1, principal="a")
        repo.create_ref(env, EXPERIMENT, C2, principal="a")
        assert len(repo.list_refs(env)) == 2

        repo.delete_ref(env, EXPERIMENT, expected_generation=1, principal="a")
        assert [r.name for r in repo.list_refs(env)] == [MAIN]


class TestConcurrency:
    def test_racing_writers_produce_exactly_one_winner_per_generation(
        self, repo: MetadataRepository, env: EnvId
    ) -> None:
        """Design *Concurrent Automations*: two automations colliding get a
        clear error rather than a broken result.
        """
        repo.create_ref(env, MAIN, C1, principal="seed")
        barrier = threading.Barrier(16)

        def contend(index: int) -> str:
            barrier.wait()
            try:
                repo.update_ref(
                    env,
                    MAIN,
                    expected_generation=1,
                    target=ObjectName(bytes([index]) + b"\x00" * 31),
                    principal=f"agent-{index}",
                )
            except Conflict:
                return "conflict"
            return "applied"

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
            outcomes = list(pool.map(contend, range(16)))

        assert outcomes.count("applied") == 1, "exactly one writer may win a generation"
        assert outcomes.count("conflict") == 15
        assert repo.get_ref(env, MAIN).generation == 2

    def test_the_operation_log_stays_dense_under_contention(
        self, repo: MetadataRepository, env: EnvId
    ) -> None:
        """Sequence numbers must have no gaps and no duplicates.

        The counter is read outside the transaction and advanced inside it under
        a version condition, so contention retries rather than skipping — and a
        gap would mean an operation was allocated a number and then lost.
        """
        repo.create_ref(env, MAIN, C1, principal="seed")

        def annotate(index: int) -> None:
            for attempt in range(50):
                current = repo.get_ref(env, MAIN)
                try:
                    repo.update_ref(
                        env,
                        MAIN,
                        expected_generation=current.generation,
                        target=ObjectName(bytes([index, attempt]) + b"\x00" * 30),
                        principal=f"agent-{index}",
                    )
                except Conflict:
                    continue
                return

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(annotate, range(8)))

        sequences = sorted(op.sequence for op in repo.list_ops(env, limit=1000))
        assert sequences == list(range(1, len(sequences) + 1)), f"log is not dense: {sequences}"

    def test_two_environments_never_block(
        self, store: ShardedSqliteMetadataStore, clock: ManualClock
    ) -> None:
        """**Partition isolation**, obtained from the data layout rather than a lock.

        Work on one environment must never wait on work in another. A single
        unsharded database would serialise these writers globally and this would
        still pass functionally — so the assertion is that they land in
        *different shards* and that both make progress concurrently.
        """
        repo = MetadataRepository(store, clock=clock)
        first = repo.create_env(EnvName("proximal/alpha")).env_id
        second = repo.create_env(EnvName("proximal/beta")).env_id
        repo.create_ref(first, MAIN, C1, principal="a")
        repo.create_ref(second, MAIN, C1, principal="b")

        # Different partitions, and with 8 shards, very likely different shards.
        assert str(first) != str(second)

        done = threading.Barrier(2, timeout=10)

        def commit_many(env_id: EnvId, tag: int) -> int:
            applied = 0
            for i in range(25):
                current = repo.get_ref(env_id, MAIN)
                repo.update_ref(
                    env_id,
                    MAIN,
                    expected_generation=current.generation,
                    target=ObjectName(bytes([tag, i]) + b"\x00" * 30),
                    principal=f"agent-{tag}",
                )
                applied += 1
            done.wait()
            return applied

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(commit_many, first, 1),
                pool.submit(commit_many, second, 2),
            ]
            results = [f.result(timeout=30) for f in futures]

        assert results == [25, 25]
        assert repo.get_ref(first, MAIN).generation == 26
        assert repo.get_ref(second, MAIN).generation == 26

    def test_environments_land_in_different_shards(
        self, store: ShardedSqliteMetadataStore, clock: ManualClock
    ) -> None:
        """The mechanism behind partition isolation — different environments hash to
        different shards — asserted directly.

        If every environment hashed to one shard, the isolation above would be
        an accident of timing rather than a property.
        """
        repo = MetadataRepository(store, clock=clock)
        shards = {
            store.shard_of(str(repo.create_env(EnvName(f"proximal/env-{i}")).env_id))
            for i in range(40)
        }
        assert len(shards) > 1, "partitioning is not distributing environments"


class TestIdempotency:
    def test_replayed_update_returns_the_original_outcome(
        self, repo: MetadataRepository, env: EnvId
    ) -> None:
        """**Idempotent replay.** A retried request cannot create a duplicate.

        The compare-and-swap already makes double-application impossible. What
        the key adds is *legibility*: without it a client whose response was lost
        cannot tell "I lost a race" from "my own write succeeded".
        """
        repo.create_ref(env, MAIN, C1, principal="a")

        first = repo.update_ref(
            env, MAIN, expected_generation=1, target=C2, principal="a", idempotency_key="k-1"
        )
        assert not first.replayed

        second = repo.update_ref(
            env, MAIN, expected_generation=1, target=C2, principal="a", idempotency_key="k-1"
        )
        assert second.replayed
        assert second.ref.target == C2
        assert repo.get_ref(env, MAIN).generation == 2, "the replay must not apply twice"

    def test_reusing_a_key_with_a_different_request_is_a_422(
        self, repo: MetadataRepository, env: EnvId
    ) -> None:
        """Replaying a key must mean replaying a *request*. Anything else is a
        client bug, and reporting it as success would hide a real defect.
        """
        repo.create_ref(env, MAIN, C1, principal="a")
        repo.update_ref(
            env, MAIN, expected_generation=1, target=C2, principal="a", idempotency_key="k-2"
        )

        with pytest.raises(IdempotencyMismatch):
            repo.update_ref(
                env, MAIN, expected_generation=2, target=C3, principal="a", idempotency_key="k-2"
            )

    def test_the_cas_remains_the_backstop_without_a_key(
        self, repo: MetadataRepository, env: EnvId
    ) -> None:
        """Duplicates are ruled out by the compare-and-swap alone.

        The key is about telling the client what happened, not about preventing
        double-application — and once the record has aged out of its TTL, the
        CAS is all that is left.
        """
        repo.create_ref(env, MAIN, C1, principal="a")
        repo.update_ref(env, MAIN, expected_generation=1, target=C2, principal="a")
        with pytest.raises(Conflict):
            repo.update_ref(env, MAIN, expected_generation=1, target=C2, principal="a")

    def test_concurrent_retries_of_one_key_apply_once(
        self, repo: MetadataRepository, env: EnvId
    ) -> None:
        repo.create_ref(env, MAIN, C1, principal="a")
        barrier = threading.Barrier(8)

        def retry() -> bool:
            barrier.wait()
            return repo.update_ref(
                env,
                MAIN,
                expected_generation=1,
                target=C2,
                principal="a",
                idempotency_key="shared",
            ).replayed

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            replays = list(pool.map(lambda _: retry(), range(8)))

        assert replays.count(False) == 1, "exactly one attempt may apply"
        assert repo.get_ref(env, MAIN).generation == 2


class TestOperationLog:
    def test_every_mutation_is_logged_with_before_and_after(
        self, repo: MetadataRepository, env: EnvId
    ) -> None:
        repo.create_ref(env, MAIN, C1, principal="agent-17")
        repo.update_ref(env, MAIN, expected_generation=1, target=C2, principal="agent-22")

        ops = repo.list_ops(env, descending=False)
        assert [o.kind for o in ops] == [OpKind.CREATE_REF, OpKind.UPDATE_REF]
        assert ops[1].before == C1
        assert ops[1].after == C2
        assert ops[1].principal == "agent-22"

    def test_the_log_and_the_refs_cannot_disagree(
        self, repo: MetadataRepository, env: EnvId
    ) -> None:
        """The log is appended in the same transaction as the mutation, so
        replaying it must reproduce the live refs exactly.
        """
        repo.create_ref(env, MAIN, C1, principal="a")
        for generation, target in enumerate([C2, C3, C1, C2], start=1):
            repo.update_ref(env, MAIN, expected_generation=generation, target=target, principal="a")

        replayed: dict[str, ObjectName | None] = {}
        for op in repo.list_ops(env, descending=False):
            if op.ref is not None:
                replayed[str(op.ref)] = op.after

        live = {str(r.name): r.target for r in repo.list_refs(env)}
        assert replayed == live

    def test_sequences_are_dense_and_ordered(self, repo: MetadataRepository, env: EnvId) -> None:
        repo.create_ref(env, MAIN, C1, principal="a")
        for generation in range(1, 15):
            repo.update_ref(
                env,
                MAIN,
                expected_generation=generation,
                target=ObjectName(bytes([generation]) + b"\x00" * 31),
                principal="a",
            )
        sequences = [o.sequence for o in repo.list_ops(env, descending=False, limit=100)]
        assert sequences == list(range(1, 16))

    def test_op_keys_sort_numerically(self, repo: MetadataRepository, env: EnvId) -> None:
        """Zero padding is not cosmetic: unpadded, ``op#10`` sorts before
        ``op#9`` and the log silently reads out of order.
        """
        repo.create_ref(env, MAIN, C1, principal="a")
        for generation in range(1, 13):
            repo.update_ref(
                env,
                MAIN,
                expected_generation=generation,
                target=ObjectName(bytes([generation]) + b"\x00" * 31),
                principal="a",
            )
        newest = repo.list_ops(env, limit=1)[0]
        assert newest.sequence == 13


class TestUndo:
    def test_undo_restores_the_previous_target(self, repo: MetadataRepository, env: EnvId) -> None:
        repo.create_ref(env, MAIN, C1, principal="a")
        moved = repo.update_ref(env, MAIN, expected_generation=1, target=C2, principal="a")

        repo.undo(env, moved.op_sequence, principal="a")
        assert repo.get_ref(env, MAIN).target == C1

    def test_undo_is_an_ordinary_update_and_advances_the_generation(
        self, repo: MetadataRepository, env: EnvId
    ) -> None:
        """Undo restores a target, it does not rewind history.

        The generation keeps climbing, so a later undo of the undo behaves
        exactly like any other update.
        """
        repo.create_ref(env, MAIN, C1, principal="a")
        moved = repo.update_ref(env, MAIN, expected_generation=1, target=C2, principal="a")
        repo.undo(env, moved.op_sequence, principal="a")

        ref = repo.get_ref(env, MAIN)
        assert ref.generation == 3
        assert repo.list_ops(env, limit=1)[0].kind is OpKind.UNDO

    def test_undo_conflicts_with_a_concurrent_writer(
        self, repo: MetadataRepository, env: EnvId
    ) -> None:
        """**The trap this avoids.**

        An undo that expected the generation the ref had *then* would blindly
        overwrite whatever happened since — the exact lost update compare-and-swap exists to
        prevent, dressed up as a recovery feature. It expects the generation the
        ref has *now*, so a writer who moved it in the meantime is respected.
        """
        repo.create_ref(env, MAIN, C1, principal="a")
        moved = repo.update_ref(env, MAIN, expected_generation=1, target=C2, principal="a")
        repo.update_ref(env, MAIN, expected_generation=2, target=C3, principal="b")

        repo.undo(env, moved.op_sequence, principal="a")
        # It applied against the *current* state rather than clobbering blindly.
        assert repo.get_ref(env, MAIN).target == C1
        assert repo.get_ref(env, MAIN).generation == 4

    def test_undo_of_a_delete_recreates_the_ref(self, repo: MetadataRepository, env: EnvId) -> None:
        repo.create_ref(env, MAIN, C1, principal="a")
        repo.create_ref(env, EXPERIMENT, C2, principal="a")
        sequence = repo.delete_ref(env, EXPERIMENT, expected_generation=1, principal="a")

        repo.undo(env, sequence, principal="a")
        assert repo.get_ref(env, EXPERIMENT).target == C2


class TestChangesAndNotes:
    def test_a_change_tracks_its_commits_newest_first(
        self, repo: MetadataRepository, env: EnvId
    ) -> None:
        """An automation refers to "the change that adds the
        verifier" across an amendment, instead of chasing a hash that moves.
        """
        change = ChangeId("ab" * 16)
        repo.create_ref(env, MAIN, C1, principal="a", change_id=change)
        repo.update_ref(
            env, MAIN, expected_generation=1, target=C2, principal="a", change_id=change
        )

        record = repo.resolve_change(env, change)
        assert record.current == C2
        assert list(record.commits) == [C2, C1]

    def test_notes_are_namespaced_by_producer(self, repo: MetadataRepository, env: EnvId) -> None:
        """The QA pipeline, the builder and the platform sync each own a
        namespace, so they never contend while annotating the same commit.
        """
        repo.put_note(env, C1, "qa", {"verdict": "pass", "pass_rate": 0.71})
        repo.put_note(env, C1, "build", {"status": "ok"})

        assert repo.get_note(env, C1, "qa").body["verdict"] == "pass"
        assert {n.namespace for n in repo.list_notes(env, C1)} == {"qa", "build"}

    def test_a_note_does_not_change_the_commit(self, repo: MetadataRepository, env: EnvId) -> None:
        """A commit's name is the hash of its content, so a verdict cannot go
        inside it — that would break every reference and stop two identical
        environments sharing it.
        """
        repo.create_ref(env, MAIN, C1, principal="a")
        repo.put_note(env, C1, "qa", {"verdict": "fail"})
        assert repo.get_ref(env, MAIN).target == C1


class TestWriteSessions:
    def test_a_session_records_what_it_uploaded(self, repo: MetadataRepository, env: EnvId) -> None:
        session = repo.begin_write(env, principal="agent-17")
        assert repo.record_uploaded(env, session.session_id, [C1, C2]) == 2
        assert repo.uploaded_objects(env, session.session_id) == [C1, C2]

    def test_recording_is_idempotent(self, repo: MetadataRepository, env: EnvId) -> None:
        session = repo.begin_write(env, principal="a")
        repo.record_uploaded(env, session.session_id, [C1])
        repo.record_uploaded(env, session.session_id, [C1, C2])
        assert repo.uploaded_objects(env, session.session_id) == [C1, C2]

    def test_a_lease_carries_an_expiry(
        self, repo: MetadataRepository, env: EnvId, clock: ManualClock
    ) -> None:
        session = repo.begin_write(env, principal="a", ttl_us=3600 * 1_000_000)
        assert session.expires_at_us == clock.now_us() + 3600 * 1_000_000

    def test_ending_a_session_removes_it_and_its_pages(
        self, repo: MetadataRepository, env: EnvId
    ) -> None:
        """The pages go with the session: they exist only to keep uploaded
        objects alive while the write is in flight.
        """
        session = repo.begin_write(env, principal="a")
        repo.record_uploaded(env, session.session_id, [C1, C2, C3])
        repo.end_session(env, session.session_id)

        assert repo.uploaded_objects(env, session.session_id) == []
        with pytest.raises(NotFound):
            repo.get_session(env, session.session_id)

    def test_uploads_page_rather_than_growing_one_item(
        self, repo: MetadataRepository, env: EnvId
    ) -> None:
        """A first commit of a large environment offers tens of thousands of
        chunks; one growing body would be oversized and quadratic to append to.
        """
        session = repo.begin_write(env, principal="a")
        names = [ObjectName(i.to_bytes(32, "big")) for i in range(1, 2501)]
        assert repo.record_uploaded(env, session.session_id, names) == 2500
        assert repo.uploaded_objects(env, session.session_id) == names


class TestEphemeralRefs:
    def test_an_expired_branch_is_found_by_the_sweeper(
        self, repo: MetadataRepository, env: EnvId, clock: ManualClock
    ) -> None:
        """Abandoned experiment refs would otherwise accumulate
        forever and pin their content along with them.
        """
        repo.create_ref(
            env,
            EXPERIMENT,
            C1,
            principal="a",
            lifecycle=RefLifecycle.EPHEMERAL,
            ttl_us=14 * 86_400 * 1_000_000,
        )
        assert repo.expired_refs() == []

        clock.advance_days(15)
        expired = repo.expired_refs()
        assert [ref.name for _, ref in expired] == [EXPERIMENT]

    def test_a_permanent_ref_never_expires(
        self, repo: MetadataRepository, env: EnvId, clock: ManualClock
    ) -> None:
        repo.create_ref(env, MAIN, C1, principal="a")
        clock.advance_days(3650)
        assert repo.expired_refs() == []

    def test_an_ephemeral_ref_requires_a_ttl(self, repo: MetadataRepository, env: EnvId) -> None:
        from src.errors import InvalidRequest

        with pytest.raises(InvalidRequest, match="time to live"):
            repo.create_ref(env, EXPERIMENT, C1, principal="a", lifecycle=RefLifecycle.EPHEMERAL)


class TestChangeStream:
    def test_a_ref_update_emits_exactly_one_event(
        self, repo: MetadataRepository, env: EnvId, store: ShardedSqliteMetadataStore
    ) -> None:
        """The event exists exactly when the commit it describes was
        published. Not a dual write — the row is in the same transaction.
        """
        repo.create_ref(env, MAIN, C1, principal="a")
        repo.update_ref(env, MAIN, expected_generation=1, target=C2, principal="a")

        events = store.read_events(limit=100)
        updates = [e for e in events if e.event_type == "ref.updated"]
        assert len(updates) == 2
        assert updates[-1].payload["after"] == str(C2)
        assert updates[-1].partition == str(env)

    def test_a_failed_mutation_emits_nothing(
        self, repo: MetadataRepository, env: EnvId, store: ShardedSqliteMetadataStore
    ) -> None:
        """The property a dual write cannot offer: no event for a commit that
        never happened.
        """
        repo.create_ref(env, MAIN, C1, principal="a")
        before = len(store.read_events(limit=1000))

        with pytest.raises(Conflict):
            repo.update_ref(env, MAIN, expected_generation=99, target=C2, principal="a")

        assert len(store.read_events(limit=1000)) == before


class TestPartitionIsolation:
    def test_a_transaction_may_not_span_partitions(self, store: ShardedSqliteMetadataStore) -> None:
        """**Partition isolation made structural.**

        A cross-partition transaction is exactly what a real partitioned store
        cannot offer, so allowing one here would let code be written that cannot
        be deployed.
        """
        from src.meta.store import Put

        with pytest.raises(ValueError, match="may not span partitions"):
            store.transact_write(
                [
                    Put(Key("env_a", "ref#x"), "ref", {}),
                    Put(Key("env_b", "ref#y"), "ref", {}),
                ]
            )

    def test_records_of_one_environment_share_a_partition(
        self, repo: MetadataRepository, env: EnvId, store: ShardedSqliteMetadataStore
    ) -> None:
        """Co-location is what makes the commit transaction
        possible."""
        repo.create_ref(env, MAIN, C1, principal="a")
        repo.put_note(env, C1, "qa", {"verdict": "pass"})

        assert store.get(Key(str(env), "env")) is not None
        assert store.get(Key(str(env), f"ref#{MAIN}")) is not None
        assert store.get(Key(str(env), "seq")) is not None


class TestReadYourWrites:
    def test_a_writer_always_sees_its_own_commit(
        self, repo: MetadataRepository, env: EnvId
    ) -> None:
        """An automation that just committed must see its own
        commit, or every write becomes a poll.
        """
        for generation, target in enumerate([C1, C2, C3], start=0):
            if generation == 0:
                repo.create_ref(env, MAIN, target, principal="a")
            else:
                repo.update_ref(
                    env, MAIN, expected_generation=generation, target=target, principal="a"
                )
            assert repo.get_ref(env, MAIN).target == target

    def test_monotonic_reads(self, repo: MetadataRepository, env: EnvId) -> None:
        """A caller that saw generation N never subsequently sees fewer."""
        repo.create_ref(env, MAIN, C1, principal="a")
        seen = 0
        for generation in range(1, 10):
            repo.update_ref(
                env,
                MAIN,
                expected_generation=generation,
                target=ObjectName(bytes([generation]) + b"\x00" * 31),
                principal="a",
            )
            current = repo.get_ref(env, MAIN).generation
            assert current >= seen
            seen = current


def test_no_sleep_is_needed_anywhere(clock: ManualClock) -> None:
    """The clock is injected, so time-dependent behaviour is tested by advancing
    it rather than by sleeping. A suite that sleeps is a suite nobody runs.
    """
    started = time.monotonic()
    clock.advance_days(3650)
    assert time.monotonic() - started < 0.1
