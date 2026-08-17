"""The two moving parts: a dispatcher that turns events into work, and a worker
that does it.

::

    UpdateRef succeeds ──▶ outbox row (same transaction)
                              │
        Dispatcher.poll() ────┴── events → queue entries
                              │
        BuildWorker.run_once() ── lease → build → sync → note

**The dispatcher never invents an event.** It reads the outbox that the ref
update wrote inside its own transaction, so an event exists exactly when the
mutation it describes was published, and there is no second write to keep
consistent. If a ref update rolled back, there is nothing to read.

**The worker is the only thing that can fail, and it fails softly.** A build that
crashes records a failure and lets go of its environment; it never blocks or
reverses the commit that triggered it. The commit is already published — that
decision was made by the ref update, and a build has no vote.

The cache check happens *before* anything expensive. A fork that changed nothing
never materializes, never runs, and never asks a runner anything: it finds its
parent's result keyed by the commit they share, syncs its own environment, and is
done. That is what a build being a pure function of a commit buys, and it is the
difference between forking being free and forking costing a full rebuild.
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Final, final

from src.build.manifest import Manifest as EnvManifest
from src.build.manifest import describe, find_manifest, load_manifest
from src.build.models import BuildOutcome, BuildResult, BuildStatus, SyncRecord
from src.build.queue import BuildLease, BuildQueue
from src.build.results import BuildResults
from src.build.runner import RunRequest
from src.build.sync import SyncRequest
from src.errors import Conflict, LedgerError
from src.format.model import Commit
from src.ids import EnvId, ObjectName, RefName
from src.meta import keys
from src.meta.keys import GlobalSpace, ItemKind
from src.meta.models import Note
from src.meta.store import Key, Put, StreamCursor

if TYPE_CHECKING:
    from src.build.runner import Runner
    from src.build.sync import PlatformSync
    from src.instance import Ledger

__all__ = ["SYNC_NAMESPACE", "BuildWorker", "DispatchReport", "Dispatcher", "trigger"]

logger = logging.getLogger("ledger.build")

#: The note namespace the sync record lands in. Namespaced by producer, so the
#: QA pipeline, the builder and the platform sync never contend even while
#: annotating the same commit.
SYNC_NAMESPACE: Final = "sync"

REF_UPDATED: Final = "ref.updated"


@final
@dataclass(frozen=True, slots=True)
class DispatchReport:
    events: int = 0
    enqueued: int = 0
    #: Events that were real but not worth building — a branch that is not the
    #: environment's build ref, a deletion, an environment since removed.
    ignored: int = 0

    def __add__(self, other: DispatchReport) -> DispatchReport:
        return DispatchReport(
            events=self.events + other.events,
            enqueued=self.enqueued + other.enqueued,
            ignored=self.ignored + other.ignored,
        )


@final
class Dispatcher:
    """Turns ref-update events into queued builds.

    Its cursor is durable and **per shard**, because the stream's sequences are
    per shard. A consumer that collapsed them into one number would silently stop
    delivering events from the quieter shards, and the environments living there
    would simply never build again — with nothing anywhere reporting an error.
    """

    __slots__ = ("_consumer", "_ledger", "_queue")

    def __init__(self, ledger: Ledger, queue: BuildQueue, *, consumer: str = "builder") -> None:
        self._ledger = ledger
        self._queue = queue
        self._consumer = consumer

    def poll(self, *, limit: int = 200) -> DispatchReport:
        """Drain what is new, enqueue what should build, and checkpoint."""
        cursor = self.cursor()
        events = self._ledger.meta.read_events(cursor, limit=limit)
        if not events:
            return DispatchReport()

        report = DispatchReport(events=len(events))
        build_refs: dict[str, str] = {}

        for event in events:
            if event.event_type != REF_UPDATED:
                report += DispatchReport(ignored=1)
                continue

            partition = event.partition
            if partition not in build_refs:
                build_refs[partition] = self._build_ref(partition)
            if build_refs[partition] != str(event.payload.get("ref", "")):
                report += DispatchReport(ignored=1)
                continue

            enqueued = self._queue.enqueue(
                partition,
                str(event.payload["after"]),
                ref=str(event.payload["ref"]),
                op_sequence=int(event.payload["op_sequence"]),
            )
            report += DispatchReport(enqueued=int(enqueued), ignored=int(not enqueued))

        # Checkpointed only after the work is durably queued, so a crash here
        # replays events rather than losing them — and replay is harmless
        # because enqueueing is keyed on the operation.
        self._checkpoint(cursor.advanced(events))
        return report

    def _build_ref(self, partition: str) -> str:
        """Which ref of this environment triggers a build.

        Its default ref, read rather than configured, so an environment that
        renamed its default branch keeps building without anybody updating a
        pipeline.
        """
        try:
            return str(self._ledger.repo.get_env(EnvId(partition)).default_ref)
        except LedgerError:
            return ""  # environment gone; its events are history now

    # ── the cursor ───────────────────────────────────────────────────────────

    def cursor(self) -> StreamCursor:
        item = self._ledger.meta.read_global(GlobalSpace.CURSORS, self._consumer)
        shards = self._ledger.meta.shard_count
        if item is None:
            return StreamCursor.start(shards)
        positions = [int(p) for p in item.body.get("positions", ())]
        # Padded rather than trusted: a shard count that grew since the cursor
        # was written must read the new shards from the beginning, not crash.
        positions.extend([0] * (shards - len(positions)))
        return StreamCursor(positions=tuple(positions[:shards]))

    def _checkpoint(self, cursor: StreamCursor) -> None:
        self._ledger.meta.put_global(
            GlobalSpace.CURSORS, self._consumer, {"positions": list(cursor.positions)}
        )


@final
class BuildWorker:
    """Leases an environment, builds one commit, syncs it, records what happened.

    Holds no state between turns. Everything it needs is in the queue and the
    result store, which is what lets several of these run at once and what makes
    a restart indistinguishable from a pause.
    """

    __slots__ = ("_ledger", "_platform", "_queue", "_results", "_runner", "_worker_id")

    def __init__(
        self,
        ledger: Ledger,
        *,
        runner: Runner,
        platform: PlatformSync,
        worker_id: str = "worker-1",
        queue: BuildQueue | None = None,
        results: BuildResults | None = None,
    ) -> None:
        self._ledger = ledger
        self._runner = runner
        self._platform = platform
        self._worker_id = worker_id
        self._queue = queue or BuildQueue(ledger.meta, clock=ledger.clock)
        self._results = results or BuildResults(ledger.meta, clock=ledger.clock)

    @property
    def queue(self) -> BuildQueue:
        return self._queue

    @property
    def results(self) -> BuildResults:
        return self._results

    def drain(self, *, limit: int = 100) -> list[BuildOutcome]:
        """Work until the queue is empty or ``limit`` builds have been done."""
        outcomes: list[BuildOutcome] = []
        for _ in range(limit):
            outcome = self.run_once()
            if outcome is None:
                break
            outcomes.append(outcome)
        return outcomes

    def run_once(self) -> BuildOutcome | None:
        """One turn. ``None`` when there is nothing to do."""
        lease = self._queue.lease(worker=self._worker_id)
        if lease is None:
            return None

        commit = lease.work.commit
        try:
            cached = self._results.claim(commit, worker=self._worker_id)
            if cached is not None and cached.status.terminal:
                return self._finish_from_cache(lease, cached)
            if cached is not None:
                # Someone else is genuinely building it. Give the environment
                # back with the work still queued rather than duplicating five
                # minutes of effort.
                self._queue.release(lease)
                return None

            result = self._build(lease.work.env_id, commit)
        except Conflict:
            # The lease was taken over while we worked. Everything we might have
            # written is keyed by commit and idempotent, so there is nothing to
            # undo — but this worker no longer owns the environment and must not
            # touch its queue.
            logger.warning("build lease lost for %s", lease.env_id)
            self._results.release(commit)
            return None
        except Exception as exc:
            logger.exception("build failed for %s", commit)
            now = self._ledger.clock.now_us()
            return self._give_up(
                lease,
                BuildResult(
                    commit=commit,
                    status=BuildStatus.FAILED,
                    started_at_us=now,
                    finished_at_us=now,
                    exit_code=None,
                    log_excerpt=f"{type(exc).__name__}: {exc}",
                    worker=self._worker_id,
                ),
            )

        if not result.succeeded:
            return self._give_up(lease, result)

        self._results.record(result)
        return self._finish(lease, result, cache_hit=False)

    def _give_up(self, lease: BuildLease, result: BuildResult) -> BuildOutcome:
        """Record an attempt, and record a *failure* only once retries run out.

        The ordering here is the whole of "retry with backoff, then record a
        failure and stop". A failure written on the first attempt would be cached
        like any other result — and since results are cached by commit, the retry
        would find it and return it, so the retries would never happen at all.
        The claim is therefore released between attempts and only made durable
        when there will be no further attempt.
        """
        attempts = lease.work.attempts + 1
        exhausted = self._queue.fail(lease)
        final = replace(result, attempts=attempts)
        if exhausted:
            self._results.record(final)
        else:
            self._results.release(result.commit)
        return BuildOutcome(work=lease.work, result=final, cache_hit=False)

    # ── building ─────────────────────────────────────────────────────────────

    def _build(self, env_id: str, commit: str) -> BuildResult:
        """Materialize, read the manifest, run. The expensive half."""
        started = self._ledger.clock.now_us()
        name = ObjectName.parse(commit)
        env_name = str(self._ledger.repo.get_env(EnvId(env_id)).name)

        with tempfile.TemporaryDirectory(prefix="ledger-build-") as scratch:
            workspace = Path(scratch) / "env"
            self._ledger.materializer.materialize_commit(name, workspace)

            manifest_path = find_manifest(workspace)
            manifest = load_manifest(manifest_path) if manifest_path else EnvManifest()
            images = self._images(name)

            outcome = self._runner.run(
                RunRequest(
                    commit=commit,
                    env_name=env_name,
                    workspace=workspace,
                    manifest=manifest,
                    images=images,
                )
            )

        return BuildResult(
            commit=commit,
            status=BuildStatus.SUCCEEDED if outcome.succeeded else BuildStatus.FAILED,
            started_at_us=started,
            finished_at_us=self._ledger.clock.now_us(),
            exit_code=outcome.exit_code,
            images=outcome.images,
            log_excerpt=outcome.output,
            manifest=describe(manifest),
            worker=self._worker_id,
        )

    def _images(self, commit: ObjectName) -> tuple[str, ...]:
        """What images this version pinned, as ``name@sha256:…``.

        Read from the commit rather than from the manifest: the commit is what
        actually contains them, and a manifest that disagreed would be a second
        source of truth about a version's contents.
        """
        from src.oci.layout import read_index
        from src.oci.model import REF_NAME_ANNOTATION

        tree = self._ledger.store.get_as(commit, Commit).tree
        index = read_index(self._ledger.store, tree)
        return tuple(
            f"{descriptor.annotation_map.get(REF_NAME_ANNOTATION, '?')}@{descriptor.digest}"
            for descriptor in index.manifests
        )

    # ── finishing ────────────────────────────────────────────────────────────

    def _finish_from_cache(self, lease: BuildLease, cached: BuildResult) -> BuildOutcome:
        if not cached.succeeded:
            # A failure is a fact about the commit, so a second environment
            # reaching the same commit inherits it rather than re-running a build
            # that is known not to work. A manual retrigger is how you force it.
            self._queue.complete(lease)
            return BuildOutcome(work=lease.work, result=cached, cache_hit=True)
        return self._finish(lease, cached, cache_hit=True)

    def _finish(self, lease: BuildLease, result: BuildResult, *, cache_hit: bool) -> BuildOutcome:
        """Sync, then remove the work and write the note in one transaction."""
        env = EnvId(lease.env_id)
        record = self._ledger.repo.get_env(env)

        outcome = self._platform.sync(
            SyncRequest(
                commit=result.commit,
                env_id=lease.env_id,
                env_name=str(record.name),
                images=result.images,
                metadata={str(k): str(v) for k, v in record.labels.items()},
            ),
            now_us=self._ledger.clock.now_us(),
        )
        sync = SyncRecord(
            platform_id=outcome.platform_id,
            commit=result.commit,
            synced_at_us=outcome.synced_at_us,
            images=result.images,
        )

        # The note and the queue entry move together: either the environment
        # records what happened and the work disappears, or neither does. Both
        # live in the environment's partition, so this stays one single-partition
        # transaction and no environment can block another.
        note = Note(
            commit=ObjectName.parse(result.commit),
            namespace=SYNC_NAMESPACE,
            body=sync.to_body(),
            updated_by=self._worker_id,
            updated_at_us=self._ledger.clock.now_us(),
        )
        self._queue.complete(
            lease,
            extra=[
                Put(
                    Key(lease.env_id, keys.note_sk(note.commit, SYNC_NAMESPACE)),
                    ItemKind.NOTE,
                    note.to_body(),
                )
            ],
        )
        return BuildOutcome(work=lease.work, result=result, cache_hit=cache_hit, synced=sync)


def trigger(
    ledger: Ledger,
    env: EnvId,
    ref: RefName,
    *,
    queue: BuildQueue | None = None,
    rebuild: bool = False,
) -> bool:
    """Ask for a build directly — reruns and backfills.

    ``rebuild`` **invalidates the cached result** rather than telling the worker
    to ignore it. That distinction matters: a build is a pure function of a
    commit, so "build it again" is a statement that the previous answer is no
    longer trusted — and the honest way to say that is to remove the answer,
    where every consumer sees it, rather than to let one caller quietly disagree.

    The queue key is the environment's newest operation, so a manual trigger
    lands behind whatever ref updates are already waiting rather than jumping
    the queue and syncing an older version after a newer one.
    """
    target = ledger.repo.get_ref(env, ref)
    work_queue = queue or BuildQueue(ledger.meta, clock=ledger.clock)
    recent = ledger.repo.list_ops(env, limit=1)
    sequence = recent[0].sequence if recent else 0

    if rebuild:
        BuildResults(ledger.meta, clock=ledger.clock).invalidate(target.target)
    return work_queue.enqueue(env, target.target, ref=ref, op_sequence=sequence, manual=True)
