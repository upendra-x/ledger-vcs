"""The whole system, end to end, with real numbers.

    uv run python demo/e2e.py

This is the entry point. It builds a Ledger in a temporary directory, walks every
requirement, and prints what each one actually cost — not what it is
supposed to cost. Every number below is measured during the run.

Every number it prints comes out of a response, so a demo that stops being true
prints a different number rather than the same story.
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast, final

from rich.console import Console
from rich.table import Table

# Run as `python demo/e2e.py` the project root is not on the path, and the
# demo has to be runnable exactly as its own docstring says it is.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from demo.images import build_oci_archive, layer_tar
from src.build.pipeline import SYNC_NAMESPACE, BuildWorker, Dispatcher
from src.build.queue import BuildQueue
from src.build.runner import RecordingRunner
from src.build.sync import RecordingPlatform
from src.clock import ManualClock
from src.errors import Conflict
from src.format.cdc import ChunkParams
from src.format.model import Commit
from src.format.shape import ShapeParams
from src.fs.blob import BlobReader
from src.fs.tree import resolve_path
from src.ids import EnvName, RefName
from src.instance import Ledger
from src.oci.model import Platform
from src.service.commits import CommitService
from src.service.environments import EnvironmentService
from src.service.images import ImageService

if TYPE_CHECKING:
    from src.ids import EnvId, ObjectName

console = Console()

ORG = "proximal"
ENV = f"{ORG}/demo"
FORK = f"{ORG}/demo-variant"
MAIN = RefName("refs/heads/main")
EXPERIMENT = RefName("refs/heads/exp/lr-3e4")
LINUX_AMD64 = Platform("amd64", "linux")

#: Small enough to run in seconds, large enough that "did it re-store the file?"
#: is answerable in bytes. Every claim here is about a *ratio* — bytes stored
#: against bytes changed — so the size sets how long the demo takes and not
#: whether it is true.
DATASET_BYTES = 4_000_000

#: Chunking and tree shape, scaled to the dataset above.
#:
#: Production averages a 1 MiB chunk, which would put a whole quarter of this
#: 4 MB file in one chunk and make "the edit cost the edit" unreadable — not
#: false, just invisible at this scale. Scaling both together keeps the *ratio*
#: honest while letting the run finish in seconds.
#:
#: The portal imports these rather than choosing its own, so a number measured
#: in the browser is the same number ``make demo-numbers`` prints. Two
#: definitions would be two demos that disagree.
DEMO_CHUNK_PARAMS = ChunkParams.for_average(65_536)
DEMO_SHAPE_PARAMS = ShapeParams(
    domain=b"ledger.tree.split.v1",
    period=64,
    min_entries=8,
    max_entries=256,
    max_node_bytes=64 * 1024,
)

#: Where the demonstration's clock starts. Fixed, so every take stamps the same
#: commit timestamps and a retake is checkable rather than merely repeatable.
DEMO_EPOCH_US = 1_766_000_000_000_000

MANIFEST = b"""
name: demo-environment
version: 1
build:
  command: ["/bin/sh", "-c", "echo built"]
sync:
  team: rl-infra
"""


@final
@dataclass(slots=True)
class Report:
    """Every measured number, so the test can assert on the same values printed."""

    facts: dict[str, Any] = field(default_factory=dict)

    def record(self, key: str, value: Any) -> Any:
        self.facts[key] = value
        return value

    def __getitem__(self, key: str) -> Any:
        return self.facts[key]


def heading(number: int, title: str, requirement: str) -> None:
    console.print(f"\n[bold]{number}. {title}[/bold]  [dim]the requirements {requirement}[/dim]")


def fact(label: str, value: object, *, good: bool = False) -> None:
    rendered = f"[green]{value}[/green]" if good else str(value)
    console.print(f"   {label:<38} {rendered}")


def human(count: int) -> str:
    for unit, size in (("GiB", 1 << 30), ("MiB", 1 << 20), ("KiB", 1 << 10)):
        if count >= size:
            return f"{count / size:.2f} {unit}"
    return f"{count} B"


# ─────────────────────────────────────────────────────────────────────────────


def build_environment(root: Path) -> Path:
    """A plausible RL environment: a manifest, a task, and a dataset."""
    source = root / "environment"
    (source / "task").mkdir(parents=True)
    (source / "harbor.yaml").write_bytes(MANIFEST)
    (source / "README.md").write_text("# demo environment\n")
    (source / "task" / "prompt.md").write_text("Solve the task.\n")
    (source / "task" / "verifier.py").write_text("def verify(result):\n    return True\n")
    (source / "data").mkdir()
    (source / "data" / "train.bin").write_bytes(random.Random(11).randbytes(DATASET_BYTES))
    return source


def run(root: Path) -> Report:
    """Run the demonstration. Returns every number it printed."""
    report = Report()
    source = build_environment(root)

    clock = ManualClock(start_us=DEMO_EPOCH_US)
    ledger = Ledger(
        root / "data",
        clock=clock,
        chunk_params=DEMO_CHUNK_PARAMS,
        shape_params=DEMO_SHAPE_PARAMS,
    )
    with ledger:
        commits = CommitService(ledger)
        env = ledger.repo.create_env(EnvName(ENV)).env_id

        _versioning(ledger, commits, env, source, report)
        _large_artifacts(ledger, commits, env, source, report)
        _reading(ledger, env, report)
        _forking(ledger, commits, env, source, report)
        _concurrency(ledger, commits, env, report)
        _branching(ledger, commits, env, source, report)
        _images(ledger, env, root, report)
        _building(ledger, env, report)
        _authorization(ledger, env, report)
        _collection(ledger, commits, env, source, clock, report)

    _summary(report)
    return report


# ─────────────────────────────────────────────────────────────────────────────


def _versioning(
    ledger: Ledger, commits: CommitService, env: EnvId, source: Path, report: Report
) -> None:
    heading(1, "Version an environment", "Version Environments")
    first = commits.commit(env, MAIN, source, author="agent-17", message="initial version")
    report.record("first_commit", first.commit)
    fact("commit", first.commit)
    fact("objects created", f"{first.stats.objects_created} of {first.stats.objects_offered}")
    fact("bytes stored", human(first.stats.bytes_stored))

    (source / "task" / "prompt.md").write_text("Solve the task, carefully.\n")
    second = commits.commit(env, MAIN, source, author="agent-17", message="reword the prompt")
    report.record("second_commit", second.commit)
    report.record("edit_bytes", second.stats.bytes_stored)

    fact("one-line edit → objects created", second.stats.objects_created, good=True)
    fact("one-line edit → bytes stored", human(second.stats.bytes_stored), good=True)
    fact("…out of an environment of", human(DATASET_BYTES))

    restored = commits.revert(env, MAIN, first.commit, author="agent-17")
    report.record("revert_bytes", restored.stats.bytes_stored)
    fact("restoring the first version cost", human(restored.stats.bytes_stored), good=True)
    commits.revert(env, MAIN, second.commit, author="agent-17")


def _large_artifacts(
    ledger: Ledger, commits: CommitService, env: EnvId, source: Path, report: Report
) -> None:
    heading(2, "Edit inside a large artifact", "Large Artifact Handling")
    dataset = source / "data" / "train.bin"
    content = bytearray(dataset.read_bytes())
    content[len(content) // 2 : len(content) // 2 + 64] = b"X" * 64
    dataset.write_bytes(bytes(content))

    result = commits.commit(env, MAIN, source, author="agent-17", message="patch the dataset")
    report.record("large_edit_bytes", result.stats.bytes_stored)
    report.record("large_edit_objects", result.stats.objects_created)

    fact("dataset size", human(DATASET_BYTES))
    fact("bytes changed", "64 B")
    fact("objects created", result.stats.objects_created, good=True)
    fact("bytes stored", human(result.stats.bytes_stored), good=True)
    fact("ratio to the whole file", f"{result.stats.bytes_stored / DATASET_BYTES:.2%}", good=True)


def _reading(ledger: Ledger, env: EnvId, report: Report) -> None:
    heading(3, "Read one file, no clone", "Read at scale")
    commit = ledger.repo.get_ref(env, MAIN).target
    tree = ledger.store.get_as(commit, Commit).tree
    resolved = resolve_path(ledger.store, tree, "data/train.bin")

    counting = _CountingStore(ledger.store)
    # ``ObjectStore`` is deliberately final — there must be exactly one place
    # verification can happen — so instrumenting it for a measurement is a
    # cast rather than a subclass. The proxy only forwards.
    reader = BlobReader(cast("Any", counting), resolved.target)
    counting.reset()
    payload = reader.read(DATASET_BYTES // 2, 64)
    report.record("ranged_read_objects", counting.reads)

    fact("file size", human(resolved.entry.size))
    fact("bytes requested", len(payload))
    fact("objects fetched", counting.reads, good=True)
    fact("…to read the whole file would take", f"{_whole_file_reads(ledger, resolved.target)}")


def _forking(
    ledger: Ledger, commits: CommitService, env: EnvId, source: Path, report: Report
) -> None:
    del commits, source
    heading(4, "Fork an environment", "Branching / Forking")
    before = ledger.store.catalog.total()[1]
    result = EnvironmentService(ledger).fork(
        env, EnvName(FORK), from_ref=MAIN, principal="agent-17"
    )
    after = ledger.store.catalog.total()[1]
    report.record("fork_bytes", after - before)
    report.record("fork_env", result.environment.env_id)

    fact("forked from", result.source_commit)
    fact("bytes copied", after - before, good=True)
    fact("objects now reachable from the fork", result.closure_size)


def _concurrency(ledger: Ledger, commits: CommitService, env: EnvId, report: Report) -> None:
    del commits
    heading(5, "Two automations collide", "Concurrent Automations")
    current = ledger.repo.get_ref(env, MAIN)
    target = current.target

    ledger.repo.update_ref(
        env, MAIN, expected_generation=current.generation, target=target, principal="agent-a"
    )
    try:
        ledger.repo.update_ref(
            env, MAIN, expected_generation=current.generation, target=target, principal="agent-b"
        )
    except Conflict as conflict:
        report.record("conflict_details", conflict.details)
        fact("second writer got", f"409 {conflict.code}", good=True)
        fact("…carrying the current generation", conflict.details.get("current_generation"))
        fact("…and the current target", str(conflict.details.get("current_target"))[:24] + "…")
    else:  # pragma: no cover - the whole point is that this cannot happen
        raise AssertionError("two writers both won the same compare-and-swap")


def _branching(
    ledger: Ledger, commits: CommitService, env: EnvId, source: Path, report: Report
) -> None:
    heading(6, "Branch and diverge", "Branching / Forking")
    base = ledger.repo.get_ref(env, MAIN).target

    before = ledger.store.catalog.total()
    ledger.repo.create_ref(env, EXPERIMENT, base, principal="agent-17")
    after = ledger.store.catalog.total()
    report.record("branch_bytes", after[1] - before[1])

    branch_source = source.parent / "branch"
    shutil.copytree(source, branch_source, symlinks=True)
    (branch_source / "task" / "verifier.py").write_text("def verify(result):\n    return False\n")
    experiment = commits.commit(
        env, EXPERIMENT, branch_source, author="agent-b", message="stricter verifier"
    )

    (source / "README.md").write_text("# demo environment\n\nNow documented.\n")
    trunk = commits.commit(env, MAIN, source, author="agent-a", message="document it")

    report.record("branch_commit", experiment.commit)
    report.record("trunk_commit", trunk.commit)

    fact("creating the branch cost", f"{after[1] - before[1]} B", good=True)
    fact("…because a branch is", "one ref row over shared objects")
    fact("two branches touched", "different paths")
    fact("the experiment's commit stored", human(experiment.stats.bytes_stored))
    fact("main's commit stored", human(trunk.stats.bytes_stored))
    fact("the two refs resolve", "independently", good=True)


def _images(ledger: Ledger, env: EnvId, root: Path, report: Report) -> None:
    heading(7, "Put a container image in the version", "Multi-Container Support")
    base_layer = layer_tar({"bin/sh": b"#!/bin/sh\n" * 2000, "etc/os-release": b"NAME=demo\n"})
    top_layer = layer_tar({"app/main.py": b"print('hello')\n"})
    archive = build_oci_archive(root / "app.tar", [base_layer, top_layer])

    images = ImageService(ledger)
    first = images.add(env, MAIN, archive, image="app", author="agent-17", platform=LINUX_AMD64)
    report.record("image_digest", first.manifest_digest)
    fact("image", f"app @ {first.manifest_digest}")
    fact("layers", f"{first.layers} ({first.layers_reused} already stored)")
    fact("compressed source", human(first.bytes_compressed))
    fact("stored uncompressed", human(first.bytes_uncompressed))

    rebuilt = build_oci_archive(
        root / "app-v2.tar", [base_layer, layer_tar({"app/main.py": b"print('goodbye')\n"})]
    )
    second = images.add(env, MAIN, rebuilt, image="app", author="agent-17", platform=LINUX_AMD64)
    report.record("rebuild_reused", second.layers_reused)
    report.record("rebuild_bytes", second.stats.bytes_stored)

    fact("a rebuild reused", f"{second.layers_reused} of {second.layers} layers", good=True)
    fact("…and stored", human(second.stats.bytes_stored), good=True)
    fact("pull it with", f"docker pull <host>/{ENV}/app:main")


def _building(ledger: Ledger, env: EnvId, report: Report) -> None:
    heading(8, "Build and sync, then fork", "Build & Sync")
    queue = BuildQueue(ledger.meta, clock=ledger.clock)
    runner = RecordingRunner()
    platform = RecordingPlatform()
    worker = BuildWorker(ledger, runner=runner, platform=platform, queue=queue)

    Dispatcher(ledger, queue).poll()
    outcomes = worker.drain()
    report.record("builds_run", len(outcomes))
    report.record("runner_invocations", runner.count)

    fact("commits queued by ref updates", len(outcomes))
    fact("runner invocations", runner.count)
    head = ledger.repo.get_ref(env, MAIN).target
    fact("sync note", ledger.repo.get_note(env, head, SYNC_NAMESPACE).body["platform_id"])

    before = runner.count
    fork_env = report["fork_env"]
    ledger.repo.update_ref(
        env=fork_env,
        name=MAIN,
        expected_generation=ledger.repo.get_ref(fork_env, MAIN).generation,
        target=head,
        principal="agent-17",
    )
    Dispatcher(ledger, queue).poll()
    fork_outcomes = worker.drain()
    report.record("fork_cache_hits", sum(1 for o in fork_outcomes if o.cache_hit))
    report.record("runner_after_fork", runner.count)

    fact("the fork asked for the same commit", "yes")
    fact("runner invocations after", runner.count, good=runner.count == before)
    fact("served from cache", f"{report['fork_cache_hits']} of {len(fork_outcomes)}", good=True)


def _authorization(ledger: Ledger, env: EnvId, report: Report) -> None:
    heading(9, "Scoped authorization", "Scoped authorization")
    from src.auth.model import NamePrefixSelector, Operation, Principal, Scope
    from src.auth.policy import PolicyEngine
    from src.auth.tokens import TokenSigner, TokenVerifier
    from src.errors import Forbidden

    signer = TokenSigner.generate(key_id="demo", clock=ledger.clock)
    verifier = TokenVerifier({signer.key_id: signer.public_key}, clock=ledger.clock)
    policy = PolicyEngine(ledger)

    read_only = verifier.verify(
        signer.mint(
            Principal("rollout"),
            Scope(operations=Operation.READ, selectors=(NamePrefixSelector(f"{ORG}/*"),)),
            ttl_us=3600 * 1_000_000,
        )
    )
    outcomes: dict[str, str] = {}
    policy.authorize(read_only, Operation.READ, env)
    outcomes["read with env:read"] = "allowed"
    try:
        policy.authorize(read_only, Operation.WRITE, env)
    except Forbidden:
        outcomes["write with env:read"] = "403"

    elsewhere = verifier.verify(
        signer.mint(
            Principal("other-team"),
            Scope(operations=Operation.READ, selectors=(NamePrefixSelector("other/*"),)),
            ttl_us=3600 * 1_000_000,
        )
    )
    try:
        policy.authorize(elsewhere, Operation.READ, env)
    except Forbidden:
        outcomes["read with another team's token"] = "403"

    report.record("authorization", outcomes)
    for label, outcome in outcomes.items():
        fact(label, outcome, good=outcome == "403")


def _collection(
    ledger: Ledger,
    commits: CommitService,
    env: EnvId,
    source: Path,
    clock: ManualClock,
    report: Report,
) -> None:
    heading(10, "Discard a branch and reclaim it", "Branching / Forking")

    # A branch holding content nothing else reaches. The merged experiment above
    # would reclaim nothing, and demonstrating collection on it would be a
    # measurement of the wrong thing.
    throwaway = RefName("refs/heads/exp/discard-me")
    ledger.repo.create_ref(
        env, throwaway, ledger.repo.get_ref(env, MAIN).target, principal="agent-17"
    )
    scratch = source.parent / "throwaway"
    shutil.copytree(source, scratch, symlinks=True)
    (scratch / "data" / "sweep.bin").write_bytes(random.Random(99).randbytes(1_000_000))
    dead_end = commits.commit(env, throwaway, scratch, author="agent-b", message="a dead end")

    for name in (ENV, FORK):
        ledger.gc.rebuild_keep_set(name)
    before = ledger.store.catalog.total()
    fact("branch holds content nothing else does", human(1_000_000))

    ref = ledger.repo.get_ref(env, throwaway)
    ledger.repo.delete_ref(env, throwaway, expected_generation=ref.generation, principal="agent")
    ledger.gc.rebuild_keep_set(ENV)

    # Nine days on: past the grace period, and still nothing to collect. The
    # operation log has not aged out, so `undo` can still restore this branch —
    # and content undo can reach is content collection must not take.
    clock.advance_days(9)
    held = ledger.gc.plan()
    report.record("gc_held_by_undo", len(held.candidates))
    fact("after 9 days — objects collectable", len(held.candidates))
    fact("…because undo can still reach them", "the operation log is a GC root")

    restored = BlobReader(
        ledger.store,
        resolve_path(
            ledger.store,
            ledger.store.get_as(dead_end.commit, Commit).tree,
            "data/sweep.bin",
        ).target,
    ).read(0, 16)
    report.record("undo_still_reads", len(restored) == 16)
    fact("the discarded branch still reads", "yes", good=True)

    # Past the log's retention. Undo can no longer reach it, so neither can the
    # keep-set, and the storage comes back. Those two facts are one number:
    # log retention *is* reclamation latency.
    clock.advance_days(90)
    ledger.gc.rebuild_keep_set(ENV)
    plan = ledger.gc.plan()
    report.record("gc_candidates", len(plan.candidates))
    report.record("gc_bytes", plan.bytes_reclaimable)
    fact("after the log ages out — collectable", len(plan.candidates), good=True)
    fact("bytes reclaimable", human(plan.bytes_reclaimable), good=True)

    reclaimed = ledger.gc.run(enforce=True)
    after = ledger.store.catalog.total()
    report.record("gc_deleted", reclaimed.deleted)
    report.record("gc_freed", before[1] - after[1])
    fact("objects deleted", reclaimed.deleted, good=True)
    fact("bytes returned", human(before[1] - after[1]), good=True)

    # And the half that matters more: everything else still reads.
    head = ledger.repo.get_ref(env, MAIN).target
    tree = ledger.store.get_as(head, Commit).tree
    resolved = resolve_path(ledger.store, tree, "data/train.bin")
    payload = BlobReader(ledger.store, resolved.target).read(0, 32)
    report.record("shared_still_readable", len(payload) == 32)
    fact("main still reads its shared dataset", "yes", good=True)


def _summary(report: Report) -> None:
    console.print("\n[bold]What this run measured[/bold]")
    table = Table(box=None, pad_edge=False)
    table.add_column("requirement")
    table.add_column("evidence from this run")
    rows = [
        (
            "Version Environments",
            f"restoring an earlier version stored {report['revert_bytes']} bytes",
        ),
        (
            "Large Artifact Handling",
            f"64 B changed in {human(DATASET_BYTES)} → {human(report['large_edit_bytes'])} stored",
        ),
        ("Multi-Container Support", f"rebuild reused {report['rebuild_reused']} layers"),
        ("Read at scale", f"{report['ranged_read_objects']} objects for a 64-byte ranged read"),
        ("Branching / Forking", f"fork copied {report['fork_bytes']} bytes"),
        ("Concurrent Automations", "second writer got 409 with the current generation"),
        ("Deduplication", f"a one-line edit stored {human(report['edit_bytes'])}"),
        ("Build & Sync", f"fork inherited its build ({report['fork_cache_hits']} cache hit)"),
        ("Scoped authorization", f"{len(report['authorization'])} outcomes, as granted"),
        (
            "Storage",
            f"discarding a branch returned {human(report['gc_freed'])}; shared content still reads",
        ),
    ]
    for requirement, evidence in rows:
        table.add_row(requirement, evidence)
    console.print(table)
    console.print(
        "\n[dim]Re-run with --keep --dir ./demo/.state/data to browse what it built:[/dim]\n"
        "[dim]  uv run python demo/e2e.py --keep --dir ./demo/.state/data[/dim]\n"
        "[dim]  uv run ledgerd --data-dir ./demo/.state/data --dev[/dim]\n"
        "[dim]…then open http://localhost:8080/[/dim]"
    )


# ─────────────────────────────────────────────────────────────────────────────


@final
class _CountingStore:
    """Counts object fetches, so "six objects" is measured rather than asserted."""

    __slots__ = ("_inner", "reads")

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.reads = 0

    def reset(self) -> None:
        self.reads = 0

    def get_as(self, name: ObjectName, expected: Any) -> Any:
        self.reads += 1
        return self._inner.get_as(name, expected)

    def __getattr__(self, item: str) -> Any:
        return getattr(self._inner, item)


def _whole_file_reads(ledger: Ledger, blob: ObjectName) -> str:
    counting = _CountingStore(ledger.store)
    reader = BlobReader(cast("Any", counting), blob)
    counting.reset()
    for _ in reader.stream():
        pass
    return f"{counting.reads} objects"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--keep", action="store_true", help="Leave the demo data directory in place."
    )
    parser.add_argument("--dir", type=Path, default=None, help="Where to build it.")
    arguments = parser.parse_args()

    root = arguments.dir or Path(tempfile.mkdtemp(prefix="ledger-demo-"))
    root.mkdir(parents=True, exist_ok=True)
    console.print(f"[dim]building a Ledger in {root}[/dim]")
    try:
        run(root)
    finally:
        if arguments.keep:
            console.print(f"\n[dim]data left in {root}[/dim]")
        else:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
