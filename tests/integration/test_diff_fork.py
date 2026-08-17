"""Diff and forking.

Two claims are asserted as measurements rather than described:

* **diff cost is proportional to what changed** — a counting store shows that
  comparing two versions of a large environment touches a handful of objects;
* **a fork copies zero bytes** — the object count before and after is identical.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import pytest

from src.clock import ManualClock
from src.format.cdc import ChunkParams
from src.format.model import Commit
from src.format.shape import ShapeParams
from src.fs.closure import commit_closure
from src.fs.diff import ChangeKind, diff_trees
from src.ids import EnvName, RefName
from src.instance import Ledger
from src.service.commits import CommitService
from src.service.environments import EnvironmentService

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from src.ids import EnvId, ObjectName

MAIN = RefName("refs/heads/main")
BRANCH = RefName("refs/heads/exp/a")


@pytest.fixture
def ledger(tmp_path: Path) -> Iterator[Ledger]:
    with Ledger(
        tmp_path / "ledger",
        clock=ManualClock(start_us=1_700_000_000_000_000),
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
    (root / "images").mkdir()
    (root / "harbor.yaml").write_text("name: demo\n")
    (root / "task" / "prompt.md").write_text("Solve the failing test.\n")
    (root / "task" / "verifier.py").write_text("import sys\nsys.exit(0)\n")
    (root / "data" / "train.bin").write_bytes(random.Random(42).randbytes(200_000))
    (root / "images" / "layer.tar").write_bytes(random.Random(7).randbytes(150_000))
    return root


class CountingStore:
    """Counts fetches, so cost claims become assertions."""

    def __init__(self, inner: object) -> None:
        self._inner = inner
        self.fetches = 0

    def __getattr__(self, item: str) -> object:
        return getattr(self._inner, item)

    def get_as(self, name: object, expected: type) -> object:
        self.fetches += 1
        return self._inner.get_as(name, expected)  # type: ignore[attr-defined]


def tree_of(ledger: Ledger, commit: ObjectName) -> ObjectName:
    return ledger.store.get_as(commit, Commit).tree


class TestDiff:
    def test_reports_added_removed_and_modified(
        self, ledger: Ledger, commits: CommitService, env: EnvId, source: Path
    ) -> None:
        first = commits.commit(env, MAIN, source, author="a", message="v1")

        (source / "task" / "prompt.md").write_text("A different prompt.\n")
        (source / "task" / "verifier.py").unlink()
        (source / "task" / "extra.md").write_text("new file\n")
        second = commits.commit(env, MAIN, source, author="a", message="v2")

        changes = list(
            diff_trees(ledger.store, tree_of(ledger, first.commit), tree_of(ledger, second.commit))
        )
        by_path = {c.display_path: c.kind for c in changes}
        assert by_path == {
            "task/prompt.md": ChangeKind.MODIFIED,
            "task/verifier.py": ChangeKind.REMOVED,
            "task/extra.md": ChangeKind.ADDED,
        }

    def test_identical_trees_produce_no_changes(
        self, ledger: Ledger, commits: CommitService, env: EnvId, source: Path
    ) -> None:
        first = commits.commit(env, MAIN, source, author="a", message="v1")
        tree = tree_of(ledger, first.commit)
        assert list(diff_trees(ledger.store, tree, tree)) == []

    def test_cost_is_proportional_to_the_change(
        self, ledger: Ledger, commits: CommitService, env: EnvId, source: Path
    ) -> None:
        """An unchanged subtree has an unchanged hash, so an entire
        branch is dismissed by comparing two 32-byte names.

        ``images/`` and ``data/`` hold most of the bytes and neither is touched,
        so neither should be descended into.
        """
        first = commits.commit(env, MAIN, source, author="a", message="v1")
        (source / "task" / "prompt.md").write_text("one line changed\n")
        second = commits.commit(env, MAIN, source, author="a", message="v2")

        counting = CountingStore(ledger.store)
        changes = list(
            diff_trees(counting, tree_of(ledger, first.commit), tree_of(ledger, second.commit))  # type: ignore[arg-type]
        )
        assert len(changes) == 1
        assert counting.fetches <= 6, (
            f"{counting.fetches} fetches to diff one changed file — untouched "
            f"subtrees should have been skipped by name"
        )

    def test_a_mode_change_is_a_modification(
        self, ledger: Ledger, commits: CommitService, env: EnvId, source: Path
    ) -> None:
        first = commits.commit(env, MAIN, source, author="a", message="v1")
        (source / "task" / "verifier.py").chmod(0o755)
        second = commits.commit(env, MAIN, source, author="a", message="chmod")

        changes = list(
            diff_trees(ledger.store, tree_of(ledger, first.commit), tree_of(ledger, second.commit))
        )
        assert [c.display_path for c in changes] == ["task/verifier.py"]
        assert changes[0].kind is ChangeKind.MODIFIED

    def test_a_whole_directory_added(
        self, ledger: Ledger, commits: CommitService, env: EnvId, source: Path
    ) -> None:
        first = commits.commit(env, MAIN, source, author="a", message="v1")
        (source / "extra").mkdir()
        (source / "extra" / "a.txt").write_text("a\n")
        (source / "extra" / "b.txt").write_text("b\n")
        second = commits.commit(env, MAIN, source, author="a", message="v2")

        changes = list(
            diff_trees(ledger.store, tree_of(ledger, first.commit), tree_of(ledger, second.commit))
        )
        assert sorted(c.display_path for c in changes) == ["extra/a.txt", "extra/b.txt"]
        assert all(c.kind is ChangeKind.ADDED for c in changes)


class TestFork:
    def test_a_fork_copies_zero_bytes(
        self, ledger: Ledger, commits: CommitService, env: EnvId, source: Path
    ) -> None:
        """Forking costs **0 objects, 0 bytes** — two metadata
        rows."""
        head = commits.commit(env, MAIN, source, author="a", message="v1")
        before = ledger.store.catalog.total()

        result = EnvironmentService(ledger).fork(
            env, EnvName("proximal/variant-a"), principal="agent-17"
        )

        assert ledger.store.catalog.total() == before, "a fork must copy no bytes"
        assert result.bytes_copied == 0
        assert result.source_commit == head.commit

    def test_the_fork_resolves_to_the_same_commit(
        self, ledger: Ledger, commits: CommitService, env: EnvId, source: Path
    ) -> None:
        head = commits.commit(env, MAIN, source, author="a", message="v1")
        result = EnvironmentService(ledger).fork(env, EnvName("proximal/variant-a"))

        forked = result.environment.env_id
        assert commits.resolve(forked, MAIN) == head.commit

    def test_provenance_is_recorded(
        self, ledger: Ledger, commits: CommitService, env: EnvId, source: Path
    ) -> None:
        head = commits.commit(env, MAIN, source, author="a", message="v1")
        result = EnvironmentService(ledger).fork(env, EnvName("proximal/variant-a"))

        record = ledger.repo.get_env(result.environment.env_id)
        assert record.forked_from_env == env
        assert record.forked_from_commit == head.commit

    def test_the_fork_and_its_parent_are_independent(
        self, ledger: Ledger, commits: CommitService, env: EnvId, source: Path
    ) -> None:
        commits.commit(env, MAIN, source, author="a", message="v1")
        forked = (
            EnvironmentService(ledger).fork(env, EnvName("proximal/variant-a")).environment.env_id
        )

        (source / "task" / "prompt.md").write_text("only in the fork\n")
        commits.commit(forked, MAIN, source, author="a", message="fork change")

        assert commits.resolve(env, MAIN) != commits.resolve(forked, MAIN)
        assert ledger.repo.get_ref(env, MAIN).generation == 1

    def test_the_forks_keep_set_records_the_closure(
        self, ledger: Ledger, commits: CommitService, env: EnvId, source: Path
    ) -> None:
        """Without this the fork's keep-set would be empty while its ref reached
        live content, and the first ``DeleteRef`` in the source environment would
        sweep objects the fork still needs.
        """
        commits.commit(env, MAIN, source, author="a", message="v1")
        result = EnvironmentService(ledger).fork(env, EnvName("proximal/variant-a"))
        assert result.closure_size > 5

    def test_the_closure_includes_chunk_names(
        self, ledger: Ledger, commits: CommitService, env: EnvId, source: Path
    ) -> None:
        """The "never chunks" means never *read*, not never *record*.

        Omitting chunk names would build a metadata-only keep-set — and total,
        silent data loss for forked environments on the first sweep.
        """
        from src.format.model import Chunk

        head = commits.commit(env, MAIN, source, author="a", message="v1")
        closure = commit_closure(ledger.store, head.commit)

        kinds = [type(ledger.store.get_object(n)).__name__ for n in closure]
        assert Chunk.__name__ in kinds, "chunk digests must be recorded in the closure"
